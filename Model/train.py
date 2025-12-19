# train_physics_unet_pbfm_ddp.py

import argparse
import csv
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector

from dataset import FDTDDataset
from physics_unet import PhysicsUNet, HelmholtzResidual2D
from flow_matching import psi_t, u_t, sample_t, cfm_loss_residual, sample as fm_sample

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.preferred_linalg_library("cusolver")

dtype = torch.float32


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def create_logger(logging_dir: str | None):
    if is_rank0():
        assert logging_dir is not None
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logging_dir, "log.txt")),
            ],
        )
        return logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger


def ddp_allreduce_mean_(x: torch.Tensor) -> torch.Tensor:
    """In-place mean across ranks."""
    if dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def broadcast_object(obj, src=0):
    """Broadcast a picklable python object from src to all ranks."""
    if not dist.is_initialized():
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def main(args):
    # -----------------------
    # DDP init
    # -----------------------
    assert torch.cuda.is_available(), "Need CUDA GPUs for DDP training."
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Deterministic-ish seeding per rank
    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------
    # experiment dirs / logging (rank 0 only)
    # -----------------------
    results_dir = "logs_physics_unet_pbfm"
    experiment_dir = os.path.join(results_dir, args.version)
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")

    if is_rank0():
        os.makedirs(results_dir, exist_ok=True)

        if (not args.resume_from) and os.path.exists(experiment_dir):
            # clear old artifacts
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    dist.barrier()
    logger = create_logger(experiment_dir if is_rank0() else None)
    if is_rank0():
        logger.info(f"DDP: rank={rank}/{world}, local_rank={local_rank}, device={device}")
    logger.info(f"Experiment dir: {experiment_dir}")

    # CSV header (rank 0 only)
    val_csv_path = os.path.join(experiment_dir, "validation.csv")
    if is_rank0():
        mode = "a" if (args.resume_from and os.path.exists(val_csv_path)) else "w"
        with open(val_csv_path, mode, encoding="UTF8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            if mode == "w":
                writer.writerow([
                    "epoch",
                    "val_fm_loss",
                    "val_residual_loss",
                    "val_phase_loss",
                    "val_endpoint_loss",
                    "val_phase_grad_loss",
                    "sample_residual_mean"
                ])

    # wandb (rank 0 only)
    wandb_run = None
    if args.use_wandb and is_rank0():
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.version,
            config=vars(args),
        )

    # -----------------------
    # Dataset + stats broadcast
    # -----------------------
    stats = None
    if is_rank0():
        train_ds_tmp = FDTDDataset(
            root_dir=args.data_root,
            split="train",
            train_fraction=args.train_fraction,
            normalize_eps=args.normalize_eps,
        )
        stats = train_ds_tmp.get_stats()
    stats = broadcast_object(stats, src=0)

    train_ds = FDTDDataset(
        root_dir=args.data_root,
        split="train",
        train_fraction=args.train_fraction,
        stats=stats,
        normalize_eps=args.normalize_eps,
    )
    val_ds = FDTDDataset(
        root_dir=args.data_root,
        split="val",
        train_fraction=args.train_fraction,
        stats=stats,
        normalize_eps=args.normalize_eps,
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )

    # GLOBAL batch size semantics
    assert args.batch_size % world == 0, f"--batch-size (global) must be divisible by world_size={world}"
    per_gpu_batch = args.batch_size // world

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, per_gpu_batch),
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    if is_rank0():
        logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
        logger.info(f"Global batch={args.batch_size}, per_gpu_batch={per_gpu_batch}, world={world}")

    # -----------------------
    # Model / DDP
    # -----------------------
    dx = args.dx
    lam0 = args.lambda_um
    omega = 2.0 * math.pi / lam0

    base_model = PhysicsUNet(
        in_channels=3,
        out_channels=2,
        model_channels=args.hidden_size,
        num_res_blocks=3,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        cond_dim=1,
        dx=dx,
        dy=dx,
        omega=omega,
    ).to(device)

    base_model.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    ema = deepcopy(base_model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_rank0():
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6, last_epoch=-1)

    pml_cells = 30
    helmholtz_op = HelmholtzResidual2D(
        dx=args.dx, dy=args.dx, omega=omega, pml_cells=pml_cells, normalize=False
    ).to(device)

    update_ema(ema, model.module, decay=0.0)

    # -----------------------
    # Resume
    # -----------------------
    start_epoch = 1
    if args.resume_from:
        if is_rank0():
            logger.info(f"Resuming from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint["model"], strict=True)
            ema.load_state_dict(checkpoint["ema"], strict=True)
            opt.load_state_dict(checkpoint["opt"])

            model.module.set_normalization_stats(stats, normalize_eps=args.normalize_eps)
            ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

        try:
            ckpt_name = os.path.basename(args.resume_from)
            last_epoch = int(os.path.splitext(ckpt_name)[0])
        except Exception:
            last_epoch = 0
        start_epoch = last_epoch + 1
        scheduler.last_epoch = last_epoch

    # -----------------------
    # Train
    # -----------------------
    use_config = bool(args.use_residual)
    use_endpoint = args.lambda_endpoint > 0
    config_failed = False

    lam_mean = float(stats["lambda_um_mean"])
    lam_std = float(stats["lambda_um_std"])

    running_fm_loss = 0.0
    running_residual_loss = 0.0
    running_phase_loss = 0.0
    running_endpoint_loss = 0.0
    running_phase_grad_loss = 0.0
    train_steps = 0
    start_time = time()

    if is_rank0():
        logger.info(f"Training for {args.epochs} epochs, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        residual_weight = args.lambda_residual
        if args.residual_warmup_epochs > 0:
            residual_weight *= min(1.0, epoch / args.residual_warmup_epochs)

        phase_weight = args.lambda_phase
        if args.phase_warmup_epochs > 0:
            phase_weight *= min(1.0, epoch / args.phase_warmup_epochs)

        endpoint_weight = args.lambda_endpoint
        if args.endpoint_warmup_epochs > 0 and endpoint_weight > 0:
            endpoint_weight *= min(1.0, epoch / args.endpoint_warmup_epochs)

        phase_grad_weight = args.lambda_phase_grad
        if args.phase_grad_warmup_epochs > 0:
            phase_grad_weight *= min(1.0, epoch / args.phase_grad_warmup_epochs)

        # device-focus schedule
        if args.device_focus_warmup_epochs > 0:
            ramp = min(1.0, epoch / args.device_focus_warmup_epochs)
        else:
            ramp = 1.0
        device_focus = args.device_focus_max * ramp
        if args.device_focus_hold_epochs > 0:
            if epoch <= args.device_focus_warmup_epochs + args.device_focus_hold_epochs:
                device_focus = args.device_focus_max
        if args.device_focus_decay_epochs > 0:
            e0 = args.device_focus_warmup_epochs + args.device_focus_hold_epochs
            if epoch > e0:
                frac = min(1.0, (epoch - e0) / args.device_focus_decay_epochs)
                device_focus = args.device_focus_max * (1.0 - frac)

        for x_full, cond in train_loader:
            x_full = x_full.to(device, dtype=dtype, non_blocking=True)
            cond = cond.to(device, dtype=dtype, non_blocking=True)

            fields_1 = x_full[:, 0:2]
            eps = x_full[:, 2:3]
            lambda_um = cond * lam_std + lam_mean  # [B,1] physical

            x0_fields = torch.randn_like(fields_1)
            t = sample_t(fields_1)  # [B,1,1,1]
            x_t_fields = psi_t(x0_fields, fields_1, t)
            v_t_fields = u_t(x0_fields, fields_1)
            x_t_input = torch.cat([x_t_fields, eps], dim=1)

            fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad_loss = cfm_loss_residual(
                model,
                x_t_input,
                t,
                v_t_fields,
                helmholtz_op,
                stats,
                args.normalize_eps,
                fields_1,
                cond=cond,
                lambda_um=lambda_um,
                device_focus=device_focus,
                w_min=args.w_min,
                eps_thr=args.eps_thr,
                dilate=args.dilate,
                weight_residual=True,
            )

            # scale residual to grid units
            residual_loss = residual_loss * (args.dx ** 4)

            weighted_residual_loss = residual_weight * residual_loss
            weighted_phase_loss = phase_weight * phase_loss
            weighted_endpoint_loss = endpoint_weight * endpoint_loss
            weighted_phase_grad_loss = phase_grad_weight * phase_grad_loss

            if use_config and (not config_failed):
                opt.zero_grad(set_to_none=True)
                fm_loss.backward(retain_graph=True)
                g1 = get_gradient_vector(model.module).detach()
                opt.zero_grad(set_to_none=True)

                weighted_residual_loss.backward(retain_graph=True)
                g2 = get_gradient_vector(model.module).detach()
                opt.zero_grad(set_to_none=True)

                weighted_phase_loss.backward(retain_graph=True)
                g3 = get_gradient_vector(model.module).detach()
                opt.zero_grad(set_to_none=True)

                if endpoint_weight > 0:
                    weighted_endpoint_loss.backward(retain_graph=True)
                    g4 = get_gradient_vector(model.module).detach()
                    opt.zero_grad(set_to_none=True)
                else:
                    g4 = torch.zeros_like(g1)

                weighted_phase_grad_loss.backward(retain_graph=True)
                g5 = get_gradient_vector(model.module).detach()
                opt.zero_grad(set_to_none=True)

                ddp_allreduce_mean_(g1)
                ddp_allreduce_mean_(g2)
                ddp_allreduce_mean_(g3)
                ddp_allreduce_mean_(g4)
                ddp_allreduce_mean_(g5)

                valid = all(torch.isfinite(g).all() for g in (g1, g2, g3, g4, g5))
                if not valid:
                    if is_rank0():
                        logger.warning("Non-finite ConFIG grads; falling back to summed grads.")
                    g_sum = torch.zeros_like(g1)
                    for g in (g1, g2, g3, g4, g5):
                        if torch.isfinite(g).all():
                            g_sum += g
                    apply_gradient_vector(model.module, g_sum)
                else:
                    try:
                        g_cfg = ConFIG_update([g1, g2, g3, g4, g5])
                        if not torch.isfinite(g_cfg).all():
                            raise ValueError("ConFIG produced non-finite grad")
                        apply_gradient_vector(model.module, g_cfg)
                    except Exception as e:
                        if is_rank0():
                            logger.warning(f"ConFIG failed: {e}; switching to summed grads permanently.")
                        config_failed = True
                        apply_gradient_vector(model.module, (g1 + g2 + g3 + g4 + g5))

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            else:
                total_loss = (
                    fm_loss
                    + weighted_residual_loss
                    + weighted_phase_loss
                    + weighted_endpoint_loss
                    + weighted_phase_grad_loss
                )
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            update_ema(ema, model.module)

            running_fm_loss += fm_loss.item()
            running_residual_loss += residual_loss.item()
            running_phase_loss += phase_loss.item()
            running_endpoint_loss += endpoint_loss.item()
            running_phase_grad_loss += phase_grad_loss.item()
            train_steps += 1

        # -----------------------
        # Validation (DDP-averaged)
        # -----------------------
        if epoch % args.eval_every == 0:
            model.eval()
            ema.eval()

            with torch.no_grad():
                eval_fm = 0.0
                eval_res = 0.0
                eval_phase = 0.0
                eval_endpoint = 0.0
                eval_phase_grad = 0.0
                eval_steps = 0

                for x_full, cond in val_loader:
                    x_full = x_full.to(device, dtype=dtype, non_blocking=True)
                    cond = cond.to(device, dtype=dtype, non_blocking=True)

                    fields_1 = x_full[:, 0:2]
                    eps = x_full[:, 2:3]
                    lambda_um = cond * lam_std + lam_mean

                    x0_fields = torch.randn_like(fields_1)
                    t = sample_t(fields_1)
                    x_t_fields = psi_t(x0_fields, fields_1, t)
                    v_t_fields = u_t(x0_fields, fields_1)
                    x_t_input = torch.cat([x_t_fields, eps], dim=1)

                    fm_v, res_v, ph_v, end_v, phg_v = cfm_loss_residual(
                        ema,
                        x_t_input,
                        t,
                        v_t_fields,
                        helmholtz_op,
                        stats,
                        args.normalize_eps,
                        fields_1,
                        cond=cond,
                        lambda_um=lambda_um,
                        device_focus=device_focus,
                        w_min=args.w_min,
                        eps_thr=args.eps_thr,
                        dilate=args.dilate,
                        weight_residual=True,
                    )

                    res_v = res_v * (args.dx ** 4)

                    eval_fm += fm_v.item()
                    eval_res += res_v.item()
                    eval_phase += ph_v.item()
                    eval_endpoint += end_v.item()
                    eval_phase_grad += phg_v.item()
                    eval_steps += 1
                    break  # keep it fast

                fm_t = torch.tensor(eval_fm / max(eval_steps, 1), device=device)
                res_t = torch.tensor(eval_res / max(eval_steps, 1), device=device)
                ph_t = torch.tensor(eval_phase / max(eval_steps, 1), device=device)
                end_t = torch.tensor(eval_endpoint / max(eval_steps, 1), device=device)
                phg_t = torch.tensor(eval_phase_grad / max(eval_steps, 1), device=device)

                ddp_allreduce_mean_(fm_t)
                ddp_allreduce_mean_(res_t)
                ddp_allreduce_mean_(ph_t)
                ddp_allreduce_mean_(end_t)
                ddp_allreduce_mean_(phg_t)

                val_fm_loss = float(fm_t.item())
                val_res_loss = float(res_t.item())
                val_phase_loss = float(ph_t.item())
                val_endpoint_loss = float(end_t.item())
                val_phase_grad_loss = float(phg_t.item())

            # -----------------------
            # Sample residual eval + images (rank 0 only)
            # -----------------------
            sample_residual_mean = 0.0
            if is_rank0():
                residuals = []
                with torch.no_grad():
                    n_samples = min(len(train_ds), args.sample_eval_limit)
                    for s in range(n_samples):
                        x_full_s, cond_s = train_ds[s]
                        x_full_s = x_full_s.unsqueeze(0).to(device, dtype=dtype)
                        cond_s = cond_s.unsqueeze(0).to(device, dtype=dtype)

                        eps_s = x_full_s[:, 2:3]
                        x0_fields_s = torch.randn_like(x_full_s[:, 0:2])

                        lambda_um_s = cond_s * lam_std + lam_mean  # [1,1]

                        x1_fields_pred = fm_sample(
                            ema,
                            x0_fields_s,
                            num_steps=args.fm_steps,
                            use_stoc_samp=args.use_stoc_samp,
                            cond_eps=eps_s,
                            cond=cond_s,
                            lambda_um=lambda_um_s,
                        )
                        x1_pred = torch.cat([x1_fields_pred, eps_s], dim=1)

                        # denorm to physical
                        x1_pred[:, 0] = x1_pred[:, 0] * stats["ez_real_std"] + stats["ez_real_mean"]
                        x1_pred[:, 1] = x1_pred[:, 1] * stats["ez_imag_std"] + stats["ez_imag_mean"]
                        if args.normalize_eps:
                            x1_pred[:, 2] = x1_pred[:, 2] * stats["eps_std"] + stats["eps_mean"]

                        B = x1_pred.shape[0]
                        k0 = (2.0 * torch.pi) / lambda_um_s.view(B)
                        R = helmholtz_op(x1_pred[:, 0:2], x1_pred[:, 2:3], k0=k0)
                        res_mag = torch.sqrt(R[:, 0:1] ** 2 + R[:, 1:2] ** 2 + 1e-12) * (args.dx**2)
                        residuals.append(res_mag.mean().item())

                        # Save images on s==0 (same as your original)
                        if s == 0:
                            sample_dir = os.path.join(experiment_dir, "samples")
                            os.makedirs(sample_dir, exist_ok=True)

                            x_gt = x_full_s.clone()
                            x_gt[:, 0] = x_gt[:, 0] * stats["ez_real_std"] + stats["ez_real_mean"]
                            x_gt[:, 1] = x_gt[:, 1] * stats["ez_imag_std"] + stats["ez_imag_mean"]
                            if args.normalize_eps:
                                x_gt[:, 2] = x_gt[:, 2] * stats["eps_std"] + stats["eps_mean"]

                            eps_img = x_gt[0, 2].detach().cpu().numpy()
                            ezr_gt_img = x_gt[0, 0].detach().cpu().numpy()
                            ezi_gt_img = x_gt[0, 1].detach().cpu().numpy()

                            ezr_pred_img = x1_pred[0, 0].detach().cpu().numpy()
                            ezi_pred_img = x1_pred[0, 1].detach().cpu().numpy()

                            mag_gt = np.sqrt(ezr_gt_img**2 + ezi_gt_img**2)
                            mag_pred = np.sqrt(ezr_pred_img**2 + ezi_pred_img**2)
                            mag_err = np.sqrt((ezr_pred_img - ezr_gt_img) ** 2 + (ezi_pred_img - ezi_gt_img) ** 2)

                            res_img = res_mag[0, 0].detach().cpu().numpy()

                            Hh, Ww = res_img.shape
                            p2 = min(pml_cells + 2, max(0, Hh // 2 - 1), max(0, Ww // 2 - 1))
                            mask = np.ones((Hh, Ww), dtype=np.float32)
                            if p2 > 0:
                                mask[:p2, :] = 0
                                mask[-p2:, :] = 0
                                mask[:, :p2] = 0
                                mask[:, -p2:] = 0
                            res_masked = np.ma.masked_where(mask <= 0.0, res_img)
                            vals = res_masked.compressed()
                            if vals.size > 0:
                                v_hi = float(np.quantile(vals, 0.99))
                                v_lo = float(np.quantile(vals, 0.05))
                                v_hi = max(v_hi, 1e-12)
                                v_lo = max(min(v_lo, v_hi * 0.5), v_hi * 1e-4, 1e-12)
                            else:
                                v_hi, v_lo = 1.0, 1e-6

                            extent = (0.0, Ww * args.dx, 0.0, Hh * args.dx)

                            def _imshow(ax, arr, title, cmap="magma", norm=None, vmin=None, vmax=None):
                                h = ax.imshow(
                                    arr,
                                    origin="lower",
                                    extent=extent,
                                    aspect="equal",
                                    cmap=cmap,
                                    norm=norm,
                                    vmin=vmin,
                                    vmax=vmax,
                                )
                                ax.set_title(title, fontsize=10)
                                ax.set_xlabel("x (µm)")
                                ax.set_ylabel("y (µm)")
                                ax.set_xticks([extent[0], extent[1]])
                                ax.set_yticks([extent[2], extent[3]])
                                return h

                            vmax_ezr = float(max(np.max(np.abs(ezr_gt_img)), np.max(np.abs(ezr_pred_img)), 1e-12))
                            vmax_ezi = float(max(np.max(np.abs(ezi_gt_img)), np.max(np.abs(ezi_pred_img)), 1e-12))

                            fig, axes = plt.subplots(2, 5, figsize=(18, 7))
                            axes = axes.reshape(2, 5)
                            fig.suptitle(
                                f"epoch {epoch:04d} | sample={s} | λ={float(lambda_um_s.item()):.4f} µm",
                                fontsize=11,
                            )

                            h0 = _imshow(axes[0, 0], eps_img, "eps (input)", cmap="viridis")
                            h1 = _imshow(axes[0, 1], ezr_gt_img, "Ez_real (GT)", cmap="RdBu", vmin=-vmax_ezr, vmax=vmax_ezr)
                            h2 = _imshow(axes[0, 2], ezi_gt_img, "Ez_imag (GT)", cmap="RdBu", vmin=-vmax_ezi, vmax=vmax_ezi)
                            h3 = _imshow(axes[0, 3], mag_gt, "|Ez| (GT)", cmap="magma")
                            h4 = _imshow(axes[0, 4], res_masked, "|R(Ez, eps)| (pred)", cmap="magma", norm=LogNorm(vmin=v_lo, vmax=v_hi))

                            h5 = _imshow(axes[1, 0], ezr_pred_img, "Ez_real (pred)", cmap="RdBu", vmin=-vmax_ezr, vmax=vmax_ezr)
                            h6 = _imshow(axes[1, 1], ezi_pred_img, "Ez_imag (pred)", cmap="RdBu", vmin=-vmax_ezi, vmax=vmax_ezi)
                            h7 = _imshow(axes[1, 2], mag_pred, "|Ez| (pred)", cmap="magma")
                            h8 = _imshow(axes[1, 3], mag_err, "|err|", cmap="magma")
                            axes[1, 4].axis("off")

                            for ax, h in [
                                (axes[0, 0], h0),
                                (axes[0, 1], h1),
                                (axes[0, 2], h2),
                                (axes[0, 3], h3),
                                (axes[0, 4], h4),
                                (axes[1, 0], h5),
                                (axes[1, 1], h6),
                                (axes[1, 2], h7),
                                (axes[1, 3], h8),
                            ]:
                                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

                            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
                            plt.savefig(os.path.join(sample_dir, f"sample_epoch_{epoch:04d}.png"), dpi=200)
                            plt.close(fig)

                            # Save arrays
                            np.savez_compressed(
                                os.path.join(sample_dir, f"sample_epoch_{epoch:04d}.npz"),
                                eps=eps_img,
                                ez_real_gt=ezr_gt_img,
                                ez_imag_gt=ezi_gt_img,
                                ez_real_pred=ezr_pred_img,
                                ez_imag_pred=ezi_pred_img,
                                mag_gt=mag_gt,
                                mag_pred=mag_pred,
                                mag_err=mag_err,
                                residual_mag=res_img,
                                dx=float(args.dx),
                                wavelength_um=float(lambda_um_s.item()),
                            )

                sample_residual_mean = float(np.mean(residuals)) if residuals else 0.0

                log_msg = (
                    f"[epoch {epoch:04d}] val_fm={val_fm_loss:.4e}, val_res={val_res_loss:.4e}, "
                    f"val_phase={val_phase_loss:.4e}, val_phase_grad={val_phase_grad_loss:.4e}, "
                    f"sample_residual={sample_residual_mean:.4e}"
                )
                if use_endpoint:
                    log_msg = log_msg.replace(
                        "val_phase_grad",
                        f"val_endpoint={val_endpoint_loss:.4e}, val_phase_grad"
                    )
                logger.info(log_msg)

                with open(val_csv_path, "a", encoding="UTF8", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    row = [
                        epoch,
                        val_fm_loss,
                        val_res_loss,
                        val_phase_loss,
                        val_endpoint_loss if use_endpoint else "",
                        val_phase_grad_loss,
                        sample_residual_mean
                    ]
                    writer.writerow(row)

                if wandb_run is not None:
                    log_dict = {
                        "epoch": epoch,
                        "val/fm_loss": val_fm_loss,
                        "val/residual_loss": val_res_loss,
                        "val/phase_loss": val_phase_loss,
                        "val/phase_grad_loss": val_phase_grad_loss,
                        "val/sample_residual": sample_residual_mean,
                    }
                    if use_endpoint:
                        log_dict["val/endpoint_loss"] = val_endpoint_loss
                    wandb_run.log(log_dict)

        # -----------------------
        # Train logging (DDP-averaged)
        # -----------------------
        if epoch % args.log_every == 0:
            torch.cuda.synchronize()
            sec_per_epoch = time() - start_time

            fm_mean = torch.tensor(running_fm_loss / max(train_steps, 1), device=device)
            res_mean = torch.tensor(running_residual_loss / max(train_steps, 1), device=device)
            ph_mean = torch.tensor(running_phase_loss / max(train_steps, 1), device=device)
            end_mean = torch.tensor(running_endpoint_loss / max(train_steps, 1), device=device)
            phg_mean = torch.tensor(running_phase_grad_loss / max(train_steps, 1), device=device)

            ddp_allreduce_mean_(fm_mean)
            ddp_allreduce_mean_(res_mean)
            ddp_allreduce_mean_(ph_mean)
            ddp_allreduce_mean_(end_mean)
            ddp_allreduce_mean_(phg_mean)

            if is_rank0():
                msg = (
                    f"(epoch={epoch:04d}) train_fm={fm_mean.item():.4e}, "
                    f"train_residual={res_mean.item():.4e}, train_phase={ph_mean.item():.4e}, "
                    f"train_phase_grad={phg_mean.item():.4e}, sec_per_epoch={sec_per_epoch:.3e}"
                )
                if use_endpoint:
                    msg = msg.replace(
                        "train_phase_grad",
                        f"train_endpoint={end_mean.item():.4e}, train_phase_grad"
                    )
                logger.info(msg)

            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train/fm_loss": fm_mean.item(),
                    "train/residual_loss": res_mean.item(),
                    "train/phase_loss": ph_mean.item(),
                    "train/phase_grad_loss": phg_mean.item(),
                    "train/sec_per_epoch": sec_per_epoch,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/phase_weight": phase_weight,
                    "train/residual_weight": residual_weight,
                    "train/endpoint_weight": endpoint_weight,
                    "train/phase_grad_weight": phase_grad_weight,
                    "train/device_focus": device_focus,
                    "train/use_config": int(use_config and (not config_failed)),
                }
                if use_endpoint:
                    log_dict["train/endpoint_loss"] = end_mean.item()
                wandb_run.log(log_dict)

            start_time = time()
            running_fm_loss = 0.0
            running_residual_loss = 0.0
            running_phase_loss = 0.0
            running_endpoint_loss = 0.0
            running_phase_grad_loss = 0.0
            train_steps = 0

        scheduler.step()

        # -----------------------
        # Checkpoint (rank 0 only)
        # -----------------------
        if epoch % args.ckpt_every == 0:
            if is_rank0():
                ckpt_path = os.path.join(ckpt_dir, f"{epoch:07d}.pt")
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "stats": stats,
                }
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()

    if is_rank0():
        logger.info("Done.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data_generation/data")
    parser.add_argument("--train-fraction", type=float, default=0.8)

    # IMPORTANT: treat as GLOBAL batch size for DDP
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-seed", type=int, default=15)

    parser.add_argument("--version", type=str, default="physics_unet_pbfm_ddp")
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--dx", type=float, default=1.0 / 30.0)
    parser.add_argument("--lambda-um", type=float, default=1.55)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--use-residual", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--residual-warmup-epochs", type=int, default=50)

    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--phase-warmup-epochs", type=int, default=50)

    parser.add_argument("--lambda-endpoint", type=float, default=0.0)
    parser.add_argument("--endpoint-warmup-epochs", type=int, default=0)

    # NEW: phase-gradient loss (kills drift)
    parser.add_argument("--lambda-phase-grad", type=float, default=0.2)
    parser.add_argument("--phase-grad-warmup-epochs", type=int, default=100)

    parser.add_argument("--normalize-eps", type=bool, default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--sample-eval-limit", type=int, default=16)

    parser.add_argument("--resume-from", type=str, default="")

    parser.add_argument("--use-wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project", type=str, default="Rayfield")
    parser.add_argument("--wandb-entity", type=str, default=None)

    parser.add_argument("--device-focus-max", type=float, default=3.0)
    parser.add_argument("--device-focus-warmup-epochs", type=int, default=100)
    parser.add_argument("--device-focus-hold-epochs", type=int, default=0)
    parser.add_argument("--device-focus-decay-epochs", type=int, default=200)
    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--eps-thr", type=float, default=6.0)
    parser.add_argument("--dilate", type=int, default=9)

    args = parser.parse_args()
    main(args)
