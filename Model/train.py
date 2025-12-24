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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

import wandb

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


def ddp_barrier(device: torch.device | None = None):
    """
    Barrier with explicit device_ids when using NCCL.
    This avoids NCCL hangs/errors when rank->GPU mapping is heterogeneous.
    """
    if not dist.is_initialized():
        return
    try:
        if device is not None and device.type == "cuda" and device.index is not None:
            dist.barrier(device_ids=[int(device.index)])
        else:
            dist.barrier()
    except TypeError:
        # Older torch versions don't accept device_ids
        dist.barrier()


def broadcast_object(obj, src=0):
    """Broadcast a picklable python object from src to all ranks."""
    if not dist.is_initialized():
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def ramp_linear(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    """
    Linear ramp from 0 to 1 starting at start_epoch (exclusive),
    reaching 1 after warmup_epochs.
    """
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch) / float(warmup_epochs))


def _norm(a: torch.Tensor) -> float:
    return float(torch.linalg.norm(a).item())


def _finite_frac(x: torch.Tensor) -> float:
    return float(torch.isfinite(x).float().mean().item())


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    na = torch.linalg.norm(a).clamp_min(eps)
    nb = torch.linalg.norm(b).clamp_min(eps)
    return float((torch.dot(a.flatten(), b.flatten()) / (na * nb)).item())


def _move_aux_to_device(aux, device: torch.device):
    """
    aux comes from DataLoader collate. Depending on how the dataset returns fields,
    dict values might be tensors, lists, numpy arrays, or None.
    We normalize it to a dict with tensors on device where applicable.
    """
    if not isinstance(aux, dict):
        return None

    out = dict(aux)

    # port_masks: float tensor [B, P, H, W] (or None)
    if out.get("port_masks", None) is not None:
        if not torch.is_tensor(out["port_masks"]):
            out["port_masks"] = torch.as_tensor(out["port_masks"])
        out["port_masks"] = out["port_masks"].to(device, dtype=dtype, non_blocking=True)

    # port_ids: typically long [B, P] or similar
    if out.get("port_ids", None) is not None:
        if not torch.is_tensor(out["port_ids"]):
            out["port_ids"] = torch.as_tensor(out["port_ids"])
        out["port_ids"] = out["port_ids"].to(device, non_blocking=True)

    # sparams_true: complex [B, ...]
    if out.get("sparams_true", None) is not None:
        if not torch.is_tensor(out["sparams_true"]):
            out["sparams_true"] = torch.as_tensor(out["sparams_true"])
        out["sparams_true"] = out["sparams_true"].to(device, non_blocking=True)

    # in_port_idx: should be per-sample; collate might produce list[int] -> make tensor [B]
    if out.get("in_port_idx", None) is not None:
        if torch.is_tensor(out["in_port_idx"]):
            out["in_port_idx"] = out["in_port_idx"].to(device, non_blocking=True)
        else:
            # could be list[int] or numpy
            out["in_port_idx"] = torch.as_tensor(out["in_port_idx"], dtype=torch.long, device=device)

    return out


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

    ddp_barrier(device)
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
                    "val_sparam_loss",
                    "sample_residual_mean",
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
    include_sweeps = None
    if args.include_sweeps:
        include_sweeps = [s.strip() for s in args.include_sweeps.split(",") if s.strip()]

    def _parse_kv_ints(s: str) -> dict[str, int]:
        out: dict[str, int] = {}
        s = (s or "").strip()
        if not s:
            return out
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected entries like name=3000, got: {part}")
            k, v = part.split("=", 1)
            out[k.strip()] = int(v.strip())
        return out

    subset_train = _parse_kv_ints(getattr(args, "subset_train_per_sweep", ""))
    subset_val = _parse_kv_ints(getattr(args, "subset_val_per_sweep", ""))
    subset_seed = int(getattr(args, "subset_seed", 0))

    # If user specifies train subset but not val, derive val per sweep by ratio.
    if subset_train and (not subset_val):
        val_total = int(getattr(args, "subset_val_total", 0))
        train_total = sum(subset_train.values())
        if val_total <= 0:
            # default to 25% of train (matches 4000->1000)
            val_total = max(1, int(round(0.25 * train_total)))
        # preserve per-sweep ratios
        for k, n in subset_train.items():
            subset_val[k] = int(round(val_total * (n / max(train_total, 1))))
        # fix rounding drift so totals match exactly
        drift = val_total - sum(subset_val.values())
        if drift != 0 and subset_val:
            keys = sorted(subset_val.keys())
            step = 1 if drift > 0 else -1
            for i in range(abs(drift)):
                subset_val[keys[i % len(keys)]] += step

    stats = None
    if is_rank0():
        train_ds_tmp = FDTDDataset(
            root_dir=args.data_root,
            split="train",
            train_fraction=args.train_fraction,
            normalize_eps=args.normalize_eps,
            use_shards=args.use_shards,
            shard_subdir=args.shard_subdir,
            shard_index_name=args.shard_index_name,
            include_sweeps=include_sweeps,
            subset_train_per_sweep=subset_train if subset_train else None,
            subset_val_per_sweep=subset_val if subset_train or subset_val else None,
            subset_seed=subset_seed,
        )
        stats = train_ds_tmp.get_stats()
    stats = broadcast_object(stats, src=0)

    train_ds = FDTDDataset(
        root_dir=args.data_root,
        split="train",
        train_fraction=args.train_fraction,
        stats=stats,
        normalize_eps=args.normalize_eps,
        use_shards=args.use_shards,
        shard_subdir=args.shard_subdir,
        shard_index_name=args.shard_index_name,
        include_sweeps=include_sweeps,
        return_aux=True,
        subset_train_per_sweep=subset_train if subset_train else None,
        subset_val_per_sweep=subset_val if subset_train or subset_val else None,
        subset_seed=subset_seed,
    )
    val_ds = FDTDDataset(
        root_dir=args.data_root,
        split="val",
        train_fraction=args.train_fraction,
        stats=stats,
        normalize_eps=args.normalize_eps,
        use_shards=args.use_shards,
        shard_subdir=args.shard_subdir,
        shard_index_name=args.shard_index_name,
        include_sweeps=include_sweeps,
        return_aux=True,
        subset_train_per_sweep=subset_train if subset_train else None,
        subset_val_per_sweep=subset_val if subset_train or subset_val else None,
        subset_seed=subset_seed,
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

    train_prefetch = args.prefetch_factor if args.train_num_workers > 0 else 2
    val_prefetch = args.prefetch_factor if args.val_num_workers > 0 else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.train_num_workers,
        prefetch_factor=train_prefetch,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.train_num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, per_gpu_batch),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.val_num_workers,
        prefetch_factor=val_prefetch,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.val_num_workers > 0,
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
        in_channels=4,
        out_channels=2,
        model_channels=args.hidden_size,
        num_res_blocks=4,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        cond_dim=int(getattr(train_ds, "cond_dim", 1)),
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
    scaler = GradScaler("cuda", enabled=bool(getattr(args, "amp", True)))
    amp_enabled = scaler.is_enabled()

    def optimizer_step():
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

    # Smooth LR schedule: linear warmup -> cosine decay (no phase jumps).
    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    min_lr = float(getattr(args, "min_lr", 5e-6))
    min_lr = max(min_lr, 0.0)

    if warmup_epochs > 0:
        # Ramp from warmup_start_factor * lr to lr over warmup_epochs epochs.
        warmup_start_factor = float(getattr(args, "warmup_start_factor", 0.1))
        warmup_start_factor = min(max(warmup_start_factor, 1e-6), 1.0)
        warmup = LinearLR(opt, start_factor=warmup_start_factor, total_iters=warmup_epochs)

        # Cosine over the remaining epochs, starting after warmup.
        cos_T = max(1, args.epochs - warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=cos_T, eta_min=min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=min_lr)

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
    # Train settings
    # -----------------------
    use_config = bool(args.config)
    use_endpoint = args.lambda_endpoint > 0
    use_residual = args.lambda_residual > 0
    use_phase = args.lambda_phase > 0
    use_phase_grad = args.lambda_phase_grad > 0
    use_sparam = args.lambda_sparam > 0

    # Optional dependency: conflictfree (only needed when --config is enabled).
    ConFIG_update = None
    apply_gradient_vector = None
    get_gradient_vector = None
    if use_config:
        try:
            from conflictfree.grad_operator import ConFIG_update as _ConFIG_update
            from conflictfree.utils import apply_gradient_vector as _apply_gradient_vector
            from conflictfree.utils import get_gradient_vector as _get_gradient_vector
            ConFIG_update = _ConFIG_update
            apply_gradient_vector = _apply_gradient_vector
            get_gradient_vector = _get_gradient_vector
        except Exception as e:
            if is_rank0():
                logger.warning(
                    f"--config enabled but 'conflictfree' is not available ({type(e).__name__}: {e}). "
                    "Disabling ConFIG; continuing with standard weighted loss."
                )
            use_config = False

    config_failed = False

    lam_mean = float(stats["lambda_um_mean"])
    lam_std = float(stats["lambda_um_std"])

    # persist conditioning shape for checkpoint consumers (e.g., Model/sample.py)
    try:
        args.cond_dim = int(getattr(train_ds, "cond_dim", 1))
    except Exception:
        args.cond_dim = 1

    # running averages (epoch-local)
    running_fm_loss = 0.0
    running_residual_loss = 0.0
    running_phase_loss = 0.0
    running_endpoint_loss = 0.0
    running_phase_grad_loss = 0.0
    running_sparam_loss = 0.0
    train_steps_epoch = 0
    phase_grad_steps_epoch = 0
    sparam_steps_epoch = 0
    start_time = time()

    global_step = 0

    if is_rank0():
        logger.info(f"Training for {args.epochs} epochs, starting at epoch {start_epoch}")
        logger.info(f"Schedule: PhaseA<= {args.phaseA_epochs}, PhaseB<= {args.phaseB_epochs}, PhaseC> {args.phaseB_epochs}")
        logger.info(f"ConFIG enabled: {use_config}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        # -----------------------
        # Curriculum schedule (A/B/C)
        # -----------------------
        N1 = args.phaseA_epochs
        N2 = args.phaseB_epochs

        # Endpoint weight
        endpoint_weight = args.lambda_endpoint
        if args.endpoint_warmup_epochs > 0 and endpoint_weight > 0:
            endpoint_weight *= min(1.0, epoch / args.endpoint_warmup_epochs)

        # Residual weight
        if use_residual and epoch > N1:
            residual_weight = args.lambda_residual * ramp_linear(epoch, start_epoch=N1, warmup_epochs=args.residual_warmup_epochs)
        else:
            residual_weight = 0.0

        # Phase weight
        if use_phase and epoch > N2:
            phase_weight = args.lambda_phase * ramp_linear(epoch, start_epoch=N2, warmup_epochs=args.phase_warmup_epochs)
        else:
            phase_weight = 0.0

        # Feature gates (0..1)
        phys_gate = 0.0
        if args.lambda_residual > 0:
            phys_gate = float(residual_weight / args.lambda_residual)
            phys_gate = max(0.0, min(1.0, phys_gate))

        phase_gate = 0.0
        if args.lambda_phase > 0:
            phase_gate = float(phase_weight / args.lambda_phase)
            phase_gate = max(0.0, min(1.0, phase_gate))

        # Phase-grad weight (sparse)
        if use_phase_grad and epoch > N2:
            phase_grad_weight = args.lambda_phase_grad * ramp_linear(epoch, start_epoch=N2, warmup_epochs=args.phase_grad_warmup_epochs)
        else:
            phase_grad_weight = 0.0

        compute_phase_epoch = (phase_weight > 0.0)

        # S-parameter weight (sparse)
        if use_sparam:
            if args.sparam_start_phase == "B":
                sparam_start = N1
            else:
                sparam_start = N2

            if epoch > sparam_start:
                sparam_weight = args.lambda_sparam * ramp_linear(
                    epoch, start_epoch=sparam_start, warmup_epochs=args.sparam_warmup_epochs
                )
            else:
                sparam_weight = 0.0
        else:
            sparam_weight = 0.0

        # device-focus schedule
        focus_start = 1.0
        focus_end = 1.0
        if args.device_focus_warmup_epochs > 0:
            ramp = min(1.0, epoch / args.device_focus_warmup_epochs)
        else:
            ramp = 1.0
        device_focus = focus_start + (args.device_focus_max - focus_start) * ramp

        if args.device_focus_hold_epochs > 0:
            if epoch <= args.device_focus_warmup_epochs + args.device_focus_hold_epochs:
                device_focus = args.device_focus_max
        if args.device_focus_decay_epochs > 0:
            e0 = args.device_focus_warmup_epochs + args.device_focus_hold_epochs
            if epoch > e0:
                frac = min(1.0, (epoch - e0) / args.device_focus_decay_epochs)
                device_focus = device_focus * (1.0 - frac) + focus_end * frac

        # -----------------------
        # Train loop
        # -----------------------
        for x_full, cond, aux in train_loader:
            x_full = x_full.to(device, dtype=dtype, non_blocking=True)
            cond = cond.to(device, dtype=dtype, non_blocking=True)
            aux = _move_aux_to_device(aux, device)

            fields_1 = x_full[:, 0:2]
            eps = x_full[:, 2:3]
            src = x_full[:, 3:4]
            # cond[0] is lambda_norm by construction (see Model/dataset.py)
            lambda_um = cond[:, 0:1] * lam_std + lam_mean  # [B,1] physical

            x0_fields = torch.randn_like(fields_1)
            with autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                t = sample_t(fields_1)  # [B,1,1,1]
                x_t_fields = psi_t(x0_fields, fields_1, t)
                v_t_fields = u_t(x0_fields, fields_1)
                x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

                compute_phase_grad_step = (phase_grad_weight > 0.0) and (global_step % args.phase_grad_every == 0)
                compute_sparam_step = (sparam_weight > 0.0) and (global_step % args.sparam_every == 0)
                compute_residual_step = (residual_weight > 0.0)

                fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad_loss, sparam_loss = cfm_loss_residual(
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
                    aux=aux,
                    compute_sparam=compute_sparam_step,
                    device_focus=device_focus,
                    w_min=args.w_min,
                    eps_thr=args.eps_thr,
                    dilate=args.dilate,
                    weight_residual=True,
                    compute_endpoint=use_endpoint,
                    compute_phase=compute_phase_epoch,
                    compute_phase_grad=compute_phase_grad_step,
                    phys_gate=phys_gate,
                    phase_gate=phase_gate,
                    compute_residual=compute_residual_step,
                )

                # scale residual to grid units
                residual_loss = residual_loss * (args.dx ** 4)

            # -----------------------
            # Backprop
            # -----------------------
            if use_config and (not config_failed):
                eps_cfg = float(args.config_eps)

                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(fm_loss).backward(retain_graph=True)
                    scaler.unscale_(opt)
                else:
                    fm_loss.backward(retain_graph=True)
                g_fm = get_gradient_vector(model.module).detach()
                opt.zero_grad(set_to_none=True)

                if residual_weight > 0.0:
                    if scaler.is_enabled():
                        scaler.scale(residual_loss).backward(retain_graph=True)
                        scaler.unscale_(opt)
                    else:
                        residual_loss.backward(retain_graph=True)
                    g_res = get_gradient_vector(model.module).detach()
                else:
                    g_res = None
                opt.zero_grad(set_to_none=True)

                if phase_weight > 0.0:
                    if scaler.is_enabled():
                        scaler.scale(phase_loss).backward(retain_graph=True)
                        scaler.unscale_(opt)
                    else:
                        phase_loss.backward(retain_graph=True)
                    g_phase = get_gradient_vector(model.module).detach()
                else:
                    g_phase = None
                opt.zero_grad(set_to_none=True)

                if endpoint_weight > 0.0:
                    if scaler.is_enabled():
                        scaler.scale(endpoint_loss).backward(retain_graph=True)
                        scaler.unscale_(opt)
                    else:
                        endpoint_loss.backward(retain_graph=True)
                    g_end = get_gradient_vector(model.module).detach()
                else:
                    g_end = None
                opt.zero_grad(set_to_none=True)

                if compute_phase_grad_step and (phase_grad_weight > 0.0):
                    if scaler.is_enabled():
                        scaler.scale(phase_grad_loss).backward(retain_graph=True)
                        scaler.unscale_(opt)
                    else:
                        phase_grad_loss.backward(retain_graph=True)
                    g_phg = get_gradient_vector(model.module).detach()
                else:
                    g_phg = None
                opt.zero_grad(set_to_none=True)

                if compute_sparam_step and (sparam_weight > 0.0):
                    if scaler.is_enabled():
                        scaler.scale(sparam_loss).backward(retain_graph=True)
                        scaler.unscale_(opt)
                    else:
                        sparam_loss.backward(retain_graph=True)
                    g_sp = get_gradient_vector(model.module).detach()
                else:
                    g_sp = None
                opt.zero_grad(set_to_none=True)

                # DDP-average
                ddp_allreduce_mean_(g_fm)
                if g_res is not None:
                    ddp_allreduce_mean_(g_res)
                if g_phase is not None:
                    ddp_allreduce_mean_(g_phase)
                if g_end is not None:
                    ddp_allreduce_mean_(g_end)
                if g_phg is not None:
                    ddp_allreduce_mean_(g_phg)
                if g_sp is not None:
                    ddp_allreduce_mean_(g_sp)

                # ConFIG diagnostics
                do_cfg_log = (is_rank0() and (global_step % int(args.config_log_every) == 0))
                if do_cfg_log:
                    grad_dict = {"fm": g_fm, "res": g_res, "phase": g_phase, "end": g_end, "phg": g_phg, "sp": g_sp}
                    w_dict = {
                        "fm": 1.0,
                        "res": float(residual_weight),
                        "phase": float(phase_weight),
                        "end": float(endpoint_weight),
                        "phg": float(phase_grad_weight),
                        "sp": float(sparam_weight),
                    }

                    lines = []
                    for name, g in grad_dict.items():
                        if g is None:
                            lines.append(f"{name}: None")
                            continue
                        n = _norm(g)
                        ff = _finite_frac(g)
                        mx = float(g.abs().max().item())
                        mn = float(g.abs().mean().item())
                        lines.append(
                            f"{name}: w={w_dict[name]:.3g} ||g||={n:.3e} finite={ff:.3f} |g|max={mx:.3e} |g|mean={mn:.3e}"
                        )

                    cos_lines = []
                    for name in ["res", "phase", "end", "phg", "sp"]:
                        g = grad_dict.get(name, None)
                        if g is None:
                            continue
                        cos_lines.append(f"cos(fm,{name})={_cos(g_fm, g):+.3f}")

                    logger.info(
                        "[ConFIG dbg] "
                        f"epoch={epoch} step={global_step} "
                        f"compute_phg={int(compute_phase_grad_step)} compute_sp={int(compute_sparam_step)} "
                        f"weights: res={residual_weight:.3g} phase={phase_weight:.3g} end={endpoint_weight:.3g} "
                        f"phg={phase_grad_weight:.3g} sp={sparam_weight:.3g} | "
                        + " | ".join(lines)
                        + (" | " + " ".join(cos_lines) if cos_lines else "")
                    )

                # Build active task list
                tasks = [("fm", g_fm, 1.0)]  # anchor

                def _maybe_add(name, g, weight):
                    if g is None:
                        return
                    if not torch.isfinite(g).all():
                        return
                    n = torch.linalg.norm(g)
                    if not torch.isfinite(n) or n.item() < eps_cfg:
                        return
                    if float(weight) <= 0.0:
                        return
                    tasks.append((name, g, float(weight)))

                _maybe_add("res", g_res, residual_weight)
                _maybe_add("phase", g_phase, phase_weight)
                _maybe_add("end", g_end, endpoint_weight)
                _maybe_add("phg", g_phg, phase_grad_weight)
                _maybe_add("sp", g_sp, sparam_weight)

                if do_cfg_log:
                    kept = [(n, _norm(g), w) for (n, g, w) in tasks]
                    logger.info(f"[ConFIG dbg] kept_tasks={kept} eps_cfg={eps_cfg:.1e} normalize={int(args.config_normalize)}")

                if len(tasks) == 1:
                    apply_gradient_vector(model.module, tasks[0][1])
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer_step()
                else:
                    grads = []
                    weights = []
                    norms = []
                    for _, g, w in tasks:
                        grads.append(g)
                        weights.append(w)
                        norms.append(torch.linalg.norm(g).clamp_min(eps_cfg))

                    if args.config_normalize:
                        grads = [(g / n) for g, n in zip(grads, norms)]
                    grads = [g * w for g, w in zip(grads, weights)]

                    try:
                        grads64 = [g.double() for g in grads]
                        g_cfg = ConFIG_update(grads64).float()

                        if do_cfg_log:
                            logger.info(f"[ConFIG dbg] g_cfg ||g||={_norm(g_cfg):.3e} finite={_finite_frac(g_cfg):.3f}")

                        if not torch.isfinite(g_cfg).all():
                            raise ValueError("ConFIG produced non-finite grad")

                        apply_gradient_vector(model.module, g_cfg)

                    except Exception as e:
                        if is_rank0():
                            logger.warning(f"ConFIG step failed: {type(e).__name__}: {e} | fallback=sum(grads)")

                        g_sum = torch.zeros_like(g_fm)
                        for g in grads:
                            if torch.isfinite(g).all():
                                g_sum += g
                        apply_gradient_vector(model.module, g_sum)

                        if not hasattr(main, "_cfg_fail_count"):
                            main._cfg_fail_count = 0
                        main._cfg_fail_count += 1
                        if main._cfg_fail_count >= int(args.config_fail_max):
                            if is_rank0():
                                logger.warning("ConFIG failed too many times; disabling permanently.")
                            config_failed = True

                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer_step()

            else:
                opt.zero_grad(set_to_none=True)
                total_loss = fm_loss
                if endpoint_weight > 0.0:
                    total_loss = total_loss + endpoint_weight * endpoint_loss
                if residual_weight > 0.0:
                    total_loss = total_loss + residual_weight * residual_loss
                if phase_weight > 0.0:
                    total_loss = total_loss + phase_weight * phase_loss
                if compute_phase_grad_step and (phase_grad_weight > 0.0):
                    total_loss = total_loss + phase_grad_weight * phase_grad_loss
                if compute_sparam_step and (sparam_weight > 0.0):
                    total_loss = total_loss + sparam_weight * sparam_loss

                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(opt)
                else:
                    total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer_step()

            update_ema(ema, model.module)

            # -----------------------
            # Logging accumulation
            # -----------------------
            running_fm_loss += fm_loss.item()
            running_residual_loss += residual_loss.item()
            running_phase_loss += phase_loss.item()
            if use_endpoint:
                running_endpoint_loss += endpoint_loss.item()
            if compute_phase_grad_step:
                running_phase_grad_loss += phase_grad_loss.item()
                phase_grad_steps_epoch += 1
            if compute_sparam_step:
                running_sparam_loss += sparam_loss.item()
                sparam_steps_epoch += 1

            train_steps_epoch += 1
            global_step += 1

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
                eval_sparam = 0.0
                eval_steps = 0

                max_val_batches = args.val_batches

                for i, (x_full, cond, aux) in enumerate(val_loader):
                    x_full = x_full.to(device, dtype=dtype, non_blocking=True)
                    cond = cond.to(device, dtype=dtype, non_blocking=True)
                    aux = _move_aux_to_device(aux, device)

                    with autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                        fields_1 = x_full[:, 0:2]
                        eps = x_full[:, 2:3]
                        src = x_full[:, 3:4]
                        lambda_um = cond[:, 0:1] * lam_std + lam_mean

                        x0_fields = torch.randn_like(fields_1)
                        t = sample_t(fields_1)
                        x_t_fields = psi_t(x0_fields, fields_1, t)
                        v_t_fields = u_t(x0_fields, fields_1)
                        x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

                        compute_phase_grad_val = (phase_grad_weight > 0.0)
                        compute_residual_val = (residual_weight > 0.0)

                        # In val, you can compute sparam every batch (cheap) or keep it gated:
                        compute_sparam_val = (sparam_weight > 0.0)

                        fm_v, res_v, ph_v, end_v, phg_v, sp_v = cfm_loss_residual(
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
                            aux=aux,
                            compute_sparam=compute_sparam_val,
                            device_focus=device_focus,
                            w_min=args.w_min,
                            eps_thr=args.eps_thr,
                            dilate=args.dilate,
                            weight_residual=True,
                            compute_endpoint=use_endpoint,
                            compute_phase=compute_phase_epoch,
                            compute_phase_grad=compute_phase_grad_val,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            compute_residual=compute_residual_val,
                        )

                        res_v = res_v * (args.dx ** 4)

                    eval_fm += fm_v.item()
                    eval_res += res_v.item()
                    eval_phase += ph_v.item()
                    if use_endpoint:
                        eval_endpoint += end_v.item()
                    if compute_phase_grad_val:
                        eval_phase_grad += phg_v.item()
                    if compute_sparam_val:
                        eval_sparam += sp_v.item()

                    eval_steps += 1
                    if max_val_batches > 0 and (i + 1) >= max_val_batches:
                        break

                fm_t = torch.tensor(eval_fm / max(eval_steps, 1), device=device)
                res_t = torch.tensor(eval_res / max(eval_steps, 1), device=device)
                ph_t = torch.tensor(eval_phase / max(eval_steps, 1), device=device)
                end_t = torch.tensor(eval_endpoint / max(eval_steps, 1), device=device)
                phg_t = torch.tensor(eval_phase_grad / max(eval_steps, 1), device=device)
                sp_t = torch.tensor(eval_sparam / max(eval_steps, 1), device=device)

                ddp_allreduce_mean_(fm_t)
                ddp_allreduce_mean_(res_t)
                ddp_allreduce_mean_(ph_t)
                if use_endpoint:
                    ddp_allreduce_mean_(end_t)
                if phase_grad_weight > 0.0:
                    ddp_allreduce_mean_(phg_t)
                if sparam_weight > 0.0:
                    ddp_allreduce_mean_(sp_t)

                val_fm_loss = float(fm_t.item())
                val_res_loss = float(res_t.item())
                val_phase_loss = float(ph_t.item())
                val_endpoint_loss = float(end_t.item()) if use_endpoint else 0.0
                val_phase_grad_loss = float(phg_t.item()) if (phase_grad_weight > 0.0) else 0.0
                val_sparam_loss = float(sp_t.item()) if (sparam_weight > 0.0) else 0.0

            # -----------------------
            # Sample residual eval + images (rank 0 only)
            # -----------------------
            sample_residual_mean = 0.0
            if is_rank0():
                residuals = []
                with torch.no_grad():
                    n_samples = min(len(train_ds), args.sample_eval_limit)
                    for s in range(n_samples):
                        x_full_s, cond_s, aux_s = train_ds[s]
                        x_full_s = x_full_s.unsqueeze(0).to(device, dtype=dtype)
                        cond_s = cond_s.unsqueeze(0).to(device, dtype=dtype)

                        with autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                            eps_s = x_full_s[:, 2:3]
                            src_s = x_full_s[:, 3:4]
                            x0_fields_s = torch.randn_like(x_full_s[:, 0:2])

                            # cond[0] is lambda_norm by construction (see Model/dataset.py)
                            lambda_um_s = cond_s[:, 0:1] * lam_std + lam_mean  # [B,1] physical
                            cond_maps_s = torch.cat([eps_s, src_s], dim=1)  # [B,2,H,W]

                            x1_fields_pred = fm_sample(
                                ema,
                                x0_fields_s,
                                num_steps=args.fm_steps,
                                use_stoc_samp=args.use_stoc_samp,
                                cond_maps=cond_maps_s,
                                cond=cond_s,
                                lambda_um=lambda_um_s,
                                phys_gate=phys_gate,
                                phase_gate=phase_gate,
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
                        res_mag = torch.sqrt(R[:, 0:1] ** 2 + R[:, 1:2] ** 2 + 1e-12) * (args.dx ** 2)
                        residuals.append(res_mag.mean().item())

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

                            mag_gt = np.sqrt(ezr_gt_img ** 2 + ezi_gt_img ** 2)
                            mag_pred = np.sqrt(ezr_pred_img ** 2 + ezi_pred_img ** 2)
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
                f"[epoch {epoch:04d}] val_fm={val_fm_loss:.4e}, val_res={val_res_loss:.4e}, val_phase={val_phase_loss:.4e}"
            )
            if use_endpoint:
                log_msg += f", val_endpoint={val_endpoint_loss:.4e}"
            if phase_grad_weight > 0.0:
                log_msg += f", val_phase_grad={val_phase_grad_loss:.4e}"
            if sparam_weight > 0.0:
                log_msg += f", val_sparam={val_sparam_loss:.4e}"
            log_msg += f", sample_residual={sample_residual_mean:.4e}"
            logger.info(log_msg)

            with open(val_csv_path, "a", encoding="UTF8", newline="") as f_csv:
                writer = csv.writer(f_csv)
                row = [
                    epoch,
                    val_fm_loss,
                    val_res_loss,
                    val_phase_loss,
                    val_endpoint_loss if use_endpoint else "",
                    val_phase_grad_loss if (phase_grad_weight > 0.0) else "",
                    val_sparam_loss if (sparam_weight > 0.0) else "",
                    sample_residual_mean,
                ]
                writer.writerow(row)

            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "val/fm_loss": val_fm_loss,
                    "val/residual_loss": val_res_loss,
                    "val/phase_loss": val_phase_loss,
                    "val/sample_residual": sample_residual_mean,
                }
                if use_endpoint:
                    log_dict["val/endpoint_loss"] = val_endpoint_loss
                if phase_grad_weight > 0.0:
                    log_dict["val/phase_grad_loss"] = val_phase_grad_loss
                if sparam_weight > 0.0:
                    log_dict["val/sparam_loss"] = val_sparam_loss
                wandb_run.log(log_dict, step=epoch)

        # -----------------------
        # Train logging (DDP-averaged)
        # -----------------------
        if epoch % args.log_every == 0:
            torch.cuda.synchronize()
            sec_per_epoch = time() - start_time

            fm_mean = torch.tensor(running_fm_loss / max(train_steps_epoch, 1), device=device)
            res_mean = torch.tensor(running_residual_loss / max(train_steps_epoch, 1), device=device)
            ph_mean = torch.tensor(running_phase_loss / max(train_steps_epoch, 1), device=device)
            end_mean = torch.tensor(running_endpoint_loss / max(train_steps_epoch, 1), device=device)

            phg_mean = torch.tensor(running_phase_grad_loss / max(phase_grad_steps_epoch, 1), device=device)
            sp_mean = torch.tensor(running_sparam_loss / max(sparam_steps_epoch, 1), device=device)

            ddp_allreduce_mean_(fm_mean)
            ddp_allreduce_mean_(res_mean)
            ddp_allreduce_mean_(ph_mean)
            if use_endpoint:
                ddp_allreduce_mean_(end_mean)
            if phase_grad_weight > 0.0:
                ddp_allreduce_mean_(phg_mean)
            if sparam_weight > 0.0:
                ddp_allreduce_mean_(sp_mean)

            if is_rank0():
                msg = (
                    f"(epoch={epoch:04d}) train_fm={fm_mean.item():.4e}, "
                    f"train_residual={res_mean.item():.4e}, train_phase={ph_mean.item():.4e}"
                )
                if use_endpoint:
                    msg += f", train_endpoint={end_mean.item():.4e}"
                if phase_grad_weight > 0.0:
                    msg += f", train_phase_grad={phg_mean.item():.4e}"
                if sparam_weight > 0.0:
                    msg += f", train_sparam={sp_mean.item():.4e}"
                msg += f", sec_per_epoch={sec_per_epoch:.3e}"
                msg += (
                    f" | w_res={residual_weight:.3g}, w_phase={phase_weight:.3g}, "
                    f"w_phg={phase_grad_weight:.3g}, w_sp={sparam_weight:.3g}"
                )
                logger.info(msg)

            if wandb_run is not None and is_rank0():
                wandb_log_dict = {
                    "train/fm_loss": fm_mean.item(),
                    "train/residual_loss": res_mean.item(),
                    "train/phase_loss": ph_mean.item(),
                    "train/sec_per_epoch": sec_per_epoch,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/residual_weight": residual_weight,
                    "train/phase_weight": phase_weight,
                    "train/phase_grad_weight": phase_grad_weight,
                    "train/sparam_weight": sparam_weight,
                    "train/device_focus": device_focus,
                    "train/use_config": int(use_config and (not config_failed)),
                }
                if use_endpoint:
                    wandb_log_dict["train/endpoint_loss"] = end_mean.item()
                    wandb_log_dict["train/endpoint_weight"] = endpoint_weight
                if phase_grad_weight > 0.0:
                    wandb_log_dict["train/phase_grad_loss"] = phg_mean.item()
                if sparam_weight > 0.0:
                    wandb_log_dict["train/sparam_loss"] = sp_mean.item()
                wandb_run.log(wandb_log_dict, step=epoch)

            # reset epoch-local meters
            start_time = time()
            running_fm_loss = 0.0
            running_residual_loss = 0.0
            running_phase_loss = 0.0
            running_endpoint_loss = 0.0
            running_phase_grad_loss = 0.0
            running_sparam_loss = 0.0
            train_steps_epoch = 0
            phase_grad_steps_epoch = 0
            sparam_steps_epoch = 0

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
            ddp_barrier(device)

    if wandb_run is not None:
        wandb_run.finish()

    if is_rank0():
        logger.info("Done.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default="data_generation/data")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--use-shards",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Load dataset from shard .npz files (pack_dataset.py) instead of per-sample folders.",
    )
    parser.add_argument("--shard-subdir", type=str, default="shards",
                        help="Subdirectory inside each sweep that contains shard files and index.json.")
    parser.add_argument("--shard-index-name", type=str, default="index.json",
                        help="Filename of the shard index JSON.")
    parser.add_argument("--include-sweeps", type=str, default="",
                        help="Comma-separated list of sweep subfolders to include (default: all).")
    parser.add_argument(
        "--subset-train-per-sweep",
        type=str,
        default="",
        help="Optional: exact TRAIN sample counts per sweep, e.g. 'coupler_sweep=3000,y_branch_sweep=1000'.",
    )
    parser.add_argument(
        "--subset-val-per-sweep",
        type=str,
        default="",
        help="Optional: exact VAL sample counts per sweep, e.g. 'coupler_sweep=750,y_branch_sweep=250'. If omitted, derived from train ratios.",
    )
    parser.add_argument(
        "--subset-val-total",
        type=int,
        default=0,
        help="If subset-train-per-sweep is set and subset-val-per-sweep is omitted, derive val per sweep to sum to this total (default 25% of train).",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed for deterministic subset selection. Use a fixed value to reproduce the same subset.",
    )

    # GLOBAL batch size for DDP
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-num-workers", type=int, default=8,
                        help="Dataloader worker count for train loader.")
    parser.add_argument("--val-num-workers", type=int, default=4,
                        help="Dataloader worker count for val loader.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Prefetch factor per worker (ignored if workers=0).")

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-seed", type=int, default=15)

    parser.add_argument("--version", type=str, default="physics_unet_pbfm_ddp")
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--dx", type=float, default=1.0 / 30.0)
    parser.add_argument("--lambda-um", type=float, default=1.55)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Linear warmup epochs for LR schedule (0 disables warmup).",
    )
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.1,
        help="Warmup starts at (warmup_start_factor * lr) and ramps to lr over warmup-epochs.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=5e-6,
        help="Minimum LR for cosine decay (eta_min).",
    )

    # ConFIG toggle
    parser.add_argument(
        "--config",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable ConFIG (conflict-free gradient descent). Use --no-config to disable.",
    )

    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Enable torch.cuda.amp autocast + GradScaler.")

    # Curriculum boundaries
    parser.add_argument("--phaseA-epochs", dest="phaseA_epochs", type=int, default=50,
                        help="Phase A: epochs <= this use FM (+ endpoint) only.")
    parser.add_argument("--phaseB-epochs", dest="phaseB_epochs", type=int, default=200,
                        help="Phase B: N1<epoch<=this adds residual (ramp). Phase C starts after this.")

    # Weights + warmups
    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--residual-warmup-epochs", type=int, default=50)

    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--phase-warmup-epochs", type=int, default=50)

    parser.add_argument("--lambda-endpoint", type=float, default=0.0)
    parser.add_argument("--endpoint-warmup-epochs", type=int, default=0)

    # Phase-gradient loss (sparse)
    parser.add_argument("--lambda-phase-grad", type=float, default=0.0)
    parser.add_argument("--phase-grad-warmup-epochs", type=int, default=100)
    parser.add_argument("--phase-grad-every", type=int, default=4,
                        help="Compute phase-grad every N train steps (sparse).")

    # S-parameter loss (sparse)
    parser.add_argument("--lambda-sparam", type=float, default=0.0)
    parser.add_argument("--sparam-warmup-epochs", type=int, default=50)
    parser.add_argument("--sparam-every", type=int, default=1,
                        help="Compute sparam loss every N train steps.")
    parser.add_argument(
        "--sparam-start-phase",
        type=str,
        default="C",
        choices=["B", "C"],
        help="Which curriculum phase to start the S-parameter loss in. "
             "B = start after phaseA (epoch > phaseA_epochs). "
             "C = start after phaseB (epoch > phaseB_epochs, default).",
    )

    parser.add_argument("--normalize-eps", type=bool, default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--sample-eval-limit", type=int, default=16)

    parser.add_argument("--resume-from", type=str, default="")

    parser.add_argument("--use-wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project", type=str, default="Rayfield")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Spatial weighting / focus
    parser.add_argument("--device-focus-max", type=float, default=3.0)
    parser.add_argument("--device-focus-warmup-epochs", type=int, default=100)
    parser.add_argument("--device-focus-hold-epochs", type=int, default=0)
    parser.add_argument("--device-focus-decay-epochs", type=int, default=200)
    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--eps-thr", type=float, default=6.0)
    parser.add_argument("--dilate", type=int, default=9)

    parser.add_argument("--val-batches", type=int, default=16,
                        help="How many val batches to average each eval. 0 = full val set.")

    # -----------------------
    # ConFIG stability knobs
    # -----------------------
    parser.add_argument(
        "--config-eps",
        dest="config_eps",
        type=float,
        default=1e-8,
        help="ConFIG: minimum gradient norm threshold; tasks with ||g|| < eps are dropped.",
    )
    parser.add_argument(
        "--config-normalize",
        dest="config_normalize",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ConFIG: normalize each task gradient to unit norm before applying task weights.",
    )
    parser.add_argument(
        "--config-fail-max",
        dest="config_fail_max",
        type=int,
        default=25,
        help="Disable ConFIG permanently after this many step-level failures.",
    )
    parser.add_argument(
        "--config-log-every",
        dest="config_log_every",
        type=int,
        default=200,
        help="How often (in train steps) to log ConFIG diagnostics on rank0.",
    )

    args = parser.parse_args()
    main(args)
