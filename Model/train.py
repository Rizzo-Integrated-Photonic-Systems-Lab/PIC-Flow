# train_physics_unet_pbfm.py

import argparse
import csv
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


import wandb

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector

# If you want to use your multi-device FDTDDataset, swap this import.
# For now this assumes the simple "coupler_sweep" layout.
from dataset import CouplerDataset  # or from coupler_dataset import CouplerDataset

from physics_unet import PhysicsUNet
from flow_matching import (
    psi_t,
    u_t,
    sample_t,
    cfm_loss_residual,
    sample as fm_sample,  # renamed to avoid clashing with Python's "sample"
)
from grad_utils import GradientsHelper, generalized_b_xy_c_to_image

from matplotlib import pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Prefer cuSOLVER for linear algebra used inside ConFIG; magma was triggering
# cusolverDnSormqr_bufferSize INVALID_VALUE on some batches.
torch.backends.cuda.preferred_linalg_library("cusolver")

dtype = torch.float32


# ---------------- Utils ---------------- #

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))],
    )
    logger = logging.getLogger(__name__)
    return logger


# ---------------- Main training ---------------- #

def main(args):
    assert torch.cuda.is_available(), "Need at least one CUDA GPU."
    device = torch.device("cuda:0")
    torch.cuda.set_device(device.index)
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    # ---- experiment dir + logging ----
    results_dir = "logs_physics_unet_pbfm"
    os.makedirs(results_dir, exist_ok=True)
    experiment_dir = os.path.join(results_dir, args.version)

    if args.resume_from:
        os.makedirs(experiment_dir, exist_ok=True)
    else:
        # start fresh
        if os.path.exists(experiment_dir):
            # nuke old dir
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(experiment_dir, exist_ok=True)

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment dir: {experiment_dir}")

    # CSV for validation
    val_csv_path = os.path.join(experiment_dir, "validation.csv")
    mode = "a" if args.resume_from and os.path.exists(val_csv_path) else "w"
    with open(val_csv_path, mode, encoding="UTF8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if mode == "w":
            writer.writerow(["epoch", "mean_residual", "eval_loss"])

    # wandb
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.version,
            config=vars(args),
        )

    # ---- dataset ----
    # If you want multi-device, use your more general FDTDDataset and pass device_subdirs.
    train_ds = CouplerDataset(
        root_dir=os.path.join(args.data_root, "coupler_sweep"),
        split="train",
        train_fraction=args.train_fraction,
        normalize_eps=args.normalize_eps,
    )
    stats = train_ds.get_stats()

    val_ds = CouplerDataset(
        root_dir=os.path.join(args.data_root, "coupler_sweep"),
        split="val",
        train_fraction=args.train_fraction,
        stats=stats,
        normalize_eps=args.normalize_eps,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # ---- model: PhysicsUNet backbone with embedded Maxwell filters ----
    dx = args.dx
    lam0 = args.lambda_um
    omega = 2.0 * math.pi / lam0  # 2π / λ (in 1/µm)


    # infer H, W from one sample
    sample_x, _ = train_ds[0]   # sample_x: [3,H,W], cond is scalar
    _, H, W = sample_x.shape


    model = PhysicsUNet(
        in_channels=3,
        out_channels=2,
        model_channels=args.hidden_size,
        num_res_blocks=2,
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

    # IMPORTANT: tell the model how to de-normalize for physics features
    model.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False

    # Keep buffers consistent on ema too (safe even though deepcopy copied them)
    ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    update_ema(ema, model, decay=0.0)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6, last_epoch=-1)

    # Maxwell residual helper (Helmholtz) – this is where PDE lives
    grad_helper = GradientsHelper(
        device=device,
        dx=args.dx,
        dy=args.dx,
        wavelength_um=args.lambda_um,
    )

    # ---- resume ----
    start_epoch = 1
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])

        # re-apply normalization buffers (in case code changed / buffers missing)
        model.set_normalization_stats(stats, normalize_eps=args.normalize_eps)
        ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)
        try:
            ckpt_name = os.path.basename(args.resume_from)
            last_epoch = int(os.path.splitext(ckpt_name)[0])
        except Exception:
            last_epoch = 0
        start_epoch = last_epoch + 1
        scheduler.last_epoch = last_epoch

    # ---- training loop ----
    # Disable ConFIG entirely for now; use summed gradients of FM + weighted residual.
    use_config = False
    config_failed = False           # retained for clarity; unused when use_config is False
    train_steps = 0
    running_fm_loss = 0.0
    running_residual_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        # unrolling schedule if you later add it – for now, we just need dt for residual scaling
        steps_for_residual = args.fm_steps
        residual_weight = args.lambda_residual
        if args.residual_warmup_epochs > 0:
            residual_weight *= min(1.0, epoch / args.residual_warmup_epochs)
        residual_dt = 1.0 / max(steps_for_residual, 1)

        for x_full, cond in train_loader:
            # x_full: [B,3,H,W] = [Re, Im, eps]
            # cond:   [B,1]     = normalized λ
            x_full = x_full.to(device, dtype=dtype)
            cond   = cond.to(device, dtype=dtype)

            fields_1 = x_full[:, 0:2]
            eps      = x_full[:, 2:3]

            # --- recover physical wavelength in µm per sample ---
            lam_mean = stats["lambda_um_mean"]
            lam_std  = stats["lambda_um_std"]
            # cond is normalized: λ_norm = (λ - mean)/std
            lambda_um = cond * lam_std + lam_mean     # [B,1]

            # --- Flow matching on fields only ---
            x0_fields  = torch.randn_like(fields_1)
            t          = sample_t(fields_1)           # [B,1,1,1]
            x_t_fields = psi_t(x0_fields, fields_1, t)
            v_t_fields = u_t(x0_fields, fields_1)

            x_t_input = torch.cat([x_t_fields, eps], dim=1)  # [B,3,H,W]

            # FM + physics residual, conditioned on λ
            fm_loss, residual_loss = cfm_loss_residual(
                model,
                x_t_input,
                t,
                v_t_fields,
                grad_helper,
                args.use_dignorm,
                residual_dt,
                stats,
                args.normalize_eps,
                fields_1,
                cond=cond,          # normalized λ (for the UNet)
                lambda_um=lambda_um # physical λ (for the PDE residual)
            )



            # residual is in physical units; scale to grid units to avoid dx^-4 blowup
            residual_loss = residual_loss * (args.dx ** 4)
            weighted_residual_loss = residual_weight * residual_loss

            if use_config:
                grads = []

                opt.zero_grad()
                fm_loss.backward(retain_graph=True)
                grads.append(get_gradient_vector(model))

                opt.zero_grad()
                weighted_residual_loss.backward()
                grads.append(get_gradient_vector(model))

                # sanity check gradients
                valid = True
                for g in grads:
                    if not torch.isfinite(g).all():
                        valid = False
                        break

                if not valid:
                    print("Non-finite grads, falling back to FM-only step.")
                    opt.zero_grad()
                    if torch.isfinite(grads[0]).all():
                        apply_gradient_vector(model, grads[0])
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                else:
                    try:
                        g_cfg = ConFIG_update(grads)
                        if not torch.isfinite(g_cfg).all():
                            raise ValueError("ConFIG produced non-finite grad")
                        apply_gradient_vector(model, g_cfg)
                    except Exception as e:
                        # After a ConFIG failure, avoid repeated cusolver errors by
                        # switching to summed gradients for the rest of training.
                        if not config_failed:
                            logger.warning(f"ConFIG failed: {e}; falling back to summed grads for the rest of training.")
                            config_failed = True
                            use_config = False
                        apply_gradient_vector(model, grads[0] + grads[1])

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            else:
                total_loss = fm_loss + weighted_residual_loss
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            update_ema(ema, model)

            running_fm_loss += fm_loss.item()
            running_residual_loss += residual_loss.item()
            train_steps += 1

        # ---- validation / sampling ----
        if epoch % args.eval_every == 0:
            model.eval()
            ema.eval()

            # one eval batch
            with torch.no_grad():
                eval_fm = 0.0
                eval_steps = 0
                for x_full, cond in val_loader:
                    x_full = x_full.to(device, dtype=dtype)
                    cond   = cond.to(device, dtype=dtype)

                    fields_1 = x_full[:, 0:2]
                    eps      = x_full[:, 2:3]

                    lam_mean = stats["lambda_um_mean"]
                    lam_std  = stats["lambda_um_std"]
                    lambda_um = cond * lam_std + lam_mean

                    x0_fields  = torch.randn_like(fields_1)
                    t          = sample_t(fields_1)
                    x_t_fields = psi_t(x0_fields, fields_1, t)
                    v_t_fields = u_t(x0_fields, fields_1)
                    x_t_input  = torch.cat([x_t_fields, eps], dim=1)

                    fm_loss_val, _ = cfm_loss_residual(
                        ema,
                        x_t_input,
                        t,
                        v_t_fields,
                        grad_helper,
                        args.use_dignorm,
                        residual_dt,
                        stats,
                        args.normalize_eps,
                        fields_1,
                        cond=cond,
                        lambda_um=lambda_um,
                    )
                    eval_fm += fm_loss_val.item()
                    eval_steps += 1
                    if eval_steps >= 1:
                        break

                eval_loss = eval_fm / max(eval_steps, 1)

            # residuals of sampled fields (Helmholtz)
            residuals = []
            with torch.no_grad():
                n_samples = min(len(train_ds), args.sample_eval_limit)
                for s in range(n_samples):
                    x_full, cond = train_ds[s]      # x_full: [3,H,W], cond: [1]
                    x_full = x_full.unsqueeze(0).to(device, dtype=dtype)  # [1,3,H,W]
                    cond   = cond.unsqueeze(0).to(device, dtype=dtype)    # [1,1]

                    eps = x_full[:, 2:3]
                    x0_fields = torch.randn_like(x_full[:, 0:2])

                    lam_mean = stats["lambda_um_mean"]
                    lam_std  = stats["lambda_um_std"]
                    lambda_um = cond * lam_std + lam_mean   # [1,1]

                    x1_fields_pred = fm_sample(
                        ema,
                        x0_fields,
                        num_steps=args.fm_steps,
                        use_stoc_samp=args.use_stoc_samp,
                        cond_eps=eps,
                        cond=cond,              # conditioning for the network
                        lambda_um=lambda_um,
                    )
                    x1_pred = torch.cat([x1_fields_pred, eps], dim=1)

                    # de-normalize fields & eps (same as before)
                    x1_pred[:, 0] = x1_pred[:, 0] * stats["ez_real_std"] + stats["ez_real_mean"]
                    x1_pred[:, 1] = x1_pred[:, 1] * stats["ez_imag_std"] + stats["ez_imag_mean"]
                    if args.normalize_eps:
                        x1_pred[:, 2] = x1_pred[:, 2] * stats["eps_std"] + stats["eps_mean"]

                    # use per-sample λ in residual
                    res = grad_helper.compute_residual(x1_pred, wavelength_um=lambda_um)["residual"].abs()
    

                    res = res * (args.dx ** 2)  # grid-units scaling
                    B, Cc, Hh, Ww = x1_pred.shape
                    res_img = generalized_b_xy_c_to_image(res, pixels_x=Hh, pixels_y=Ww)

                    if s == 0:
                        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                        plt.subplots_adjust(wspace=0.5)

                        c0 = axes[0].imshow(x1_pred[0, 1].cpu().numpy().T, cmap="magma")
                        c1 = axes[1].imshow(x1_pred[0, 0].cpu().numpy().T, cmap="magma")
                        c2 = axes[2].imshow(res_img[0, 0].cpu().numpy().T, cmap="magma", norm="log")
                        cbar = [c0, c1, c2]

                        for k, axx in enumerate(axes):
                            axx.set_xlabel("x (grid index)")
                            axx.set_ylabel("y (grid index)")
                            axx.invert_yaxis()
                            axx.set_xticks([0, Ww - 1])
                            axx.set_yticks([0, Hh - 1])
                            axx.set_aspect("equal")

                            pos = axx.get_position()
                            cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
                            cb = plt.colorbar(cbar[k], cax=cax, orientation="vertical")
                            cb.minorticks_on()

                        axes[0].set_title("Im(Ez)")
                        axes[1].set_title("Re(Ez)")
                        axes[2].set_title("Residual R(Ez, eps)")

                        fig_path = os.path.join(experiment_dir, f"sample_epoch_{epoch:04d}.svg")
                        plt.savefig(fig_path, format="svg", bbox_inches="tight")
                        plt.close(fig)

                        np.save(
                            os.path.join(experiment_dir, f"sample_epoch_{epoch:04d}.npy"),
                            x1_pred.cpu().numpy(),
                        )

                    residuals.append(res_img.abs().mean().item())

            mean_residual = float(np.mean(residuals)) if residuals else 0.0

            logger.info(
                f"[epoch {epoch:04d}] val_fm_loss={eval_loss:.4e}, "
                f"sample_residual={mean_residual:.4e}"
            )

            with open(val_csv_path, "a", encoding="UTF8", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([epoch, mean_residual, eval_loss])

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "val/fm_loss": eval_loss,
                        "val/sample_residual": mean_residual,
                    }
                )

        # ---- logging ----
        if epoch % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            sec_per_epoch = end_time - start_time
            avg_fm = running_fm_loss / max(train_steps, 1)
            avg_res = running_residual_loss / max(train_steps, 1)

            logger.info(
                f"(epoch={epoch:04d}) "
                f"train_fm={avg_fm:.4e}, train_residual={avg_res:.4e}, "
                f"sec_per_epoch={sec_per_epoch:.3e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/fm_loss": avg_fm,
                        "train/residual_loss": avg_res,
                        "train/sec_per_epoch": sec_per_epoch,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                )
            start_time = time()
            running_fm_loss = 0.0
            running_residual_loss = 0.0
            train_steps = 0

        scheduler.step()

        # ---- checkpoint ----
        if epoch % args.ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{epoch:07d}.pt")
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "stats": stats,
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data_generation/data")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-seed", type=int, default=15)

    parser.add_argument("--version", type=str, default="physics_unet_pbfm")
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--dx", type=float, default=1.0 / 30.0, help="Grid spacing (um)")
    parser.add_argument("--lambda-um", type=float, default=1.55, help="Representative wavelength for omega")

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--use-dignorm", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-residual", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="If True, combine FM + residual via ConFIG")
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lambda-residual", type=float, default=1.0,
                        help="Weight for physics residual term (before ConFIG)")
    parser.add_argument("--residual-warmup-epochs", type=int, default=50,
                        help="Linearly ramp residual weight over this many epochs (0 disables)")
    parser.add_argument("--normalize-eps", type=bool, default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--sample-eval-limit", type=int, default=16,
                        help="Number of train samples to FM-sample for residual evaluation")

    parser.add_argument("--resume-from", type=str, default="", help="Path to checkpoint .pt for resume")

    parser.add_argument("--use-wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project", type=str, default="Rayfield")
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()
    main(args)
