# train.py  (train_physics_unet_pbfm_ddp.py)
# Fixed weighted-sum training with optional ConFIG (conflictfree) gradient surgery.

import argparse
import csv
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast
    _AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.cuda.amp import autocast  # type: ignore
    _AMP_NEW_API = False

import wandb

from dataset import FDTDDataset
from dataset_fast import FastFDTDDataset
from physics_unet import PhysicsUNet, HelmholtzResidual2D
from complex_physics_unet import ComplexPhysicsUNet
from flow_matching import psi_t, u_t, sample_t, cfm_loss_residual, sample as fm_sample, SIG_MIN
from sparams_loss import extract_sparams

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# Optional: ConFIG (conflictfree library)
try:
    from conflictfree.grad_operator import ConFIG_update as _ConFIG_update  # type: ignore
    from conflictfree.utils import apply_gradient_vector as _apply_gradient_vector  # type: ignore
    from conflictfree.utils import get_gradient_vector as _get_gradient_vector  # type: ignore
    _HAS_CONFLICTFREE = True
except Exception:  # pragma: no cover
    _ConFIG_update = None
    _apply_gradient_vector = None
    _get_gradient_vector = None
    _HAS_CONFLICTFREE = False

# Optional (rank0 sample plots)
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.preferred_linalg_library("cusolver")

dtype = torch.float32


def _save_eval_sample_png(
    *,
    out_path: str,
    title: str,
    eps_phys: np.ndarray,
    ezr_gt: np.ndarray,
    ezi_gt: np.ndarray,
    ezr_pred: np.ndarray,
    ezi_pred: np.ndarray,
) -> None:
    """
    Save a compact 2x4 comparison panel:
      eps | Ez_real GT | Ez_imag GT | |Ez| GT
      Ez_real pred | Ez_imag pred | |Ez| pred | |err|
    Arrays are expected as HxW numpy arrays in physical units.
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mag_gt = np.sqrt(ezr_gt**2 + ezi_gt**2 + 1e-12)
        mag_pred = np.sqrt(ezr_pred**2 + ezi_pred**2 + 1e-12)
        mag_err = np.sqrt((ezr_pred - ezr_gt) ** 2 + (ezi_pred - ezi_gt) ** 2 + 1e-12)

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.reshape(2, 4)
        fig.suptitle(title, fontsize=11)

        def im(ax, arr, t, *, cmap="magma", vmin=None, vmax=None):
            h = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(t, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return h

        hs = []
        hs.append(im(axes[0, 0], eps_phys, "eps (phys)", cmap="viridis"))
        hs.append(im(axes[0, 1], ezr_gt, "Ez_real (GT)", cmap="RdBu"))
        hs.append(im(axes[0, 2], ezi_gt, "Ez_imag (GT)", cmap="RdBu"))
        hs.append(im(axes[0, 3], mag_gt, "|Ez| (GT)", cmap="magma"))

        hs.append(im(axes[1, 0], ezr_pred, "Ez_real (pred)", cmap="RdBu"))
        hs.append(im(axes[1, 1], ezi_pred, "Ez_imag (pred)", cmap="RdBu"))
        hs.append(im(axes[1, 2], mag_pred, "|Ez| (pred)", cmap="magma"))
        hs.append(im(axes[1, 3], mag_err, "|err|", cmap="magma"))

        for ax, h in zip(axes.ravel(), hs):
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception:
        # plotting must never crash training
        try:
            plt.close("all")
        except Exception:
            pass


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
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


def ddp_allreduce_mean_(x: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def ddp_barrier(device: torch.device | None = None):
    if not dist.is_initialized():
        return
    try:
        if device is not None and device.type == "cuda" and device.index is not None:
            dist.barrier(device_ids=[int(device.index)])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


def broadcast_object(obj, src=0):
    if not dist.is_initialized():
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def ramp_linear(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    """
    Linear ramp from 0->1 over warmup_epochs, starting at start_epoch.

    Fix: use epoch < start_epoch (not <=) so "start_epoch=N" activates on epoch N.
    """
    if warmup_epochs <= 0:
        return 1.0
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch) / float(warmup_epochs))


def _phase_start_epoch(phase: str, N1: int, N2: int) -> int:
    p = str(phase).strip().upper()
    if p == "A":
        return 0
    if p == "B":
        return int(N1)
    return int(N2)


def _move_aux_to_device(aux, device: torch.device):
    if not isinstance(aux, dict):
        return None
    out = dict(aux)

    def _to(x, *, dt=None):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if dt is not None:
            return x.to(device, dtype=dt, non_blocking=True)
        return x.to(device, non_blocking=True)

    out["port_masks"] = _to(out.get("port_masks", None), dt=dtype)
    out["port_ids"] = _to(out.get("port_ids", None))
    out["sparams_true"] = _to(out.get("sparams_true", None))
    out["port_valid"] = _to(out.get("port_valid", None), dt=dtype)

    if out.get("in_port_idx", None) is not None:
        if torch.is_tensor(out["in_port_idx"]):
            out["in_port_idx"] = out["in_port_idx"].to(device, non_blocking=True)
        else:
            out["in_port_idx"] = torch.as_tensor(out["in_port_idx"], dtype=torch.long, device=device)

    return out


def _format_sparams_compact(S_true_1d, S_pred_1d, port_valid_1d=None, max_ports=4) -> str:
    if S_true_1d is None or S_pred_1d is None:
        return ""
    S_true_1d = S_true_1d.reshape(-1)[:max_ports]
    S_pred_1d = S_pred_1d.reshape(-1)[:max_ports]
    keep = None
    if port_valid_1d is not None:
        pv = port_valid_1d.reshape(-1)[:max_ports].detach().cpu()
        keep = (pv > 0.5).numpy()

    mags_t = torch.abs(S_true_1d).detach().cpu().numpy()
    mags_p = torch.abs(S_pred_1d).detach().cpu().numpy()
    ph_t = torch.angle(S_true_1d).detach().cpu().numpy()
    ph_p = torch.angle(S_true_1d).detach().cpu().numpy()  # safe if pred missing; overwritten below
    try:
        ph_p = torch.angle(S_pred_1d).detach().cpu().numpy()
    except Exception:
        pass

    parts = []
    for i in range(min(len(mags_t), len(mags_p), max_ports)):
        if keep is not None and (i >= len(keep) or not bool(keep[i])):
            continue
        parts.append(f"p{i}: |S| {mags_t[i]:.3f}->{mags_p[i]:.3f}, ∠ {ph_t[i]:+.2f}->{ph_p[i]:+.2f}")
    return " ; ".join(parts)


def _ensure_model_pml_cells(model_like, pml_cells: int) -> None:
    """
    Ensure flow_matching.py can reliably read model.(module).helmholtz.pml_cells for PML masking.
    """
    try:
        m = getattr(model_like, "module", model_like)
        if hasattr(m, "helmholtz") and (getattr(m, "helmholtz") is not None):
            h = getattr(m, "helmholtz")
            if hasattr(h, "pml_cells"):
                try:
                    h.pml_cells = int(pml_cells)
                    return
                except Exception:
                    pass
        m.helmholtz = SimpleNamespace(pml_cells=int(pml_cells))
    except Exception:
        pass


def _parse_list(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def main(args):
    assert torch.cuda.is_available(), "Need CUDA GPUs for DDP training."
    dist.init_process_group(backend="nccl")

    # Optional: force all enabled losses to start from epoch 1 (disable phased curriculum).
    # Note: a loss only participates if its corresponding lambda > 0.
    if bool(getattr(args, "all_losses_from_start", False)):
        # Collapse curriculum phases entirely: treat the whole run as a single phase starting at epoch 1.
        args.phaseA_epochs = 0
        args.phaseB_epochs = 0

        # Start phases at A (epoch 0 => active on epoch 1 loop)
        args.residual_start_phase = "A"
        args.phase_start_phase = "A"
        args.phase_grad_start_phase = "A"
        args.sparam_start_phase = "A"
        args.sparam_from_start = True

        # Remove warmups so weights are nonzero immediately (if lambdas are nonzero)
        args.residual_warmup_epochs = 0
        args.phase_warmup_epochs = 0
        args.phase_grad_warmup_epochs = 0
        args.sparam_warmup_epochs = 0
        args.endpoint_warmup_epochs = 0

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_dir = "logs_physics_unet_pbfm"
    experiment_dir = os.path.join(results_dir, args.version)
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    samples_dir = os.path.join(experiment_dir, "samples")

    if is_rank0():
        os.makedirs(results_dir, exist_ok=True)
        if (not args.resume_from) and os.path.exists(experiment_dir):
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

    ddp_barrier(device)
    logger = create_logger(experiment_dir if is_rank0() else None)
    if is_rank0():
        logger.info(f"DDP: rank={rank}/{world}, local_rank={local_rank}, device={device}")
    logger.info(f"Experiment dir: {experiment_dir}")
    # NOTE: when --all-losses-from-start is enabled we collapse phaseA/phaseB to 0 above,
    # so @B/@C tau schedules effectively start immediately.

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

    wandb_run = None
    if args.use_wandb and is_rank0():
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.version,
            config=vars(args),
        )

    include_sweeps = None
    if args.include_sweeps:
        include_sweeps = [s.strip() for s in args.include_sweeps.split(",") if s.strip()]

    # sdf sigma px -> nm
    sdf_sigma_px = float(getattr(args, "sdf_sigma_px", 0.0) or 0.0)
    if sdf_sigma_px and sdf_sigma_px > 0:
        dx_um = float(getattr(args, "dx", 1.0 / 24.0))
        args.sdf_sigma_nm = float(sdf_sigma_px) * dx_um * 1000.0

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

    if subset_train and (not subset_val):
        val_total = int(getattr(args, "subset_val_total", 0))
        train_total = sum(subset_train.values())
        if val_total <= 0:
            val_total = max(1, int(round(0.25 * train_total)))
        for k, n in subset_train.items():
            subset_val[k] = int(round(val_total * (n / max(train_total, 1))))
        drift = val_total - sum(subset_val.values())
        if drift != 0 and subset_val:
            keys = sorted(subset_val.keys())
            step = 1 if drift > 0 else -1
            for i in range(abs(drift)):
                subset_val[keys[i % len(keys)]] += step

    # Choose dataset class based on --use-fast-dataset flag
    use_fast = bool(getattr(args, "use_fast_dataset", False))

    stats = None
    if use_fast:
        # Fast dataset: load preprocessed .pt files
        if is_rank0():
            logger.info("Using FastFDTDDataset (preprocessed .pt files)")
            train_ds_tmp = FastFDTDDataset(
                preprocessed_dir=args.data_root,
                split="train",
                train_fraction=args.train_fraction,
                augment=False,
                return_aux=True,
                use_index_split=bool(getattr(args, "use_index_split", False)),
            )
            stats = train_ds_tmp.get_stats()
        stats = broadcast_object(stats, src=0)

        train_ds = FastFDTDDataset(
            preprocessed_dir=args.data_root,
            split="train",
            train_fraction=args.train_fraction,
            stats=stats,
            augment=bool(getattr(args, "augment", True)),
            return_aux=True,
            use_index_split=bool(getattr(args, "use_index_split", False)),
        )
        val_ds = FastFDTDDataset(
            preprocessed_dir=args.data_root,
            split="val",
            train_fraction=args.train_fraction,
            stats=stats,
            augment=False,
            return_aux=True,
            use_index_split=bool(getattr(args, "use_index_split", False)),
        )
    else:
        # Regular dataset: load from NPZ shards
        if is_rank0():
            train_ds_tmp = FDTDDataset(
                root_dir=args.data_root,
                split="train",
                train_fraction=args.train_fraction,
                normalize_eps=args.normalize_eps,
                include_sdf=bool(getattr(args, "include_sdf", False)),
                normalize_sdf=bool(getattr(args, "normalize_sdf", True)),
                sdf_thr_eps=float(getattr(args, "sdf_thr_eps", 3.0)),
                sdf_dx_um=float(getattr(args, "dx", 1.0 / 24.0)),
                sdf_dy_um=float(getattr(args, "dx", 1.0 / 24.0)),
                sdf_feature=str(getattr(args, "sdf_feature", "raw")),
                sdf_sigma_nm=float(getattr(args, "sdf_sigma_nm", 100.0)),
                use_shards=args.use_shards,
                shard_subdir=args.shard_subdir,
                shard_index_name=args.shard_index_name,
                include_sweeps=include_sweeps,
                subset_train_per_sweep=subset_train if subset_train else None,
                subset_val_per_sweep=subset_val if subset_train or subset_val else None,
                subset_seed=subset_seed,
                crop_pml=bool(getattr(args, "crop_pml", False)),
                pml_cells=int(getattr(args, "pml_cells", 0)),
                return_aux=True,
                use_index_split=bool(getattr(args, "use_index_split", False)),
            )
            stats = train_ds_tmp.get_stats()
        stats = broadcast_object(stats, src=0)

        train_ds = FDTDDataset(
            root_dir=args.data_root,
            split="train",
            train_fraction=args.train_fraction,
            stats=stats,
            normalize_eps=args.normalize_eps,
            include_sdf=bool(getattr(args, "include_sdf", False)),
            normalize_sdf=bool(getattr(args, "normalize_sdf", True)),
            sdf_thr_eps=float(getattr(args, "sdf_thr_eps", 3.0)),
            sdf_dx_um=float(getattr(args, "dx", 1.0 / 24.0)),
            sdf_dy_um=float(getattr(args, "dx", 1.0 / 24.0)),
            sdf_feature=str(getattr(args, "sdf_feature", "raw")),
            sdf_sigma_nm=float(getattr(args, "sdf_sigma_nm", 100.0)),
            use_shards=args.use_shards,
            shard_subdir=args.shard_subdir,
            shard_index_name=args.shard_index_name,
            include_sweeps=include_sweeps,
            return_aux=True,
            subset_train_per_sweep=subset_train if subset_train else None,
            subset_val_per_sweep=subset_val if subset_train or subset_val else None,
            subset_seed=subset_seed,
            crop_pml=bool(getattr(args, "crop_pml", False)),
            pml_cells=int(getattr(args, "pml_cells", 0)),
            augment=bool(getattr(args, "augment", True)),  # D4 augmentation during training
            use_index_split=bool(getattr(args, "use_index_split", False)),
        )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )

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
    dx = float(args.dx)
    lam0 = float(args.lambda_um)
    omega = 2.0 * math.pi / lam0

    in_channels = int(getattr(train_ds, "x_channels", int(stats.get("x_channels", 4))))
    enable_physics = bool(getattr(args, "physics_features", True))
    use_complex = bool(getattr(args, "complex_unet", False))

    if is_rank0():
        logger.info("Using ComplexPhysicsUNet" if use_complex else "Using PhysicsUNet")
        logger.info("Physics features ENABLED" if enable_physics else "Physics features DISABLED (vanilla UNet ablation)")
        try:
            sample0 = train_ds[0]
            x0 = sample0[0] if isinstance(sample0, (tuple, list)) else sample0
            h, w = int(x0.shape[-2]), int(x0.shape[-1])
            logger.info(f"Data resolution: {h}x{w} (attention at ds=8 => {h//8}x{w//8} tokens)")
        except Exception as exc:  # pragma: no cover
            logger.info(f"Data resolution: <unknown> (probe failed: {exc})")

    attn_res = () if args.no_attention else (8,)
    model_kwargs = dict(
        in_channels=in_channels,
        out_channels=2,
        model_channels=args.hidden_size,
        num_res_blocks=3,
        channel_mult=(1, 2, 4, 8,),
        attention_resolutions=attn_res,
        dropout=0.0,
        dims=2,
        num_heads=4,
        cond_dim=int(getattr(train_ds, "cond_dim", 1)),
        enable_sparam_head=(args.lambda_sparam > 0 and str(getattr(args, "sparam_mode", "project")) == "head"),
        dx=dx,
        dy=dx,
        omega=omega,
        pml_cells=int(getattr(args, "pml_cells", 0)),
        enable_physics_features=enable_physics,
        use_checkpoint=bool(getattr(args, "use_checkpoint", False)),
    )

    if use_complex:
        base_model = ComplexPhysicsUNet(**model_kwargs).to(device)
    else:
        base_model = PhysicsUNet(**model_kwargs).to(device)

    base_model.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    ema = deepcopy(base_model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

    enable_head = bool(getattr(base_model, "enable_sparam_head", False))
    ddp_find_unused = bool(
        enable_head and (
            int(getattr(args, "sparam_every", 1)) != 1
            or (not bool(getattr(args, "sparam_from_start", False)))
        )
    )

    model = DDP(
        base_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=ddp_find_unused,
        broadcast_buffers=False,
    )

    if is_rank0():
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Weighted-sum objective only (no AugLag / no MGDA / no PCGrad).

    # Optimizer (model params only)
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=0.0,
    )

    amp_enabled = bool(getattr(args, "amp", True))
    if _AMP_NEW_API:
        scaler = _GradScaler("cuda", enabled=amp_enabled)
        autocast_device_type = "cuda"
    else:
        scaler = _GradScaler(enabled=amp_enabled)
        autocast_device_type = "cuda"
    amp_enabled = scaler.is_enabled()

    def optimizer_step():
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    min_lr = max(float(getattr(args, "min_lr", 5e-6)), 0.0)
    if warmup_epochs > 0:
        warmup_start_factor = float(getattr(args, "warmup_start_factor", 0.1))
        warmup_start_factor = min(max(warmup_start_factor, 1e-6), 1.0)
        warmup = LinearLR(opt, start_factor=warmup_start_factor, total_iters=warmup_epochs)
        cos_T = max(1, args.epochs - warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=cos_T, eta_min=min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=min_lr)

    pml_cells = int(getattr(args, "pml_cells", 0))
    helmholtz_op = HelmholtzResidual2D(
        dx=dx, dy=dx, omega=omega, pml_cells=pml_cells, normalize=False
    ).to(device)

    _ensure_model_pml_cells(model, pml_cells=pml_cells)
    _ensure_model_pml_cells(ema, pml_cells=pml_cells)

    update_ema(ema, model.module, decay=0.0)

    # -----------------------
    # Resume
    # -----------------------
    start_epoch = 1
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        model.module.load_state_dict(checkpoint["model"], strict=False)
        ema.load_state_dict(checkpoint["ema"], strict=False)

        opt.load_state_dict(checkpoint["opt"])

        model.module.set_normalization_stats(stats, normalize_eps=args.normalize_eps)
        ema.set_normalization_stats(stats, normalize_eps=args.normalize_eps)

        _ensure_model_pml_cells(model, pml_cells=pml_cells)
        _ensure_model_pml_cells(ema, pml_cells=pml_cells)

        if is_rank0():
            logger.info(f"Resumed from {args.resume_from}")

        try:
            ckpt_name = os.path.basename(args.resume_from)
            last_epoch = int(os.path.splitext(ckpt_name)[0])
        except Exception:
            last_epoch = 0
        start_epoch = last_epoch + 1
        scheduler.last_epoch = last_epoch

    # -----------------------
    # Training settings
    # -----------------------
    use_config = bool(args.config)
    config_start_epoch = max(1, int(getattr(args, "config_start_epoch", 1)))
    use_endpoint = args.lambda_endpoint > 0
    use_residual = args.lambda_residual > 0
    use_phase = args.lambda_phase > 0
    use_phase_grad = args.lambda_phase_grad > 0
    use_sparam = args.lambda_sparam > 0

    if use_config and (not _HAS_CONFLICTFREE) and is_rank0():
        logger.warning("ConFIG requested (--config) but 'conflictfree' is not available; disabling ConFIG and using plain weighted sum.")
    config_active_global = bool(use_config and _HAS_CONFLICTFREE)

    lam_mean = float(stats["lambda_um_mean"])
    lam_std = float(stats["lambda_um_std"])

    try:
        args.cond_dim = int(getattr(train_ds, "cond_dim", 1))
    except Exception:
        args.cond_dim = 1

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
        logger.info("Objective: fixed weighted sum")
        cfg_log = f"enabled={config_active_global} start_epoch={config_start_epoch}" if use_config else "disabled"
        logger.info(f"Conflict-free grads (ConFIG): {cfg_log}")

    detect_anomaly = bool(getattr(args, "detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if is_rank0():
            logger.info("Autograd anomaly detection ENABLED (--detect-anomaly).")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        N1 = int(args.phaseA_epochs)
        N2 = int(args.phaseB_epochs)

        endpoint_weight = float(args.lambda_endpoint)
        if int(args.endpoint_warmup_epochs) > 0 and endpoint_weight > 0:
            endpoint_weight *= min(1.0, epoch / float(args.endpoint_warmup_epochs))

        if use_residual:
            residual_start = 0 if getattr(args, "residual_start_phase", "B") == "A" else N1
            residual_weight = float(args.lambda_residual) * ramp_linear(
                epoch, start_epoch=residual_start, warmup_epochs=int(args.residual_warmup_epochs)
            )
        else:
            residual_weight = 0.0

        if use_phase:
            phase_start = _phase_start_epoch(getattr(args, "phase_start_phase", "C"), N1=N1, N2=N2)
            phase_weight = float(args.lambda_phase) * ramp_linear(
                epoch, start_epoch=phase_start, warmup_epochs=int(args.phase_warmup_epochs)
            )
        else:
            phase_weight = 0.0

        phys_gate = 0.0
        if float(args.lambda_residual) > 0:
            phys_gate = float(residual_weight / float(args.lambda_residual))
            phys_gate = max(0.0, min(1.0, phys_gate))

        phase_gate = 0.0
        if float(args.lambda_phase) > 0:
            phase_gate = float(phase_weight / float(args.lambda_phase))
            phase_gate = max(0.0, min(1.0, phase_gate))

        if use_phase_grad:
            phase_grad_start = _phase_start_epoch(getattr(args, "phase_grad_start_phase", "C"), N1=N1, N2=N2)
            phase_grad_weight = float(args.lambda_phase_grad) * ramp_linear(
                epoch, start_epoch=phase_grad_start, warmup_epochs=int(args.phase_grad_warmup_epochs)
            )
        else:
            phase_grad_weight = 0.0

        compute_phase_epoch = (phase_weight > 0.0)

        if use_sparam:
            if bool(getattr(args, "sparam_from_start", False)):
                sparam_weight = float(args.lambda_sparam) * ramp_linear(
                    epoch, start_epoch=0, warmup_epochs=int(args.sparam_warmup_epochs)
                )
            else:
                sp_phase = str(getattr(args, "sparam_start_phase", "C")).strip().upper()
                if sp_phase == "A":
                    sparam_start = 0
                elif sp_phase == "B":
                    sparam_start = N1
                else:
                    sparam_start = N2
                sparam_weight = float(args.lambda_sparam) * ramp_linear(
                    epoch, start_epoch=sparam_start, warmup_epochs=int(args.sparam_warmup_epochs)
                )
        else:
            sparam_weight = 0.0

        focus_start = 1.0
        focus_end = 1.0
        if int(args.device_focus_warmup_epochs) > 0:
            ramp = min(1.0, epoch / float(args.device_focus_warmup_epochs))
        else:
            ramp = 1.0
        device_focus = focus_start + (float(args.device_focus_max) - focus_start) * ramp
        if int(args.device_focus_hold_epochs) > 0:
            if epoch <= int(args.device_focus_warmup_epochs) + int(args.device_focus_hold_epochs):
                device_focus = float(args.device_focus_max)
        if int(args.device_focus_decay_epochs) > 0:
            e0 = int(args.device_focus_warmup_epochs) + int(args.device_focus_hold_epochs)
            if epoch > e0:
                frac = min(1.0, (epoch - e0) / float(args.device_focus_decay_epochs))
                device_focus = device_focus * (1.0 - frac) + focus_end * frac

        use_tqdm = bool(getattr(args, "tqdm", False)) and (tqdm is not None) and is_rank0()
        it = train_loader
        if use_tqdm:
            it = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"epoch {epoch}/{args.epochs}",
                dynamic_ncols=True,
                mininterval=float(getattr(args, "tqdm_mininterval", 1.0)),
            )

        for x_full, cond, aux in it:
            x_full = x_full.to(device, dtype=dtype, non_blocking=True)
            cond = cond.to(device, dtype=dtype, non_blocking=True)
            aux = _move_aux_to_device(aux, device)

            fields_1 = x_full[:, 0:2]
            eps = x_full[:, 2:3]
            src = x_full[:, 3:4]
            extra_maps = x_full[:, 4:] if x_full.shape[1] > 4 else None

            lambda_um = cond[:, 0:1] * lam_std + lam_mean  # [B,1] physical

            x0_fields = torch.randn_like(fields_1)
            t = sample_t(fields_1)  # [B,1,1,1]
            x_t_fields = psi_t(x0_fields, fields_1, t)
            v_t_fields = u_t(x0_fields, fields_1)

            if extra_maps is not None:
                x_t_input = torch.cat([x_t_fields, eps, src, extra_maps], dim=1)
            else:
                x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

            compute_phase_grad_step = (phase_grad_weight > 0.0) and (global_step % int(args.phase_grad_every) == 0)
            compute_sparam_step = (sparam_weight > 0.0) and (global_step % int(args.sparam_every) == 0)
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
                unroll_steps=int(getattr(args, "unroll_steps", 0)),
                unroll_phase=bool(getattr(args, "unroll_phase", False)),
                phase_amp_tau=float(getattr(args, "phase_amp_tau", 0.2)),
                amp_enabled=amp_enabled,
                amp_device_type=autocast_device_type,
                amp_dtype=torch.float16,
                sparam_mode=str(getattr(args, "sparam_mode", "project")),
            )

            residual_loss = residual_loss * (dx ** 4)

            # -----------------------
            # Step
            # -----------------------
            opt.zero_grad(set_to_none=True)

            # ConFIG (conflictfree) branch: combine per-loss gradients with ConFIG_update.
            # This uses the library's get/apply_gradient_vector helpers (like your reference repo).
            config_active = bool(config_active_global) and (epoch >= config_start_epoch)
            if config_active:
                # IMPORTANT:
                # DDP does not support multiple backward passes on the same graph in one iteration
                # when different losses touch different parameter subsets (e.g. sparam head).
                # So we avoid `.backward()` entirely here and compute per-loss gradients with
                # `torch.autograd.grad`, then all-reduce the *combined* ConFIG gradient vector once.
                grads: list[torch.Tensor] = []
                loss_items: list[tuple[str, torch.Tensor, float]] = [("fm", fm_loss, 1.0)]
                if endpoint_weight > 0.0:
                    loss_items.append(("end", endpoint_loss, float(endpoint_weight)))
                if residual_weight > 0.0:
                    loss_items.append(("res", residual_loss, float(residual_weight)))
                if phase_weight > 0.0:
                    loss_items.append(("phase", phase_loss, float(phase_weight)))
                if compute_phase_grad_step and phase_grad_weight > 0.0:
                    loss_items.append(("phg", phase_grad_loss, float(phase_grad_weight)))
                if compute_sparam_step and sparam_weight > 0.0:
                    loss_items.append(("sp", sparam_loss, float(sparam_weight)))

                # Compute a gradient vector per loss term (unweighted, matching your reference implementation).
                # Note: weights still control participation (which losses are included).
                params = [p for p in model.module.parameters() if p.requires_grad]
                for i, (_name, L, w) in enumerate(loss_items):
                    if float(w) <= 0.0:
                        continue
                    retain = (i != len(loss_items) - 1)

                    g_list = torch.autograd.grad(
                        outputs=L,
                        inputs=params,
                        retain_graph=retain,
                        create_graph=False,
                        allow_unused=True,
                    )
                    vec_parts = []
                    for p, g in zip(params, g_list):
                        if g is None:
                            vec_parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
                        else:
                            vec_parts.append(g.detach().to(dtype=torch.float32).reshape(-1))
                    g_vec = torch.cat(vec_parts, dim=0)
                    g_vec = torch.nan_to_num(g_vec, nan=0.0, posinf=0.0, neginf=0.0)
                    grads.append(g_vec)

                # Fallback to fm-only if any gradient vector contains NaNs.
                g_config = None
                if len(grads) > 0 and (not grads[0].isnan().any()):
                    ok = True
                    for g in grads:
                        if g is None or g.isnan().any() or (not torch.isfinite(g).all()):
                            ok = False
                            break
                    if ok:
                        g_config = _ConFIG_update(grads)

                if g_config is None:
                    g_step = grads[0] if len(grads) > 0 else None
                else:
                    g_step = g_config

                if g_step is None:
                    # ultra fallback (shouldn't happen): do plain FM backward
                    opt.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(fm_loss).backward()
                        scaler.unscale_(opt)
                    else:
                        fm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_step()
                else:
                    # DDP-sync: average the combined grad vector across ranks once
                    if dist.is_initialized():
                        dist.all_reduce(g_step, op=dist.ReduceOp.SUM)
                        g_step = g_step / float(dist.get_world_size())

                    # Apply vector to parameter .grad and step
                    opt.zero_grad(set_to_none=True)
                    start = 0
                    with torch.no_grad():
                        for p in params:
                            n = p.numel()
                            p.grad = g_step[start : start + n].view_as(p).to(dtype=p.dtype)
                            start += n
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            else:
                # Plain weighted sum (default / when ConFIG disabled)
                    total_loss = fm_loss
                    if endpoint_weight > 0.0:
                        total_loss = total_loss + endpoint_weight * endpoint_loss
                    if residual_weight > 0.0:
                        total_loss = total_loss + residual_weight * residual_loss
                    if phase_weight > 0.0:
                        total_loss = total_loss + phase_weight * phase_loss
                    if compute_phase_grad_step and phase_grad_weight > 0.0:
                        total_loss = total_loss + phase_grad_weight * phase_grad_loss
                    if compute_sparam_step and sparam_weight > 0.0:
                        total_loss = total_loss + sparam_weight * sparam_loss

                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(opt)
                    else:
                        total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_step()

            if use_tqdm:
                try:
                    it.set_postfix(
                        fm=float(fm_loss.item()),
                        res=float(residual_loss.item()) if residual_weight > 0.0 else 0.0,
                        ph=float(phase_loss.item()) if phase_weight > 0.0 else 0.0,
                        end=float(endpoint_loss.item()) if endpoint_weight > 0.0 else 0.0,
                        sp=float(sparam_loss.item()) if compute_sparam_step and sparam_weight > 0.0 else 0.0,
                        lr=float(opt.param_groups[0]["lr"]),
                    )
                except Exception:
                    pass

            update_ema(ema, model.module)

            running_fm_loss += float(fm_loss.item())
            running_residual_loss += float(residual_loss.item())
            if phase_weight > 0.0:
                running_phase_loss += float(phase_loss.item())
            if use_endpoint:
                running_endpoint_loss += float(endpoint_loss.item())
            if compute_phase_grad_step:
                running_phase_grad_loss += float(phase_grad_loss.item())
                phase_grad_steps_epoch += 1
            if compute_sparam_step:
                running_sparam_loss += float(sparam_loss.item())
                sparam_steps_epoch += 1

            train_steps_epoch += 1
            global_step += 1

        # -----------------------
        # Validation
        # -----------------------
        if epoch % int(args.eval_every) == 0:
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
                sparams_example = None

                max_val_batches = int(args.val_batches)

                for i, (x_full, cond, aux) in enumerate(val_loader):
                    x_full = x_full.to(device, dtype=dtype, non_blocking=True)
                    cond = cond.to(device, dtype=dtype, non_blocking=True)
                    aux = _move_aux_to_device(aux, device)

                    fields_1 = x_full[:, 0:2]
                    eps = x_full[:, 2:3]
                    src = x_full[:, 3:4]
                    extra_maps = x_full[:, 4:] if x_full.shape[1] > 4 else None
                    lambda_um = cond[:, 0:1] * lam_std + lam_mean

                    x0_fields = torch.randn_like(fields_1)
                    t = sample_t(fields_1)
                    x_t_fields = psi_t(x0_fields, fields_1, t)
                    v_t_fields = u_t(x0_fields, fields_1)

                    if extra_maps is not None:
                        x_t_input = torch.cat([x_t_fields, eps, src, extra_maps], dim=1)
                    else:
                        x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

                    compute_phase_grad_val = (phase_grad_weight > 0.0)
                    compute_residual_val = (residual_weight > 0.0)
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
                        unroll_steps=int(getattr(args, "unroll_steps", 0)),
                        unroll_phase=bool(getattr(args, "unroll_phase", False)),
                        phase_amp_tau=float(getattr(args, "phase_amp_tau", 0.2)),
                        amp_enabled=amp_enabled,
                        amp_device_type=autocast_device_type,
                        amp_dtype=torch.float16,
                        sparam_mode=str(getattr(args, "sparam_mode", "project")),
                    )

                    res_v = res_v * (dx ** 4)

                    eval_fm += float(fm_v.item())
                    eval_res += float(res_v.item())
                    if phase_weight > 0.0:
                        eval_phase += float(ph_v.item())
                    if use_endpoint:
                        eval_endpoint += float(end_v.item())
                    if compute_phase_grad_val:
                        eval_phase_grad += float(phg_v.item())
                    if compute_sparam_val:
                        eval_sparam += float(sp_v.item())

                    if (
                        is_rank0()
                        and (sparams_example is None)
                        and compute_sparam_val
                        and (aux is not None)
                        and (aux.get("sparams_true", None) is not None)
                        and (aux.get("port_masks", None) is not None)
                    ):
                        B = int(x_t_input.shape[0])
                        t_vec = t.view(B) if t.dim() > 1 else t
                        if str(getattr(args, "sparam_mode", "project")) == "head":
                            _, S_pred = ema(
                                x_t_input,
                                t_vec,
                                cond=cond,
                                lambda_um=lambda_um,
                                phys_gate=phys_gate,
                                phase_gate=phase_gate,
                                return_sparams=True,
                                aux=aux,
                                sig_min=SIG_MIN,
                            )
                        else:
                            u_t_pred = ema(
                                x_t_input,
                                t_vec,
                                cond=cond,
                                lambda_um=lambda_um,
                                phys_gate=phys_gate,
                                phase_gate=phase_gate,
                            )
                            fields_t = x_t_input[:, 0:2]
                            t4 = t_vec.view(B, 1, 1, 1)
                            a = (1.0 - float(SIG_MIN))
                            b = (1.0 - a * t4)
                            x1_fields_pred = a * fields_t + b * u_t_pred  # normalized
                            Er = x1_fields_pred[:, 0] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                            Ei = x1_fields_pred[:, 1] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                            E_pred_phys_raw = torch.complex(Er, Ei)
                            S_pred = extract_sparams(
                                E_pred_phys_raw,
                                aux["port_masks"],
                                in_port_idx=aux.get("in_port_idx", None),
                                port_ids=aux.get("port_ids", None),
                            )
                        S_true = aux["sparams_true"]
                        pv0 = aux.get("port_valid", None)
                        pv0 = pv0[0].detach() if pv0 is not None else None
                        sparams_example = (S_true[0].detach(), S_pred[0].detach(), pv0)

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
                if phase_weight > 0.0:
                    ddp_allreduce_mean_(ph_t)
                if use_endpoint:
                    ddp_allreduce_mean_(end_t)
                if phase_grad_weight > 0.0:
                    ddp_allreduce_mean_(phg_t)
                if sparam_weight > 0.0:
                    ddp_allreduce_mean_(sp_t)

                val_fm_loss = float(fm_t.item())
                val_res_loss = float(res_t.item())
                val_phase_loss = float(ph_t.item()) if (phase_weight > 0.0) else 0.0
                val_endpoint_loss = float(end_t.item()) if use_endpoint else 0.0
                val_phase_grad_loss = float(phg_t.item()) if (phase_grad_weight > 0.0) else 0.0
                val_sparam_loss = float(sp_t.item()) if (sparam_weight > 0.0) else 0.0

            # Sample eval (rank0 only)
            sample_residual_mean = 0.0
            sample_amp_err_mean = 0.0
            sample_phase_err_mean = 0.0
            if is_rank0():
                residuals = []
                amp_errs = []
                phase_errs = []
                saved_any = False
                with torch.no_grad():
                    n_samples = min(len(val_ds), int(args.sample_eval_limit))
                    for s in range(n_samples):
                        x_full_s, cond_s, aux_s = val_ds[s]
                        x_full_s = x_full_s.unsqueeze(0).to(device, dtype=dtype)
                        cond_s = cond_s.unsqueeze(0).to(device, dtype=dtype)

                        eps_s = x_full_s[:, 2:3]
                        src_s = x_full_s[:, 3:4]
                        extra_maps_s = x_full_s[:, 4:] if x_full_s.shape[1] > 4 else None
                        x0_fields_s = torch.randn_like(x_full_s[:, 0:2])
                        lambda_um_s = cond_s[:, 0:1] * lam_std + lam_mean

                        if extra_maps_s is not None:
                            cond_maps_s = torch.cat([eps_s, src_s, extra_maps_s], dim=1)
                        else:
                            cond_maps_s = torch.cat([eps_s, src_s], dim=1)

                        x1_fields_pred = fm_sample(
                            ema,
                            x0_fields_s,
                            num_steps=int(args.fm_steps),
                            use_stoc_samp=bool(args.use_stoc_samp),
                            cond_maps=cond_maps_s,
                            cond=cond_s,
                            lambda_um=lambda_um_s,
                            phys_gate=1.0,
                            phase_gate=1.0,
                            sig_min=SIG_MIN,
                        )
                        x1_pred = torch.cat([x1_fields_pred, eps_s], dim=1)
                        x1_pred[:, 0] = x1_pred[:, 0] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        x1_pred[:, 1] = x1_pred[:, 1] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        if args.normalize_eps:
                            x1_pred[:, 2] = x1_pred[:, 2] * float(stats["eps_std"]) + float(stats["eps_mean"])

                        B = x1_pred.shape[0]
                        k0 = (2.0 * torch.pi) / lambda_um_s.view(B)
                        R = helmholtz_op(x1_pred[:, 0:2], x1_pred[:, 2:3], k0=k0)
                        res_mag = torch.sqrt(R[:, 0:1] ** 2 + R[:, 1:2] ** 2 + 1e-12) * (dx ** 2)
                        residuals.append(float(res_mag.mean().item()))

                        ezr_gt_s = x_full_s[:, 0:1] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        ezi_gt_s = x_full_s[:, 1:2] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        mag_gt_s = torch.sqrt(ezr_gt_s ** 2 + ezi_gt_s ** 2 + 1e-12)
                        mag_pred_s = torch.sqrt(x1_pred[:, 0:1] ** 2 + x1_pred[:, 1:2] ** 2 + 1e-12)

                        eps_phys_s = x1_pred[:, 2:3]
                        m = (eps_phys_s > float(args.eps_thr)).to(dtype=torch.float32)
                        k = int(getattr(args, "dilate", 1))
                        if k > 1:
                            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2).clamp(0.0, 1.0)
                        denom = float(m.sum().item())
                        if denom < 1.0:
                            m = torch.ones_like(m)

                        amp_err_map_s = torch.abs(mag_pred_s - mag_gt_s)
                        amp_errs.append(float((amp_err_map_s * m).sum().item() / (m.sum().item() + 1e-12)))

                        phase_gt_s = torch.atan2(ezi_gt_s, ezr_gt_s)
                        phase_pred_s = torch.atan2(x1_pred[:, 1:2], x1_pred[:, 0:1])
                        phase_err_s = torch.atan2(torch.sin(phase_pred_s - phase_gt_s), torch.cos(phase_pred_s - phase_gt_s))
                        phase_errs.append(float((torch.abs(phase_err_s) * m).sum().item() / (m.sum().item() + 1e-12)))

                        # Save a qualitative sample panel (rank0 only).
                        # Default behavior: save on every eval epoch, for the first sample only.
                        save_eval_samples = bool(getattr(args, "save_eval_samples", True))
                        save_limit = int(getattr(args, "save_eval_samples_limit", 1))
                        if save_eval_samples and (s < max(0, save_limit)):
                            # eps (phys) for display: prefer GT eps if available
                            eps_gt_phys = x_full_s[:, 2:3]
                            if args.normalize_eps:
                                eps_gt_phys = eps_gt_phys * float(stats["eps_std"]) + float(stats["eps_mean"])
                            eps_img = eps_gt_phys[0, 0].detach().float().cpu().numpy()

                            ezr_gt_img = ezr_gt_s[0, 0].detach().float().cpu().numpy()
                            ezi_gt_img = ezi_gt_s[0, 0].detach().float().cpu().numpy()
                            ezr_pred_img = x1_pred[:, 0:1][0, 0].detach().float().cpu().numpy()
                            ezi_pred_img = x1_pred[:, 1:2][0, 0].detach().float().cpu().numpy()

                            out_path = os.path.join(samples_dir, f"sample_epoch_{epoch:04d}_idx{s:03d}.png")
                            title = f"epoch={epoch:04d} idx={s:03d}"
                            _save_eval_sample_png(
                                out_path=out_path,
                                title=title,
                                eps_phys=eps_img,
                                ezr_gt=ezr_gt_img,
                                ezi_gt=ezi_gt_img,
                                ezr_pred=ezr_pred_img,
                                ezi_pred=ezi_pred_img,
                            )
                            if os.path.isfile(out_path):
                                saved_any = True

                sample_residual_mean = float(np.mean(residuals)) if residuals else 0.0
                sample_amp_err_mean = float(np.mean(amp_errs)) if amp_errs else 0.0
                sample_phase_err_mean = float(np.mean(phase_errs)) if phase_errs else 0.0

            if is_rank0():
                msg = f"[epoch {epoch:04d}] val_fm={val_fm_loss:.4e}, val_res={val_res_loss:.4e}"
                if phase_weight > 0.0:
                    msg += f", val_phase={val_phase_loss:.4e}"
                    try:
                        val_phase_rms_deg = (math.sqrt(max(0.0, 2.0 * val_phase_loss)) * 180.0 / math.pi)
                        msg += f" (≈{val_phase_rms_deg:.2f}° rms)"
                    except Exception:
                        pass
                if use_endpoint:
                    msg += f", val_endpoint={val_endpoint_loss:.4e}"
                if phase_grad_weight > 0.0:
                    msg += f", val_phase_grad={val_phase_grad_loss:.4e}"
                if sparam_weight > 0.0:
                    msg += f", val_sparam={val_sparam_loss:.4e}"
                msg += (
                    f", sample_residual={sample_residual_mean:.4e}"
                    f", sample_amp_err={sample_amp_err_mean:.4e}"
                    f", sample_phase_err={sample_phase_err_mean:.4e}"
                )
                logger.info(msg)
                if saved_any:
                    logger.info(f"[epoch {epoch:04d}] Saved eval sample(s) to {samples_dir}")

                if (sparam_weight > 0.0) and (sparams_example is not None):
                    S_true_0, S_pred_0, pv0 = sparams_example
                    logger.info(
                        f"[epoch {epoch:04d}] Sparams example (true->pred): "
                        f"{_format_sparams_compact(S_true_0, S_pred_0, port_valid_1d=pv0)}"
                    )

                with open(val_csv_path, "a", encoding="UTF8", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
                        epoch,
                        val_fm_loss,
                        val_res_loss,
                        val_phase_loss if (phase_weight > 0.0) else "",
                        val_endpoint_loss if use_endpoint else "",
                        val_phase_grad_loss if (phase_grad_weight > 0.0) else "",
                        val_sparam_loss if (sparam_weight > 0.0) else "",
                        sample_residual_mean,
                    ])

                if wandb_run is not None:
                    log_dict = {
                        "epoch": epoch,
                        "val/fm_loss": val_fm_loss,
                        "val/residual_loss": val_res_loss,
                        "val/sample_residual": sample_residual_mean,
                        "val/sample_amp_err": sample_amp_err_mean,
                        "val/sample_phase_err": sample_phase_err_mean,
                    }
                    if phase_weight > 0.0:
                        log_dict["val/phase_loss"] = val_phase_loss
                    if use_endpoint:
                        log_dict["val/endpoint_loss"] = val_endpoint_loss
                    if phase_grad_weight > 0.0:
                        log_dict["val/phase_grad_loss"] = val_phase_grad_loss
                    if sparam_weight > 0.0:
                        log_dict["val/sparam_loss"] = val_sparam_loss
                    wandb_run.log(log_dict, step=epoch)

        # -----------------------
        # Train logging
        # -----------------------
        if epoch % int(args.log_every) == 0:
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
            if phase_weight > 0.0:
                ddp_allreduce_mean_(ph_mean)
            if use_endpoint:
                ddp_allreduce_mean_(end_mean)
            if phase_grad_weight > 0.0:
                ddp_allreduce_mean_(phg_mean)
            if sparam_weight > 0.0:
                ddp_allreduce_mean_(sp_mean)

            if is_rank0():
                msg = f"(epoch={epoch:04d}) train_fm={fm_mean.item():.4e}, train_residual={res_mean.item():.4e}"
                if phase_weight > 0.0:
                    msg += f", train_phase={ph_mean.item():.4e}"
                    try:
                        ph_rms_deg = (math.sqrt(max(0.0, 2.0 * float(ph_mean.item()))) * 180.0 / math.pi)
                        msg += f" (≈{ph_rms_deg:.2f}° rms)"
                    except Exception:
                        pass
                if use_endpoint:
                    msg += f", train_endpoint={end_mean.item():.4e}"
                if phase_grad_weight > 0.0:
                    msg += f", train_phase_grad={phg_mean.item():.4e}"
                if sparam_weight > 0.0:
                    msg += f", train_sparam={sp_mean.item():.4e}"
                msg += f", sec_per_epoch={sec_per_epoch:.3e}"
                msg += f" | w_res={residual_weight:.3g}, w_phase={phase_weight:.3g}, w_phg={phase_grad_weight:.3g}, w_sp={sparam_weight:.3g}"
                if phase_grad_weight > 0.0:
                    msg += f" | phg_steps={int(phase_grad_steps_epoch)}/{int(train_steps_epoch)} (every={int(args.phase_grad_every)})"
                logger.info(msg)

            if wandb_run is not None and is_rank0():
                wandb_log_dict = {
                    "train/fm_loss": fm_mean.item(),
                    "train/residual_loss": res_mean.item(),
                    "train/sec_per_epoch": sec_per_epoch,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/residual_weight": residual_weight,
                    "train/phase_weight": phase_weight,
                    "train/phase_grad_weight": phase_grad_weight,
                    "train/sparam_weight": sparam_weight,
                    "train/device_focus": device_focus,
                    "train/use_config": int(config_active_global and (epoch >= config_start_epoch)),
                }
                if phase_weight > 0.0:
                    wandb_log_dict["train/phase_loss"] = ph_mean.item()
                if use_endpoint:
                    wandb_log_dict["train/endpoint_loss"] = end_mean.item()
                    wandb_log_dict["train/endpoint_weight"] = endpoint_weight
                if phase_grad_weight > 0.0:
                    wandb_log_dict["train/phase_grad_loss"] = phg_mean.item()
                    wandb_log_dict["train/phase_grad_steps_epoch"] = int(phase_grad_steps_epoch)
                if sparam_weight > 0.0:
                    wandb_log_dict["train/sparam_loss"] = sp_mean.item()

                wandb_run.log(wandb_log_dict, step=epoch)

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
        # Checkpoint
        # -----------------------
        if epoch % int(args.ckpt_every) == 0:
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

    parser.add_argument("--use-shards", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shard-subdir", type=str, default="shards")
    parser.add_argument("--shard-index-name", type=str, default="index.json")
    parser.add_argument("--include-sweeps", type=str, default="")
    parser.add_argument("--use-index-split", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Use pre-computed split field from index.json (unified_sweep format)")
    parser.add_argument("--use-fast-dataset", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Use preprocessed .pt files for faster loading (requires running preprocess_dataset.py first)")

    parser.add_argument("--subset-train-per-sweep", type=str, default="")
    parser.add_argument("--subset-val-per-sweep", type=str, default="")
    parser.add_argument("--subset-val-total", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-seed", type=int, default=15)

    parser.add_argument("--version", type=str, default="physics_unet_pbfm_ddp")
    parser.add_argument("--fm_steps", type=int, default=20)

    parser.add_argument("--dx", type=float, default=1.0 / 24.0)
    parser.add_argument("--lambda-um", dest="lambda_um", type=float, default=1.55)

    parser.add_argument("--crop-pml", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pml-cells", type=int, default=0)

    parser.add_argument("--augment", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Enable D4 augmentation during training (8x effective data)")

    parser.add_argument("--include-sdf", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--normalize-sdf", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sdf-thr-eps", dest="sdf_thr_eps", type=float, default=3.0)
    parser.add_argument("--sdf-feature", type=str, default="raw", choices=["raw", "clip", "exp", "clip_exp"])
    parser.add_argument("--sdf-sigma-px", type=float, default=0.0)
    parser.add_argument("--sdf-sigma-nm", type=float, default=100.0)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--no-attention", action="store_true", help="Disable attention layers to reduce memory")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=5e-6)

    # ConFIG conflict-free gradient combination (uses 'conflictfree' library).
    parser.add_argument("--config", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--config-start-epoch", type=int, default=200)

    parser.add_argument("--unroll-steps", type=int, default=0)
    parser.add_argument("--unroll-phase", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--phase-amp-tau", type=float, default=0.2)

    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--detect-anomaly", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-checkpoint", type=bool, default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--phaseA-epochs", dest="phaseA_epochs", type=int, default=50)
    parser.add_argument("--phaseB-epochs", dest="phaseB_epochs", type=int, default=200)

    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--residual-warmup-epochs", type=int, default=50)
    parser.add_argument("--residual-start-phase", type=str, default="B", choices=["A", "B"])

    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--phase-warmup-epochs", type=int, default=50)
    parser.add_argument("--phase-start-phase", type=str, default="C", choices=["A", "B", "C"])

    parser.add_argument("--lambda-endpoint", type=float, default=0.0)
    parser.add_argument("--endpoint-warmup-epochs", type=int, default=0)

    parser.add_argument("--lambda-phase-grad", type=float, default=0.0)
    parser.add_argument("--phase-grad-warmup-epochs", type=int, default=100)
    parser.add_argument("--phase-grad-every", type=int, default=4)
    parser.add_argument("--phase-grad-start-phase", type=str, default="C", choices=["A", "B", "C"])

    parser.add_argument("--lambda-sparam", type=float, default=0.0)
    parser.add_argument("--sparam-from-start", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sparam-warmup-epochs", type=int, default=50)
    parser.add_argument("--sparam-every", type=int, default=1)
    parser.add_argument("--sparam-start-phase", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--sparam-mode", type=str, default="modal", choices=["project", "modal", "head"])

    parser.add_argument("--normalize-eps", type=bool, default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--physics-features", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--complex-unet", type=bool, default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--sample-eval-limit", type=int, default=16)
    parser.add_argument("--save-eval-samples", dest="save_eval_samples", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-eval-samples-limit", dest="save_eval_samples_limit", type=int, default=1)

    parser.add_argument("--resume-from", type=str, default="")

    parser.add_argument("--use-wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project", type=str, default="Rayfield")
    parser.add_argument("--wandb-entity", type=str, default=None)

    parser.add_argument("--device-focus-max", type=float, default=3.0)
    parser.add_argument("--device-focus-warmup-epochs", type=int, default=100)
    parser.add_argument("--device-focus-hold-epochs", type=int, default=0)
    parser.add_argument("--device-focus-decay-epochs", type=int, default=200)

    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--eps-thr", type=float, default=3.0)
    parser.add_argument("--dilate", type=int, default=9)

    parser.add_argument("--val-batches", type=int, default=16)

    parser.add_argument("--tqdm", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tqdm-mininterval", dest="tqdm_mininterval", type=float, default=1.0)

    # (Removed: PCGrad/MGDA/config-method variants; ConFIG is the only supported conflict-free method.)

    # Convenience: disable staged curriculum and start all enabled losses from epoch 1.
    parser.add_argument(
        "--all-losses-from-start",
        dest="all_losses_from_start",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # (Removed: AugLag / adaptive-weighting flags. This script supports fixed weighted-sum only.)

    args = parser.parse_args()
    main(args)
