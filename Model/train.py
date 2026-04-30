# train.py  (train_physics_unet_pbfm_ddp.py)
# Fixed weighted-sum training with optional ConFIG (conflictfree) gradient surgery.

import argparse
import csv
import json
import logging
import math
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from datetime import timedelta
from time import time
from types import SimpleNamespace
from typing import List, Optional

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
from flow_matching import (
    psi_t, u_t, sample_t, cfm_loss_residual, sample as fm_sample, SIG_MIN,
    sample_mask_mode, MASK_MODE_FORWARD, MASK_MODE_INVERSE, MASK_MODE_JOINT,
    binarization_loss, sample_joint, sample_inverse,
)
from sparams_loss import extract_sparams, _resolve_in_idx
from modal_sparams import extract_sparams_modal

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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.preferred_linalg_library("cusolver")

dtype = torch.float32


def _ratio_list_stats(ratios: List[float]) -> dict:
    """Mean, std, p95, max of physical residual ratios (pred/GT)."""
    if not ratios:
        return {"mean": 0.0, "std": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    rs = np.asarray(ratios, dtype=np.float64)
    return {
        "mean": float(rs.mean()),
        "std": float(rs.std(ddof=0)) if rs.size > 1 else 0.0,
        "p95": float(np.percentile(rs, 95.0)),
        "max": float(rs.max()),
        "min": float(rs.min()),
    }


def _eval_field_noise(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Deterministic Gaussian noise for stratified FM sample eval (same every epoch if seed fixed)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return torch.randn(shape, device=device, dtype=dtype, generator=gen)


def _select_eval_sample_indices(
    dt_index: dict[str, list],
    n_per_device: int,
    args,
    epoch: int,
    logger: Optional[logging.Logger],
) -> list[tuple[int, str]]:
    """
    Stratified (global_idx, device_type) pairs for sample eval.
    Modes: epoch_random (default), fixed (same draw every epoch), or JSON file override.
    """
    json_path = (getattr(args, "eval_sample_indices_json", "") or "").strip()
    if json_path:
        if not os.path.isfile(json_path):
            if logger:
                logger.warning(f"eval_sample_indices_json not found ({json_path}); using eval-sample-mode selection.")
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out: list[tuple[int, str]] = []
            for dt_name in sorted(data.keys()):
                if dt_name not in dt_index:
                    if logger:
                        logger.warning(f"eval JSON: device_type '{dt_name}' not in validation set; skipping.")
                    continue
                pool = set(dt_index[dt_name])
                for idx in data[dt_name]:
                    gi = int(idx)
                    if gi in pool:
                        out.append((gi, dt_name))
            if out:
                return out
            if logger:
                logger.warning("eval JSON produced no valid indices; falling back to random selection.")

    mode = str(getattr(args, "eval_sample_mode", "epoch_random"))
    if mode == "fixed":
        rng = np.random.default_rng(int(getattr(args, "eval_sample_index_seed", 0)))
    else:
        rng = np.random.default_rng(int(epoch) * 137 + 7)

    out: list[tuple[int, str]] = []
    for dt_name in sorted(dt_index.keys()):
        dt_pool = dt_index[dt_name]
        n = min(int(n_per_device), len(dt_pool))
        if n <= 0:
            continue
        chosen = rng.choice(len(dt_pool), size=n, replace=False)
        for c in chosen:
            out.append((dt_pool[int(c)], dt_name))
    return out


def _unwrap_model(model):
    """Unwrap compiled / DDP wrappers to get the raw nn.Module."""
    m = model
    if hasattr(m, "_orig_mod"):       # torch.compile wrapper
        m = m._orig_mod
    if hasattr(m, "module"):          # DDP wrapper
        m = m.module
    if hasattr(m, "_orig_mod"):       # in case compile was inside DDP
        m = m._orig_mod
    return m


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


def _save_inverse_eval_png(
    *,
    out_path: str,
    title: str,
    eps_gt: np.ndarray,
    eps_gen: np.ndarray,
    eps_gen_binary: np.ndarray,
    mag_gt: np.ndarray,
    mag_gen: np.ndarray,
    port_masks_overlay: np.ndarray | None,
    sparam_text: str,
    device_type: str,
    ezr_gt: np.ndarray | None = None,
    ezi_gt: np.ndarray | None = None,
    ezr_gen: np.ndarray | None = None,
    ezi_gen: np.ndarray | None = None,
) -> None:
    """
    Save a 3×3 inverse design evaluation panel:
      Row 1: GT eps (+ port masks overlay)  |  Gen eps (continuous)  |  Gen eps (binarized)
      Row 2: GT |Ez|                        |  Gen |Ez|             |  |eps err| map
      Row 3: Ez_real GT vs Gen              |  Ez_imag GT vs Gen    |  S-param table (text)
    Falls back to 2×3 if field components are not provided.
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        has_fields = (ezr_gt is not None and ezi_gt is not None
                      and ezr_gen is not None and ezi_gen is not None)
        nrows = 3 if has_fields else 2
        fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows))
        fig.suptitle(f"{title}  [{device_type}]", fontsize=11)

        def im(ax, arr, t, *, cmap="viridis", vmin=None, vmax=None):
            h = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(t, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return h

        colorbar_pairs = []  # (handle, ax) pairs for colorbars

        # Row 1: eps panels
        h = im(axes[0, 0], eps_gt, "GT eps", cmap="viridis")
        colorbar_pairs.append((h, axes[0, 0]))
        if port_masks_overlay is not None:
            # overlay port masks as semi-transparent colored regions
            alpha_mask = np.clip(port_masks_overlay, 0, 1) * 0.4
            overlay = np.zeros((*eps_gt.shape, 4))
            overlay[..., 0] = 1.0  # red channel
            overlay[..., 1] = 0.3
            overlay[..., 3] = alpha_mask
            axes[0, 0].imshow(overlay, origin="lower")
        h = im(axes[0, 1], eps_gen, "Gen eps (continuous)", cmap="viridis")
        colorbar_pairs.append((h, axes[0, 1]))
        h = im(axes[0, 2], eps_gen_binary, "Gen eps (binary)", cmap="viridis")
        colorbar_pairs.append((h, axes[0, 2]))

        # Row 2: field magnitude + eps error
        h = im(axes[1, 0], mag_gt, "GT |Ez|", cmap="magma")
        colorbar_pairs.append((h, axes[1, 0]))
        h = im(axes[1, 1], mag_gen, "Gen |Ez|", cmap="magma")
        colorbar_pairs.append((h, axes[1, 1]))

        eps_err = np.abs(eps_gt - eps_gen)
        h = im(axes[1, 2], eps_err, "|eps err|", cmap="hot")
        colorbar_pairs.append((h, axes[1, 2]))

        # Row 3 (if field components available): real/imag fields + S-param text
        if has_fields:
            vlim_r = max(np.abs(ezr_gt).max(), np.abs(ezr_gen).max(), 1e-12)
            vlim_i = max(np.abs(ezi_gt).max(), np.abs(ezi_gen).max(), 1e-12)

            # Real: GT (top half) | Gen (bottom half) side-by-side via vertical concat
            ezr_compare = np.concatenate([ezr_gt, ezr_gen], axis=0)
            h = im(axes[2, 0], ezr_compare, "Ez_real  GT(top) | Gen(bot)",
                   cmap="RdBu", vmin=-vlim_r, vmax=vlim_r)
            colorbar_pairs.append((h, axes[2, 0]))
            # Draw separator line at midpoint
            axes[2, 0].axhline(y=ezr_gt.shape[0] - 0.5, color="k", linewidth=0.8, linestyle="--")

            # Imag: GT (top half) | Gen (bottom half)
            ezi_compare = np.concatenate([ezi_gt, ezi_gen], axis=0)
            h = im(axes[2, 1], ezi_compare, "Ez_imag  GT(top) | Gen(bot)",
                   cmap="RdBu", vmin=-vlim_i, vmax=vlim_i)
            colorbar_pairs.append((h, axes[2, 1]))
            axes[2, 1].axhline(y=ezi_gt.shape[0] - 0.5, color="k", linewidth=0.8, linestyle="--")

            # S-param text panel in row 3
            axes[2, 2].axis("off")
            axes[2, 2].set_title("S-params", fontsize=10)
            axes[2, 2].text(
                0.05, 0.95, sparam_text,
                transform=axes[2, 2].transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )
        else:
            # No field components: S-param text in row 2 slot (original layout)
            axes[1, 2].axis("off")
            axes[1, 2].set_title("S-params", fontsize=10)
            axes[1, 2].text(
                0.05, 0.95, sparam_text,
                transform=axes[1, 2].transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

        for h, ax in colorbar_pairs:
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = None) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-15:
        return 100.0
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    if data_range < 1e-15:
        return 0.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _collect_layer_grad_norms(model) -> dict:
    """Collect per-block gradient L2 norms from model parameters."""
    group_ss = defaultdict(float)  # sum of squares per group
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = ".".join(name.split(".")[:2])
        group_ss[key] += float(p.grad.data.norm().item()) ** 2
    return {k: math.sqrt(v) for k, v in group_ss.items()}


def _save_enhanced_eval_sample_png(
    *,
    out_path: str,
    title: str,
    eps_phys: np.ndarray,
    ezr_gt: np.ndarray,
    ezi_gt: np.ndarray,
    ezr_pred: np.ndarray,
    ezi_pred: np.ndarray,
    metrics: dict,
) -> None:
    """
    Save a 3x4 evaluation panel with error maps and metrics:
      Row 0: eps | Ez_real GT | Ez_imag GT | |Ez| GT
      Row 1: Ez_real pred | Ez_imag pred | |Ez| pred | |err| magnitude
      Row 2: Real error | Imag error | Phase error (amp-weighted) | Metrics text
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mag_gt = np.sqrt(ezr_gt**2 + ezi_gt**2 + 1e-12)
        mag_pred = np.sqrt(ezr_pred**2 + ezi_pred**2 + 1e-12)
        mag_err = np.sqrt((ezr_pred - ezr_gt)**2 + (ezi_pred - ezi_gt)**2 + 1e-12)
        real_err = ezr_pred - ezr_gt
        imag_err = ezi_pred - ezi_gt

        # Amplitude-weighted phase error map
        phase_gt = np.arctan2(ezi_gt, ezr_gt)
        phase_pred = np.arctan2(ezi_pred, ezr_pred)
        phase_diff = np.arctan2(np.sin(phase_pred - phase_gt), np.cos(phase_pred - phase_gt))
        amp_weight = mag_gt / (mag_gt.max() + 1e-12)
        phase_err_map = np.abs(phase_diff) * amp_weight

        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        def im(ax, arr, t, *, cmap="magma", vmin=None, vmax=None):
            h = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(t, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            return h

        hs = []
        # Row 0: Ground truth
        hs.append(im(axes[0, 0], eps_phys, r"$\varepsilon_r$", cmap="viridis"))
        vlim_r = max(np.abs(ezr_gt).max(), np.abs(ezr_pred).max(), 1e-12)
        vlim_i = max(np.abs(ezi_gt).max(), np.abs(ezi_pred).max(), 1e-12)
        hs.append(im(axes[0, 1], ezr_gt, r"$E_z^{\rm re}$ (GT)", cmap="RdBu", vmin=-vlim_r, vmax=vlim_r))
        hs.append(im(axes[0, 2], ezi_gt, r"$E_z^{\rm im}$ (GT)", cmap="RdBu", vmin=-vlim_i, vmax=vlim_i))
        hs.append(im(axes[0, 3], mag_gt, r"$|E_z|$ (GT)", cmap="magma"))

        # Row 1: Prediction + magnitude error
        hs.append(im(axes[1, 0], ezr_pred, r"$E_z^{\rm re}$ (pred)", cmap="RdBu", vmin=-vlim_r, vmax=vlim_r))
        hs.append(im(axes[1, 1], ezi_pred, r"$E_z^{\rm im}$ (pred)", cmap="RdBu", vmin=-vlim_i, vmax=vlim_i))
        hs.append(im(axes[1, 2], mag_pred, r"$|E_z|$ (pred)", cmap="magma"))
        hs.append(im(axes[1, 3], mag_err, r"$|\Delta E_z|$", cmap="hot"))

        # Row 2: Error maps + metrics text
        err_vlim_r = max(np.abs(real_err).max(), 1e-12)
        err_vlim_i = max(np.abs(imag_err).max(), 1e-12)
        hs.append(im(axes[2, 0], real_err, r"$\Delta E_z^{\rm re}$", cmap="RdBu", vmin=-err_vlim_r, vmax=err_vlim_r))
        hs.append(im(axes[2, 1], imag_err, r"$\Delta E_z^{\rm im}$", cmap="RdBu", vmin=-err_vlim_i, vmax=err_vlim_i))
        hs.append(im(axes[2, 2], phase_err_map, r"$|\Delta\phi|$ (amp-weighted)", cmap="inferno"))

        # Metrics text box
        axes[2, 3].axis("off")
        axes[2, 3].set_title("Metrics", fontsize=9)
        metric_lines = []
        for k in ["psnr", "amp_err", "phase_err", "residual", "gt_residual", "gap_dB"]:
            v = metrics.get(k, None)
            if v is not None:
                if k == "psnr":
                    metric_lines.append(f"PSNR:         {v:.2f} dB")
                elif k == "amp_err":
                    metric_lines.append(f"Amp Error:    {v:.4e}")
                elif k == "phase_err":
                    metric_lines.append(f"Phase Error:  {v:.4e}")
                elif k == "residual":
                    metric_lines.append(f"Residual:     {v:.4e}")
                elif k == "gt_residual":
                    metric_lines.append(f"GT Residual:  {v:.4e}")
                elif k == "gap_dB":
                    metric_lines.append(f"Gap:          {v:+.1f} dB")
        axes[2, 3].text(
            0.05, 0.95, "\n".join(metric_lines),
            transform=axes[2, 3].transAxes, fontsize=10, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

        for ax, h in zip(axes[:, :].ravel()[:len(hs)], hs):
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _save_device_bar_chart_png(
    *,
    out_path: str,
    title: str,
    metrics_per_device: dict,
    epoch: int,
) -> None:
    """Save a 2x2 bar chart comparing metrics across device types."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not metrics_per_device:
            return
        devices = sorted(metrics_per_device.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        colors = plt.cm.Set2(np.linspace(0, 1, max(len(devices), 1)))
        x = np.arange(len(devices))

        chart_specs = [
            ("gap_dB", "Residual Gap (dB)", axes[0, 0]),
            ("amp_err", "Amplitude Error", axes[0, 1]),
            ("phase_err", "Phase Error (weighted)", axes[1, 0]),
            ("psnr", "PSNR (dB)", axes[1, 1]),
        ]
        for key, ylabel, ax in chart_specs:
            vals = [metrics_per_device[d].get(key, 0.0) for d in devices]
            bars = ax.bar(x, vals, color=colors[:len(devices)], edgecolor="black", linewidth=0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(devices, rotation=30, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=7)
            if key == "gap_dB":
                ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _save_device_ratio_p95_chart_png(
    *,
    out_path: str,
    title: str,
    metrics_per_device: dict,
    target_ratio: float,
) -> None:
    """Bar chart of per-device ratio p95 vs acceptance target (e.g. 1.5×)."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not metrics_per_device:
            return
        devices = sorted(metrics_per_device.keys())
        fig, ax = plt.subplots(figsize=(max(10, len(devices) * 1.2), 5))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        x = np.arange(len(devices))
        p95_vals = [float(metrics_per_device[d].get("ratio_p95", 0.0)) for d in devices]
        colors = plt.cm.Spectral(np.linspace(0.15, 0.85, max(len(devices), 1)))
        bars = ax.bar(x, p95_vals, color=colors[: len(devices)], edgecolor="black", linewidth=0.5)
        ax.axhline(float(target_ratio), color="red", linestyle="--", linewidth=1.2, label=f"target={target_ratio:g}×")
        ax.set_ylabel("ratio p95 (pred/GT residual)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(devices, rotation=30, ha="right", fontsize=9)
        for bar, v in zip(bars, p95_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _save_gradient_health_png(
    *,
    out_path: str,
    title: str,
    layer_grad_norms: dict,
    epoch: int,
) -> None:
    """Save a bar chart of per-layer-group gradient L2 norms (log scale)."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not layer_grad_norms:
            return
        names = list(layer_grad_norms.keys())
        norms = [layer_grad_norms[n] for n in names]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 5))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        x = np.arange(len(names))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.bar(x, norms, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("Gradient L2 Norm", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _save_training_curves_png(
    *,
    out_path: str,
    val_csv_path: str,
    epoch: int,
) -> None:
    """Read validation.csv and plot training curves over epochs."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.isfile(val_csv_path):
            return
        with open(val_csv_path, "r", encoding="UTF8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if not rows:
            return

        def col(name):
            if name in header:
                idx = header.index(name)
                vals = []
                for r in rows:
                    try:
                        v = float(r[idx]) if r[idx] != "" else float("nan")
                    except (IndexError, ValueError):
                        v = float("nan")
                    vals.append(v)
                return np.array(vals)
            return None

        epochs = col("epoch")
        if epochs is None:
            return

        fig, axes = plt.subplots(3, 2, figsize=(12, 11))
        fig.suptitle(f"Training Curves (epoch {epoch})", fontsize=12, fontweight="bold")

        # (0,0) FM loss
        fm = col("val_fm_loss")
        if fm is not None:
            mask = np.isfinite(fm) & (fm > 0)
            if mask.any():
                axes[0, 0].semilogy(epochs[mask], fm[mask], "b-o", markersize=2, label="FM loss")
                axes[0, 0].set_ylabel("FM Loss")
                axes[0, 0].legend(fontsize=8)
                axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_title("Flow Matching Loss", fontsize=9)

        # (0,1) Residual loss
        res = col("val_residual_loss")
        if res is not None:
            mask = np.isfinite(res) & (res > 0)
            if mask.any():
                axes[0, 1].semilogy(epochs[mask], res[mask], "r-o", markersize=2, label="Residual")
                axes[0, 1].legend(fontsize=8)
                axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_title("Residual Loss", fontsize=9)
        axes[0, 1].set_ylabel("Residual Loss")

        # (1,0) Residual gap dB
        gap = col("sample_residual_gap_dB")
        if gap is not None:
            mask = np.isfinite(gap)
            if mask.any():
                axes[1, 0].plot(epochs[mask], gap[mask], "g-o", markersize=2, label="Gap (dB)")
                axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
                axes[1, 0].legend(fontsize=8)
                axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_title("Residual Gap vs GT", fontsize=9)
        axes[1, 0].set_ylabel("Gap (dB)")
        axes[1, 0].set_xlabel("Epoch")

        # (1,1) Amp error + phase error
        amp = col("sample_amp_err_mean")
        ph = col("sample_phase_err_mean")
        ax_right = axes[1, 1]
        has_amp = amp is not None and np.isfinite(amp).any()
        has_ph = ph is not None and np.isfinite(ph).any()
        if has_amp:
            mask_a = np.isfinite(amp)
            ax_right.plot(epochs[mask_a], amp[mask_a], "m-o", markersize=2, label="Amp Error")
        if has_ph:
            mask_p = np.isfinite(ph)
            ax2 = ax_right.twinx()
            ax2.plot(epochs[mask_p], ph[mask_p], "c-s", markersize=2, label="Phase Error")
            ax2.set_ylabel("Phase Error", fontsize=8, color="c")
            ax2.tick_params(axis="y", labelcolor="c")
        ax_right.set_title("Amp & Phase Error", fontsize=9)
        ax_right.set_ylabel("Amp Error", fontsize=8, color="m")
        ax_right.set_xlabel("Epoch")
        ax_right.grid(alpha=0.3)
        lines1, labels1 = ax_right.get_legend_handles_labels()
        if has_ph:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_right.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
        elif has_amp:
            ax_right.legend(fontsize=7)

        # (2,0) Worst per-device ratio p95 (acceptance metric; lower is better)
        wr = col("worst_ratio_p95")
        pt = col("physical_ratio_target")
        if wr is not None and np.isfinite(wr).any():
            mask_w = np.isfinite(wr) & (wr > 0)
            if mask_w.any():
                axes[2, 0].plot(epochs[mask_w], wr[mask_w], "k-o", markersize=2, label="worst ratio p95")
            if pt is not None and np.isfinite(pt).any():
                mask_p = np.isfinite(pt) & (pt > 0)
                if mask_p.any():
                    axes[2, 0].axhline(float(np.nanmedian(pt[mask_p])), color="red", linestyle="--", linewidth=1.0, label="target")
            axes[2, 0].set_title("Worst device ratio p95 (pred/GT)", fontsize=9)
            axes[2, 0].set_ylabel("ratio")
            axes[2, 0].set_xlabel("Epoch")
            axes[2, 0].legend(fontsize=7)
            axes[2, 0].grid(alpha=0.3)
        else:
            axes[2, 0].set_visible(False)

        axes[2, 1].set_visible(False)

        for ax in axes.ravel():
            if ax.get_visible():
                ax.tick_params(labelsize=8)

        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
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
    out["grid_dx_um"] = _to(out.get("grid_dx_um", None), dt=dtype)
    out["grid_dy_um"] = _to(out.get("grid_dy_um", None), dt=dtype)

    if out.get("in_port_idx", None) is not None:
        if torch.is_tensor(out["in_port_idx"]):
            out["in_port_idx"] = out["in_port_idx"].to(device, non_blocking=True)
        else:
            out["in_port_idx"] = torch.as_tensor(out["in_port_idx"], dtype=torch.long, device=device)

    return out


def _format_sparams_compact(S_true_1d, S_pred_1d, port_valid_1d=None,
                            in_port_idx=None, max_ports=4) -> str:
    if S_true_1d is None or S_pred_1d is None:
        return ""
    S_true_1d = S_true_1d.reshape(-1)[:max_ports]
    S_pred_1d = S_pred_1d.reshape(-1)[:max_ports]
    keep = None
    if port_valid_1d is not None:
        pv = port_valid_1d.reshape(-1)[:max_ports].detach().cpu()
        keep = (pv > 0.5).numpy()

    # Resolve input port index to skip (trivially 1.0 due to normalization)
    in_idx = None
    if in_port_idx is not None:
        try:
            t = torch.as_tensor(in_port_idx).flatten() if not torch.is_tensor(in_port_idx) else in_port_idx.detach().flatten()
            in_idx = int(t[0].item()) if t.numel() >= 1 else None
        except Exception:
            try:
                in_idx = int(in_port_idx)
            except Exception:
                pass

    mags_t = torch.abs(S_true_1d).detach().cpu().numpy()
    mags_p = torch.abs(S_pred_1d).detach().cpu().numpy()
    ph_t = torch.angle(S_true_1d).detach().cpu().numpy()
    ph_p = np.zeros_like(ph_t)
    try:
        ph_p = torch.angle(S_pred_1d).detach().cpu().numpy()
    except Exception:
        pass

    parts = []
    for i in range(min(len(mags_t), len(mags_p), max_ports)):
        if keep is not None and (i >= len(keep) or not bool(keep[i])):
            continue
        if in_idx is not None and i == in_idx:
            continue  # skip input port (always trivially 1.0)
        parts.append(f"p{i}: |S| {mags_t[i]:.3f}->{mags_p[i]:.3f}, ∠ {ph_t[i]:+.2f}->{ph_p[i]:+.2f}")
    return " ; ".join(parts)


def _ensure_model_pml_cells(model_like, pml_cells: int) -> None:
    """
    Ensure flow_matching.py can reliably read model.(module).helmholtz.pml_cells for PML masking.
    """
    try:
        m = _unwrap_model(model_like)
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
    assert torch.cuda.is_available(), "Need CUDA GPU(s) for training."

    # Try DDP init; fall back to single-GPU if env vars aren't set (plain python).
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        # Raise NCCL watchdog timeout well above the default 10 min so rank-0-only
        # sample/inverse eval work (fm_sample ODE integration, PNG rendering, etc.)
        # cannot trip the collective-timeout deadlock when other ranks are parked
        # at a barrier. Override via env NCCL_PG_TIMEOUT_MIN if needed.
        _pg_timeout_min = int(os.environ.get("NCCL_PG_TIMEOUT_MIN", "120"))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=_pg_timeout_min),
        )

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

    rank = dist.get_rank() if use_ddp else 0
    world = dist.get_world_size() if use_ddp else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_dir = "logs_physics_unet_pbfm"
    experiment_dir = os.path.join(results_dir, args.version)
    checkpoint_root = os.environ.get("RAYFIELD_CKPT_ROOT", results_dir)
    ckpt_experiment_dir = os.path.join(checkpoint_root, args.version)
    ckpt_dir = os.path.join(ckpt_experiment_dir, "checkpoints")
    samples_dir = os.path.join(experiment_dir, "samples")

    if is_rank0():
        os.makedirs(results_dir, exist_ok=True)
        if (not args.resume_from) and os.path.exists(experiment_dir):
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        if (not args.resume_from) and (ckpt_experiment_dir != experiment_dir) and os.path.exists(ckpt_experiment_dir):
            for root, dirs, files in os.walk(ckpt_experiment_dir, topdown=False):
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
        if use_ddp:
            logger.info(f"DDP: rank={rank}/{world}, local_rank={local_rank}, device={device}")
        else:
            logger.info(f"Single-GPU training on {device}")
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Checkpoint dir: {ckpt_dir}")
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
                    "sample_gt_residual_mean",
                    "sample_residual_ratio_vs_gt",
                    "sample_residual_gap_dB",
                    "sample_amp_err_mean",
                    "sample_phase_err_mean",
                    "sample_psnr_mean",
                    "worst_ratio_p95",
                    "worst_ratio_max",
                    "physical_ratio_target",
                    "all_devices_ratio_p95_le_target",
                    "all_devices_ratio_max_le_target",
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

    exclude_devices = None
    if getattr(args, "exclude_devices", ""):
        exclude_devices = [s.strip() for s in args.exclude_devices.split(",") if s.strip()]

    include_wavelengths = None
    if getattr(args, "include_wavelengths", ""):
        include_wavelengths = [float(s.strip()) for s in args.include_wavelengths.split(",") if s.strip()]

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
        _canvas_hw = tuple(args.canvas_hw) if getattr(args, "canvas_hw", None) else None
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
                exclude_devices=exclude_devices,
                include_wavelengths=include_wavelengths,
                subset_train_per_sweep=subset_train if subset_train else None,
                subset_val_per_sweep=subset_val if subset_train or subset_val else None,
                subset_seed=subset_seed,
                crop_pml=bool(getattr(args, "crop_pml", False)),
                pml_cells=int(getattr(args, "pml_cells", 0)),
                return_aux=True,
                use_index_split=bool(getattr(args, "use_index_split", False)),
                include_sparams_cond=bool(getattr(args, "include_sparams_cond", False)),
                canvas_hw=_canvas_hw,
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
            exclude_devices=exclude_devices,
            include_wavelengths=include_wavelengths,
            return_aux=True,
            subset_train_per_sweep=subset_train if subset_train else None,
            subset_val_per_sweep=subset_val if subset_train or subset_val else None,
            subset_seed=subset_seed,
            crop_pml=bool(getattr(args, "crop_pml", False)),
            pml_cells=int(getattr(args, "pml_cells", 0)),
            augment=bool(getattr(args, "augment", True)),
            use_index_split=bool(getattr(args, "use_index_split", False)),
            include_sparams_cond=bool(getattr(args, "include_sparams_cond", False)),
            canvas_hw=_canvas_hw,
        )

        val_ds = FDTDDataset(
            root_dir=args.data_root,
            split="val",
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
            exclude_devices=exclude_devices,
            include_wavelengths=include_wavelengths,
            return_aux=True,
            subset_train_per_sweep=subset_train if subset_train else None,
            subset_val_per_sweep=subset_val if subset_train or subset_val else None,
            subset_seed=subset_seed,
            crop_pml=bool(getattr(args, "crop_pml", False)),
            pml_cells=int(getattr(args, "pml_cells", 0)),
            augment=False,
            use_index_split=bool(getattr(args, "use_index_split", False)),
            include_sparams_cond=bool(getattr(args, "include_sparams_cond", False)),
            canvas_hw=_canvas_hw,
        )

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None

    assert args.batch_size % world == 0, f"--batch-size (global) must be divisible by world_size={world}"
    per_gpu_batch = args.batch_size // world

    train_prefetch = args.prefetch_factor if args.train_num_workers > 0 else 2
    val_prefetch = args.prefetch_factor if args.val_num_workers > 0 else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),
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
    if args.no_attention:
        attn_res = ()
    else:
        attn_res = tuple(int(x) for x in args.attn_resolutions.split(",") if x.strip())

    channel_mult = tuple(int(x) for x in args.channel_mult.split(",") if x.strip())

    if is_rank0():
        logger.info("Using ComplexPhysicsUNet" if use_complex else "Using PhysicsUNet")
        logger.info("Physics features ENABLED" if enable_physics else "Physics features DISABLED (vanilla UNet ablation)")
        try:
            sample0 = train_ds[0]
            x0 = sample0[0] if isinstance(sample0, (tuple, list)) else sample0
            h, w = int(x0.shape[-2]), int(x0.shape[-1])
            if len(attn_res) == 0:
                logger.info(f"Data resolution: {h}x{w} (attention disabled)")
            else:
                token_desc = ", ".join([f"ds={ds}=>{h//ds}x{w//ds}" for ds in attn_res])
                logger.info(f"Data resolution: {h}x{w} (attention: {token_desc} tokens)")
        except Exception as exc:  # pragma: no cover
            logger.info(f"Data resolution: <unknown> (probe failed: {exc})")

    joint_training = bool(getattr(args, "joint_training", False))
    out_channels = 3 if joint_training else 2

    model_kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=args.hidden_size,
        num_res_blocks=args.num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attn_res,
        dropout=args.model_dropout,
        dims=2,
        num_heads=args.num_heads,
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

    # torch.compile: enabled when S-param head is not in use (Inductor doesn't support complex backward through learned head)
    use_compile = int(getattr(args, "unroll_steps", 0)) == 0 and not bool(getattr(base_model, "enable_sparam_head", False))
    if use_compile:
        import torch._functorch.config as _ftc
        _ftc.donated_buffer = False
        base_model = torch.compile(base_model)
        if is_rank0():
            logger.info("torch.compile enabled on base_model")

    if use_ddp:
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
    else:
        model = base_model

    if is_rank0():
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Weighted-sum objective only (no AugLag / no MGDA / no PCGrad).

    # Optimizer (model params only)
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        fused=True,
    )

    amp_enabled = bool(getattr(args, "amp", True))
    amp_dtype_str = str(getattr(args, "amp_dtype", "float16"))
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    # BF16 has the same exponent range as FP32 — no overflow, so GradScaler is unnecessary
    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    if _AMP_NEW_API:
        scaler = _GradScaler("cuda", enabled=use_scaler)
        autocast_device_type = "cuda"
    else:
        scaler = _GradScaler(enabled=use_scaler)
        autocast_device_type = "cuda"
    if amp_enabled:
        logger.info(f"AMP enabled with dtype={amp_dtype}, GradScaler={'on' if use_scaler else 'off'}")

    def optimizer_step(manual_grads: bool = False):
        # When ConFIG injects gradients manually, we bypass scaler.scale().backward(),
        # so scaler.step() would fail (_scale is None). Use opt.step() directly.
        if scaler.is_enabled() and not manual_grads:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
            if scaler.is_enabled() and manual_grads:
                try:
                    scaler.update()
                except Exception:
                    pass  # update() may fail when _scale was never set

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

    update_ema(_unwrap_model(ema), _unwrap_model(model), decay=0.0)

    # -----------------------
    # Resume
    # -----------------------
    start_epoch = 1

    # Auto-resume: find latest checkpoint if --auto-resume and no explicit --resume-from
    if getattr(args, "auto_resume", False) and not args.resume_from:
        import glob as _glob
        ckpt_pattern = os.path.join(ckpt_dir, "*.pt")
        ckpt_files = sorted(_glob.glob(ckpt_pattern))
        if ckpt_files:
            args.resume_from = ckpt_files[-1]
            if is_rank0():
                logger.info(f"Auto-resume: found {args.resume_from}")

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        _unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
        _unwrap_model(ema).load_state_dict(checkpoint["ema"], strict=False)

        if getattr(args, 'reset_optimizer', False):
            if is_rank0():
                logger.info("--reset-optimizer: skipping optimizer state, using fresh optimizer")
        else:
            opt.load_state_dict(checkpoint["opt"])

        _unwrap_model(model).set_normalization_stats(stats, normalize_eps=args.normalize_eps)
        _unwrap_model(ema).set_normalization_stats(stats, normalize_eps=args.normalize_eps)

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

        # Optionally reset LR schedule for a fresh cosine cycle from args.lr
        if getattr(args, 'reset_lr_on_resume', False):
            for pg in opt.param_groups:
                pg['lr'] = args.lr
            remaining = args.epochs - last_epoch
            if warmup_epochs > 0:
                warmup_sf = float(getattr(args, "warmup_start_factor", 0.1))
                warmup_sf = min(max(warmup_sf, 1e-6), 1.0)
                warmup_r = LinearLR(opt, start_factor=warmup_sf, total_iters=warmup_epochs)
                cos_T_r = max(1, remaining - warmup_epochs)
                cosine_r = CosineAnnealingLR(opt, T_max=cos_T_r, eta_min=min_lr)
                scheduler = SequentialLR(opt, schedulers=[warmup_r, cosine_r], milestones=[warmup_epochs])
            else:
                scheduler = CosineAnnealingLR(opt, T_max=max(1, remaining), eta_min=min_lr)
            if is_rank0():
                logger.info(f"Reset LR: fresh cosine {args.lr:.2e} over {remaining} epochs (min_lr={min_lr:.2e})")

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
    running_binarize_loss = 0.0
    running_geom_loss = 0.0
    running_grad_norm = 0.0
    grad_norm_max_epoch = 0.0
    layer_grad_norms_accum = defaultdict(list)  # per-block gradient norms (sampled periodically)
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
        t_phys_min = float(getattr(args, "t_physics_min", 0.0))
        if t_phys_min > 0.0:
            logger.info(f"Time-gating: physics losses only for t >= {t_phys_min}")

    detect_anomaly = bool(getattr(args, "detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if is_rank0():
            logger.info("Autograd anomaly detection ENABLED (--detect-anomaly).")

    _prev_weights = None  # track weight changes for log deduplication
    best_worst_ratio_p95 = float("inf")  # lower is better; for --ckpt-best-metric worst_ratio_p95

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        N1 = int(args.phaseA_epochs)
        N2 = int(args.phaseB_epochs)

        if use_endpoint:
            endpoint_start = _phase_start_epoch(
                getattr(args, "endpoint_start_phase", "B"), N1=N1, N2=N2
            )
            endpoint_weight = float(args.lambda_endpoint) * ramp_linear(
                epoch, start_epoch=endpoint_start, warmup_epochs=int(args.endpoint_warmup_epochs)
            )
        else:
            endpoint_weight = 0.0

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

        # Use tensors (not plain floats) for gates so that torch.compile /
        # CUDA-Graphs treat them as dynamic values instead of baking each new
        # value into a graph guard and recompiling (which leaks graph pools).
        phys_gate = 0.0
        if float(args.lambda_residual) > 0:
            phys_gate = float(residual_weight / float(args.lambda_residual))
            phys_gate = max(0.0, min(1.0, phys_gate))
        phys_gate = torch.tensor(phys_gate)

        phase_gate = 0.0
        if float(args.lambda_phase) > 0:
            phase_gate = float(phase_weight / float(args.lambda_phase))
            phase_gate = max(0.0, min(1.0, phase_gate))
        phase_gate = torch.tensor(phase_gate)

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

            # --- Joint training: sample mask mode and handle 3-channel noise ---
            batch_mask_mode = MASK_MODE_FORWARD
            v_t_eps_target = None
            base_cond_dim = 1  # wavelength only

            if joint_training:
                batch_mask_mode = sample_mask_mode(
                    forward_ratio=float(getattr(args, "forward_ratio", 0.5)),
                    inverse_ratio=float(getattr(args, "inverse_ratio", 0.3)),
                )

                # CFG dropout: randomly zero S-param conditioning
                cfg_dropout = float(getattr(args, "cfg_dropout", 0.15))
                if torch.rand(1).item() < cfg_dropout:
                    # Zero only S-param Re/Im entries; preserve port_valid flags
                    n_sparam_reals = 2 * 4  # Re/Im for max 4 ports
                    cond[:, base_cond_dim:base_cond_dim + n_sparam_reals] = 0.0

            x0_fields = torch.randn_like(fields_1)
            t = sample_t(fields_1, mode=args.t_sample_mode,
                          loc=args.t_sample_loc, scale=args.t_sample_scale)  # [B,1,1,1]
            x_t_fields = psi_t(x0_fields, fields_1, t)
            v_t_fields = u_t(x0_fields, fields_1)

            if joint_training:
                # Noise eps channel
                x0_eps = torch.randn_like(eps)
                x_t_eps = psi_t(x0_eps, eps, t)
                v_t_eps_raw = u_t(x0_eps, eps)

                if batch_mask_mode == MASK_MODE_FORWARD:
                    # Forward mode: eps is clean (fixed), fields are noised
                    x_t_eps_input = eps
                    v_t_eps_target = None  # no eps loss
                elif batch_mask_mode == MASK_MODE_INVERSE:
                    # Inverse mode: fields are clean (fixed), eps is noised
                    x_t_fields = fields_1  # use clean fields
                    v_t_fields = torch.zeros_like(v_t_fields)  # zero field velocity target
                    x_t_eps_input = x_t_eps
                    v_t_eps_target = v_t_eps_raw
                else:  # MASK_MODE_JOINT
                    # Both noised
                    x_t_eps_input = x_t_eps
                    v_t_eps_target = v_t_eps_raw

                # Build input: [x_t_fields(2), x_t_eps(1), src(1)] = 4 channels
                if extra_maps is not None:
                    x_t_input = torch.cat([x_t_fields, x_t_eps_input, src, extra_maps], dim=1)
                else:
                    x_t_input = torch.cat([x_t_fields, x_t_eps_input, src], dim=1)
            else:
                if extra_maps is not None:
                    x_t_input = torch.cat([x_t_fields, eps, src, extra_maps], dim=1)
                else:
                    x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

            compute_phase_grad_step = (phase_grad_weight > 0.0) and (global_step % int(args.phase_grad_every) == 0)
            compute_sparam_step = (sparam_weight > 0.0) and (global_step % int(args.sparam_every) == 0)
            compute_residual_step = (residual_weight > 0.0)

            # Binarization weight with warmup
            binarize_weight = 0.0
            if joint_training and float(getattr(args, "lambda_binarize", 0.0)) > 0.0:
                binarize_weight = float(args.lambda_binarize) * ramp_linear(
                    epoch, start_epoch=0, warmup_epochs=int(getattr(args, "binarize_warmup_epochs", 100))
                )

            # Geometry loss weight with warmup
            geom_weight = 0.0
            if joint_training and float(getattr(args, "lambda_geom", 0.0)) > 0.0:
                geom_weight = float(args.lambda_geom) * ramp_linear(
                    epoch, start_epoch=0, warmup_epochs=int(getattr(args, "geom_warmup_epochs", 100))
                )

            fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad_loss, sparam_loss, binarize_loss_val, geom_loss_val = cfm_loss_residual(
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
                interface_thr=float(getattr(args, "interface_thr", 0.0)),
                unroll_steps=int(getattr(args, "unroll_steps", 0)),
                unroll_phase=bool(getattr(args, "unroll_phase", False)),
                phase_amp_tau=float(getattr(args, "phase_amp_tau", 0.2)),
                amp_enabled=amp_enabled,
                amp_device_type=autocast_device_type,
                amp_dtype=amp_dtype,
                sparam_mode=str(getattr(args, "sparam_mode", "project")),
                joint_training=joint_training,
                mask_mode=batch_mask_mode,
                v_t_eps=v_t_eps_target,
                eps_core=float(getattr(args, "eps_core", 12.25)),
                eps_clad=float(getattr(args, "eps_clad", 2.07)),
                lambda_binarize=binarize_weight,
                eps_1=eps,
                lambda_geom=geom_weight,
                t_physics_min=float(getattr(args, "t_physics_min", 0.0)),
                residual_t_power=float(getattr(args, "residual_t_power", 0.0)),
            )

            # NOTE: residual is now self-normalized inside cfm_loss_residual
            # (divided by driving-term scale <(k₀²εE_true)²>), so no dx^4
            # scaling is needed here.  The returned loss is O(1) and
            # grid-independent; lambda_residual directly controls its weight.

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
                if joint_training and binarize_weight > 0.0:
                    loss_items.append(("bin", binarize_loss_val, float(binarize_weight)))
                if joint_training and geom_weight > 0.0:
                    loss_items.append(("geom", geom_loss_val, float(geom_weight)))

                # Compute a gradient vector per loss term (unweighted, matching your reference implementation).
                # Note: weights still control participation (which losses are included).
                params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
                # When GradScaler is active, scale each loss so fp16 gradients
                # don't underflow.  We unscale the combined vector after ConFIG.
                _config_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                for i, (_name, L, w) in enumerate(loss_items):
                    if float(w) <= 0.0:
                        continue
                    # Skip losses that are placeholders (e.g. torch.zeros when time-gating
                    # excludes all samples). These have no grad_fn and would raise:
                    # "element 0 of tensors does not require grad and does not have a grad_fn"
                    if not L.requires_grad:
                        continue
                    retain = (i != len(loss_items) - 1)
                    L_scaled = L * _config_scale if _config_scale != 1.0 else L

                    g_list = torch.autograd.grad(
                        outputs=L_scaled,
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
                        try:
                            g_config = _ConFIG_update(grads)
                        except Exception:
                            g_config = None
                        # Sanitize ConFIG output — the library can produce
                        # NaN/Inf when gradient vectors are near-zero or
                        # perfectly opposing.
                        if g_config is not None and (not torch.isfinite(g_config).all()):
                            g_config = None

                if g_config is None:
                    g_step = grads[0] if len(grads) > 0 else None
                else:
                    g_step = g_config

                # Undo GradScaler scaling so the optimizer sees true magnitudes.
                if g_step is not None and _config_scale != 1.0:
                    g_step = g_step / _config_scale

                if g_step is None:
                    # ultra fallback (shouldn't happen): do plain FM backward
                    opt.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(fm_loss).backward()
                        scaler.unscale_(opt)
                    else:
                        fm_loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                    if math.isfinite(gn):
                        running_grad_norm += gn
                        grad_norm_max_epoch = max(grad_norm_max_epoch, gn)
                    optimizer_step(manual_grads=False)  # fallback uses normal backward
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
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                    if math.isfinite(gn):
                        running_grad_norm += gn
                        grad_norm_max_epoch = max(grad_norm_max_epoch, gn)
                    optimizer_step(manual_grads=True)  # ConFIG injects grads, bypass scaler.step

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
                    if joint_training and binarize_weight > 0.0:
                        total_loss = total_loss + binarize_weight * binarize_loss_val
                    if joint_training and geom_weight > 0.0:
                        total_loss = total_loss + geom_weight * geom_loss_val

                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(opt)
                    else:
                        total_loss.backward()

                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                    if math.isfinite(gn):
                        running_grad_norm += gn
                        grad_norm_max_epoch = max(grad_norm_max_epoch, gn)
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

            update_ema(_unwrap_model(ema), _unwrap_model(model))

            # Sample per-layer gradient norms periodically (every 50 steps)
            if train_steps_epoch % 50 == 0 and is_rank0():
                try:
                    step_gnorms = _collect_layer_grad_norms(_unwrap_model(model))
                    for k, v in step_gnorms.items():
                        layer_grad_norms_accum[k].append(v)
                except Exception:
                    pass

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
            if joint_training and binarize_weight > 0.0:
                running_binarize_loss += float(binarize_loss_val.item())
            if joint_training and geom_weight > 0.0:
                running_geom_loss += float(geom_loss_val.item())

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
                    t = sample_t(fields_1, mode=args.t_sample_mode,
                                  loc=args.t_sample_loc, scale=args.t_sample_scale)
                    x_t_fields = psi_t(x0_fields, fields_1, t)
                    v_t_fields = u_t(x0_fields, fields_1)

                    if extra_maps is not None:
                        x_t_input = torch.cat([x_t_fields, eps, src, extra_maps], dim=1)
                    else:
                        x_t_input = torch.cat([x_t_fields, eps, src], dim=1)

                    compute_phase_grad_val = (phase_grad_weight > 0.0)
                    compute_residual_val = (residual_weight > 0.0)
                    compute_sparam_val = (sparam_weight > 0.0)

                    fm_v, res_v, ph_v, end_v, phg_v, sp_v, _, _ = cfm_loss_residual(
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
                        interface_thr=float(getattr(args, "interface_thr", 0.0)),
                        unroll_steps=int(getattr(args, "unroll_steps", 0)),
                        unroll_phase=bool(getattr(args, "unroll_phase", False)),
                        phase_amp_tau=float(getattr(args, "phase_amp_tau", 0.2)),
                        amp_enabled=amp_enabled,
                        amp_device_type=autocast_device_type,
                        amp_dtype=amp_dtype,
                        sparam_mode=str(getattr(args, "sparam_mode", "project")),
                        t_physics_min=float(getattr(args, "t_physics_min", 0.0)),
                        residual_t_power=float(getattr(args, "residual_t_power", 0.0)),
                    )

                    # residual is self-normalized inside cfm_loss_residual
                    # (no dx^4 needed — see training loop comment)

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
                            x1_fields_pred = a * fields_t + b * u_t_pred[:, :2]  # normalized (field channels only)
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
                        ip0 = aux.get("in_port_idx", None)
                        sparams_example = (S_true[0].detach(), S_pred[0].detach(), pv0, ip0)

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

            # Sample eval (rank0 only) — per-device-type sampling
            sample_residual_mean = 0.0
            sample_gt_residual_mean = 0.0
            sample_amp_err_mean = 0.0
            sample_phase_err_mean = 0.0
            sample_psnr_mean = 0.0
            sample_metrics_per_device = {}  # {device_type: {metric: value}}
            worst_ratio_p95 = None
            worst_ratio_max = None
            all_devices_p95_le_target = None
            all_devices_max_le_target = None
            # Non-rank-0 ranks park here while rank 0 runs the (potentially long)
            # sample-eval block below. Without this barrier, other ranks would
            # race ahead into the next epoch's training AllReduces and trip the
            # NCCL watchdog timeout while rank 0 is still in fm_sample / PNG IO.
            ddp_barrier(device)
            if is_rank0():
                residuals = []
                gt_residuals = []
                amp_errs = []
                phase_errs = []
                psnr_list = []
                saved_any = False
                saved_paths = []

                # Build per-device-type index of val samples (lazy, once)
                if not hasattr(val_ds, "_device_type_index"):
                    dt_index = defaultdict(list)  # device_type -> [sample_idx, ...]
                    for idx in range(len(val_ds)):
                        try:
                            _, _, aux_idx = val_ds[idx]
                            dt = aux_idx.get("device_type", "") if isinstance(aux_idx, dict) else ""
                            if not dt:
                                dt = "unknown"
                            dt_index[dt].append(idx)
                        except Exception:
                            pass
                    val_ds._device_type_index = dict(dt_index)
                    if dt_index:
                        logger.info(f"[sample eval] Device-type index: {', '.join(f'{k}({len(v)})' for k, v in sorted(dt_index.items()))}")

                dt_index = val_ds._device_type_index
                n_auto = max(1, int(args.sample_eval_limit) // max(len(dt_index), 1))
                n_per_device = int(getattr(args, "eval_samples_per_device", 0) or 0)
                if n_per_device <= 0:
                    n_per_device = n_auto

                sample_indices = _select_eval_sample_indices(
                    dt_index, n_per_device, args, epoch, logger,
                )
                if not getattr(val_ds, "_eval_protocol_logged", False):
                    fns = int(getattr(args, "eval_flow_noise_seed", -1))
                    logger.info(
                        f"[sample eval] protocol: mode={getattr(args, 'eval_sample_mode', 'epoch_random')}, "
                        f"n_per_device={n_per_device}, total_samples={len(sample_indices)}, "
                        f"eval_flow_noise_seed={fns} (>=0: deterministic per index)"
                    )
                    val_ds._eval_protocol_logged = True

                # Per-device accumulators
                dev_residuals = defaultdict(list)
                dev_gt_residuals = defaultdict(list)
                dev_ratios = defaultdict(list)  # per-sample pred/GT residual ratio
                dev_amp_errs = defaultdict(list)
                dev_phase_errs = defaultdict(list)
                dev_psnr = defaultdict(list)
                dev_sparam_mag_errs = defaultdict(list)
                dev_sparam_phase_errs = defaultdict(list)
                ode_sparams_example = None  # S-param example from clean ODE-sampled fields

                with torch.no_grad():
                    for s_idx, (s, dt_name) in enumerate(sample_indices):
                        x_full_s, cond_s, aux_s = val_ds[s]
                        x_full_s = x_full_s.unsqueeze(0).to(device, dtype=dtype)
                        cond_s = cond_s.unsqueeze(0).to(device, dtype=dtype)

                        eps_s = x_full_s[:, 2:3]
                        src_s = x_full_s[:, 3:4]
                        extra_maps_s = x_full_s[:, 4:] if x_full_s.shape[1] > 4 else None
                        flow_noise_seed = int(getattr(args, "eval_flow_noise_seed", -1))
                        if flow_noise_seed >= 0:
                            noise_seed = int(flow_noise_seed) + int(s) * 1_000_003
                            x0_fields_s = _eval_field_noise(
                                x_full_s[:, 0:2].shape, device, dtype, noise_seed,
                            )
                        else:
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
                            time_grid=getattr(args, "time_grid", "quadratic"),
                        )
                        x1_pred = torch.cat([x1_fields_pred, eps_s], dim=1)
                        x1_pred[:, 0] = x1_pred[:, 0] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        x1_pred[:, 1] = x1_pred[:, 1] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        if args.normalize_eps:
                            x1_pred[:, 2] = x1_pred[:, 2] * float(stats["eps_std"]) + float(stats["eps_mean"])

                        B = x1_pred.shape[0]
                        k0 = (2.0 * torch.pi) / lambda_um_s.view(B)

                        # Source-free mask: Helmholtz only valid outside source region
                        src_dilated = F.max_pool2d(src_s, kernel_size=5, stride=1, padding=2)
                        src_free_m = (src_dilated < 0.5).to(dtype=torch.float32)

                        # Interface mask: exclude pixels where 5-point Laplacian stencil
                        # spans a sharp eps discontinuity (same as training in flow_matching.py)
                        eps_phys_for_iface = x1_pred[:, 2:3]
                        interface_thr_val = float(getattr(args, "interface_thr", 0.0))
                        if interface_thr_val > 0.0 and helmholtz_op is not None:
                            grad_eps_x = helmholtz_op.diff.diff_x(eps_phys_for_iface)
                            grad_eps_y = helmholtz_op.diff.diff_y(eps_phys_for_iface)
                            grad_eps_mag = (grad_eps_x ** 2 + grad_eps_y ** 2).sqrt()
                            interface_free = (grad_eps_mag < interface_thr_val).to(dtype=torch.float32)
                            residual_mask = src_free_m * interface_free
                        else:
                            residual_mask = src_free_m

                        # Predicted field residual (source + interface masked)
                        R = helmholtz_op(x1_pred[:, 0:2], x1_pred[:, 2:3], k0=k0)
                        R2 = (R[:, 0:1] ** 2 + R[:, 1:2] ** 2) * (dx ** 4)
                        res_masked = (R2 * residual_mask).sum() / residual_mask.sum().clamp_min(1.0)
                        res_val = float(res_masked.sqrt().item())
                        residuals.append(res_val)
                        dev_residuals[dt_name].append(res_val)

                        # Ground truth residual (same masking) - baseline reference
                        ezr_gt_s = x_full_s[:, 0:1] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        ezi_gt_s = x_full_s[:, 1:2] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        gt_fields = torch.cat([ezr_gt_s, ezi_gt_s], dim=1)
                        R_gt = helmholtz_op(gt_fields, x1_pred[:, 2:3], k0=k0)
                        R2_gt = (R_gt[:, 0:1] ** 2 + R_gt[:, 1:2] ** 2) * (dx ** 4)
                        res_gt_masked = (R2_gt * residual_mask).sum() / residual_mask.sum().clamp_min(1.0)
                        gt_res_val = float(res_gt_masked.sqrt().item())
                        gt_residuals.append(gt_res_val)
                        dev_gt_residuals[dt_name].append(gt_res_val)
                        ratio_ij = res_val / max(gt_res_val, 1e-12)
                        dev_ratios[dt_name].append(ratio_ij)

                        # ezr_gt_s, ezi_gt_s already computed above for GT residual
                        mag_gt_s = torch.sqrt(ezr_gt_s ** 2 + ezi_gt_s ** 2 + 1e-12)

                        # --- Global phase alignment before eval metrics ---
                        # The model may predict fields with a global phase offset.
                        # Align pred to GT via amplitude-weighted phase projection
                        # (same method used in training endpoint loss).
                        E_gt_eval = torch.complex(ezr_gt_s, ezi_gt_s)       # [B,1,H,W]
                        E_pred_eval = torch.complex(x1_pred[:, 0:1], x1_pred[:, 1:2])  # [B,1,H,W]
                        w_eval = (mag_gt_s ** 2).clamp_min(1e-12)            # amplitude^2 weight
                        dot_eval = (w_eval * torch.conj(E_pred_eval) * E_gt_eval).sum(dim=(2, 3))  # [B,1]
                        mag_dot = torch.abs(dot_eval).clamp_min(1e-8)
                        rot_eval = dot_eval / mag_dot                        # unit phasor [B,1]
                        E_pred_aligned = E_pred_eval * rot_eval.unsqueeze(-1).unsqueeze(-1)  # [B,1,H,W]
                        x1_pred_aligned = torch.stack([E_pred_aligned.real, E_pred_aligned.imag], dim=1).squeeze(2)  # [B,2,H,W]

                        mag_pred_s = torch.sqrt(x1_pred_aligned[:, 0:1] ** 2 + x1_pred_aligned[:, 1:2] ** 2 + 1e-12)

                        eps_phys_s = x1_pred[:, 2:3]
                        m = (eps_phys_s > float(args.eps_thr)).to(dtype=torch.float32)
                        k = int(getattr(args, "dilate", 1))
                        if k > 1:
                            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2).clamp(0.0, 1.0)
                        denom = float(m.sum().item())
                        if denom < 1.0:
                            m = torch.ones_like(m)

                        amp_err_map_s = torch.abs(mag_pred_s - mag_gt_s)
                        ae_val = float((amp_err_map_s * m).sum().item() / (m.sum().item() + 1e-12))
                        amp_errs.append(ae_val)
                        dev_amp_errs[dt_name].append(ae_val)

                        phase_gt_s = torch.atan2(ezi_gt_s, ezr_gt_s)
                        phase_pred_s = torch.atan2(x1_pred_aligned[:, 1:2], x1_pred_aligned[:, 0:1])
                        phase_err_s = torch.atan2(torch.sin(phase_pred_s - phase_gt_s), torch.cos(phase_pred_s - phase_gt_s))
                        # Amplitude-weighted phase error (consistent with training loss tau-gating)
                        amp_weight = mag_gt_s.clamp_min(1e-8)
                        pe_val = float((torch.abs(phase_err_s) * amp_weight * m).sum().item()
                                       / ((amp_weight * m).sum().item() + 1e-12))
                        phase_errs.append(pe_val)
                        dev_phase_errs[dt_name].append(pe_val)

                        # PSNR on magnitude field (within device mask)
                        mag_gt_np = (mag_gt_s * m)[0, 0].detach().float().cpu().numpy()
                        mag_pred_np = (mag_pred_s * m)[0, 0].detach().float().cpu().numpy()
                        psnr_val = _compute_psnr(mag_pred_np, mag_gt_np)
                        psnr_list.append(psnr_val)
                        dev_psnr[dt_name].append(psnr_val)

                        # --- S-param extraction from clean ODE-sampled fields ---
                        if (sparam_weight > 0.0) and isinstance(aux_s, dict):
                            pm_s = aux_s.get("port_masks", None)
                            st_s = aux_s.get("sparams_true", None)
                            pv_s = aux_s.get("port_valid", None)
                            ip_s = aux_s.get("in_port_idx", None)
                            pid_s = aux_s.get("port_ids", None)
                            if pm_s is not None and st_s is not None:
                                # Move to device (aux_s comes raw from dataset)
                                pm_dev = pm_s.unsqueeze(0).to(device, dtype=dtype) if pm_s.dim() == 3 else pm_s.to(device, dtype=dtype)
                                st_dev = st_s.unsqueeze(0).to(device) if st_s.dim() == 1 else st_s.to(device)
                                ip_dev = ip_s.to(device) if torch.is_tensor(ip_s) else ip_s
                                pid_dev = pid_s.to(device) if torch.is_tensor(pid_s) else pid_s
                                pv_1d = pv_s.detach() if torch.is_tensor(pv_s) else None

                                # Build complex E from un-aligned predicted fields (alignment corrupts phase for S-params)
                                E_ode = torch.complex(x1_pred[:, 0:1].squeeze(1), x1_pred[:, 1:2].squeeze(1))  # [B,H,W]
                                S_pred_ode = extract_sparams(E_ode, pm_dev, in_port_idx=ip_dev, port_ids=pid_dev)

                                # Per-port magnitude and phase errors (exclude input port, invalid ports)
                                in_idx_val = _resolve_in_idx(ip_dev, port_ids=pid_dev)
                                in_idx_int = int(in_idx_val) if not torch.is_tensor(in_idx_val) else int(in_idx_val.flatten()[0].item())
                                S_true_1d = st_dev[0] if st_dev.dim() == 2 else st_dev
                                S_pred_1d = S_pred_ode[0]
                                pv_np = pv_1d.cpu().numpy() if pv_1d is not None else np.ones(S_true_1d.shape[0])
                                for pi in range(min(len(S_true_1d), len(S_pred_1d))):
                                    if pv_np[pi] < 0.5 or pi == in_idx_int:
                                        continue
                                    mag_err = abs(abs(S_pred_1d[pi].item()) - abs(S_true_1d[pi].item()))
                                    # Phase error (wrap-safe)
                                    dp = S_pred_1d[pi] * S_true_1d[pi].conj()
                                    ph_err = abs(float(torch.angle(dp).item()))
                                    dev_sparam_mag_errs[dt_name].append(mag_err)
                                    dev_sparam_phase_errs[dt_name].append(ph_err)

                                # Capture first example for logging
                                if ode_sparams_example is None:
                                    ode_sparams_example = (S_true_1d.detach().cpu(), S_pred_1d.detach().cpu(), pv_1d, ip_dev)

                        # Save one qualitative sample panel per device type
                        save_eval_samples = bool(getattr(args, "save_eval_samples", True))
                        if save_eval_samples and s_idx < len(sample_indices):
                            # Only save first sample of each device type
                            n_already_saved_this_dt = sum(1 for prev_i, (_, prev_dt) in enumerate(sample_indices[:s_idx]) if prev_dt == dt_name)
                            if n_already_saved_this_dt == 0:
                                # eps (phys) for display: prefer GT eps if available
                                eps_gt_phys = x_full_s[:, 2:3]
                                if args.normalize_eps:
                                    eps_gt_phys = eps_gt_phys * float(stats["eps_std"]) + float(stats["eps_mean"])
                                eps_img = eps_gt_phys[0, 0].detach().float().cpu().numpy()

                                ezr_gt_img = ezr_gt_s[0, 0].detach().float().cpu().numpy()
                                ezi_gt_img = ezi_gt_s[0, 0].detach().float().cpu().numpy()
                                ezr_pred_img = x1_pred_aligned[:, 0:1][0, 0].detach().float().cpu().numpy()
                                ezi_pred_img = x1_pred_aligned[:, 1:2][0, 0].detach().float().cpu().numpy()

                                out_path = os.path.join(samples_dir, f"sample_epoch_{epoch:04d}_{dt_name}.png")
                                title = f"epoch={epoch:04d} [{dt_name}]"
                                sample_metrics = {
                                    "psnr": psnr_val,
                                    "amp_err": ae_val,
                                    "phase_err": pe_val,
                                    "residual": res_val,
                                    "gt_residual": gt_res_val,
                                    "gap_dB": 10.0 * math.log10(max(res_val / max(gt_res_val, 1e-12), 1e-12)),
                                }
                                _save_enhanced_eval_sample_png(
                                    out_path=out_path,
                                    title=title,
                                    eps_phys=eps_img,
                                    ezr_gt=ezr_gt_img,
                                    ezi_gt=ezi_gt_img,
                                    ezr_pred=ezr_pred_img,
                                    ezi_pred=ezi_pred_img,
                                    metrics=sample_metrics,
                                )
                                if os.path.isfile(out_path):
                                    saved_any = True
                                    saved_paths.append(out_path)

                sample_residual_mean = float(np.mean(residuals)) if residuals else 0.0
                sample_gt_residual_mean = float(np.mean(gt_residuals)) if gt_residuals else 0.0
                sample_amp_err_mean = float(np.mean(amp_errs)) if amp_errs else 0.0
                sample_phase_err_mean = float(np.mean(phase_errs)) if phase_errs else 0.0
                sample_psnr_mean = float(np.mean(psnr_list)) if psnr_list else 0.0

                # Aggregate per-device metrics (per-sample ratio stats for stable acceptance tests)
                ph_targ = float(getattr(args, "physical_ratio_target", 1.5))
                for dt_name in sorted(dev_residuals.keys()):
                    dr = dev_residuals[dt_name]
                    dgr = dev_gt_residuals[dt_name]
                    dae = dev_amp_errs[dt_name]
                    dpe = dev_phase_errs[dt_name]
                    dpsnr = dev_psnr.get(dt_name, [])
                    dt_res = float(np.mean(dr))
                    dt_gt_res = float(np.mean(dgr))
                    ratio_over_mean = dt_res / max(dt_gt_res, 1e-12)
                    rs_list = dev_ratios.get(dt_name, [])
                    st = _ratio_list_stats(rs_list)
                    ratio_primary = st["mean"] if rs_list else ratio_over_mean
                    dt_gap_dB = 10.0 * math.log10(max(ratio_primary, 1e-12))
                    dt_metrics = {
                        "residual": dt_res,
                        "gt_residual": dt_gt_res,
                        "ratio": ratio_primary,
                        "ratio_mean_over_mean": ratio_over_mean,
                        "ratio_std": st["std"],
                        "ratio_p95": st["p95"],
                        "ratio_max": st["max"],
                        "ratio_min": st["min"],
                        "gap_dB": dt_gap_dB,
                        "amp_err": float(np.mean(dae)),
                        "phase_err": float(np.mean(dpe)),
                        "psnr": float(np.mean(dpsnr)) if dpsnr else 0.0,
                        "n_samples": len(dr),
                    }
                    # Add ODE-sampled S-param metrics if available
                    dsme = dev_sparam_mag_errs.get(dt_name, [])
                    dspe = dev_sparam_phase_errs.get(dt_name, [])
                    if dsme:
                        dt_metrics["sparam_mag_err"] = float(np.mean(dsme))
                        dt_metrics["sparam_phase_err_deg"] = float(np.degrees(np.mean(dspe)))
                    sample_metrics_per_device[dt_name] = dt_metrics

                if sample_metrics_per_device:
                    p95s = [sample_metrics_per_device[d]["ratio_p95"] for d in sample_metrics_per_device]
                    maxes = [sample_metrics_per_device[d]["ratio_max"] for d in sample_metrics_per_device]
                    worst_ratio_p95 = float(max(p95s))
                    worst_ratio_max = float(max(maxes))
                    all_devices_p95_le_target = bool(
                        all(sample_metrics_per_device[d]["ratio_p95"] <= ph_targ for d in sample_metrics_per_device)
                    )
                    all_devices_max_le_target = bool(
                        all(sample_metrics_per_device[d]["ratio_max"] <= ph_targ for d in sample_metrics_per_device)
                    )

                eval_jsonl_path = os.path.join(experiment_dir, "eval_sample_metrics.jsonl")
                try:
                    rec = {
                        "epoch": int(epoch),
                        "physical_ratio_target": ph_targ,
                        "worst_ratio_p95": worst_ratio_p95,
                        "worst_ratio_max": worst_ratio_max,
                        "all_devices_ratio_p95_le_target": all_devices_p95_le_target,
                        "all_devices_ratio_max_le_target": all_devices_max_le_target,
                        "per_device": {k: dict(v) for k, v in sample_metrics_per_device.items()},
                        "protocol": {
                            "eval_sample_mode": getattr(args, "eval_sample_mode", "epoch_random"),
                            "eval_sample_index_seed": int(getattr(args, "eval_sample_index_seed", 0)),
                            "eval_flow_noise_seed": int(getattr(args, "eval_flow_noise_seed", -1)),
                            "eval_sample_indices_json": (getattr(args, "eval_sample_indices_json", "") or "").strip(),
                            "n_per_device": int(n_per_device),
                            "n_total_samples": int(len(sample_indices)),
                        },
                    }
                    with open(eval_jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(rec, default=str) + "\n")
                except Exception:
                    pass

                _save_device_ratio_p95_chart_png(
                    out_path=os.path.join(samples_dir, f"ratio_p95_epoch_{epoch:04d}.png"),
                    title=f"Physical residual ratio (per-device) — Epoch {epoch}",
                    metrics_per_device=sample_metrics_per_device,
                    target_ratio=ph_targ,
                )

                # Save per-device bar chart dashboard and training curves
                _save_device_bar_chart_png(
                    out_path=os.path.join(samples_dir, f"dashboard_epoch_{epoch:04d}.png"),
                    title=f"Validation Dashboard - Epoch {epoch}",
                    metrics_per_device=sample_metrics_per_device,
                    epoch=epoch,
                )
                _save_training_curves_png(
                    out_path=os.path.join(samples_dir, f"training_curves_epoch_{epoch:04d}.png"),
                    val_csv_path=val_csv_path,
                    epoch=epoch,
                )

            if is_rank0():
                if sample_gt_residual_mean > 0.0:
                    sample_residual_ratio_vs_gt = float(sample_residual_mean / max(sample_gt_residual_mean, 1e-12))
                    sample_residual_gap_dB = 10.0 * math.log10(max(sample_residual_ratio_vs_gt, 1e-12))
                else:
                    sample_residual_ratio_vs_gt = float("inf")
                    sample_residual_gap_dB = float("inf")

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
                    f"\n             sample_residual={sample_residual_mean:.4e} (GT={sample_gt_residual_mean:.4e})"
                    f", ratio={sample_residual_ratio_vs_gt:.3f}x"
                    f", residual_gap={sample_residual_gap_dB:+.1f}dB"
                    f", amp_err={sample_amp_err_mean:.4e}"
                    f", phase_err_w={sample_phase_err_mean:.4e}"
                    f", psnr={sample_psnr_mean:.2f}dB"
                )
                if worst_ratio_p95 is not None and math.isfinite(float(worst_ratio_p95)):
                    _pt = float(getattr(args, "physical_ratio_target", 1.5))
                    msg += (
                        f"\n             worst_ratio_p95={worst_ratio_p95:.3f}x, worst_ratio_max={worst_ratio_max:.3f}x "
                        f"(target≤{_pt:.2f}×) | all_p95_ok={all_devices_p95_le_target}, all_max_ok={all_devices_max_le_target}"
                    )
                # Per-device breakdown
                if sample_metrics_per_device:
                    msg += "\n             --- per-device ---"
                    for dt_name, dt_m in sorted(sample_metrics_per_device.items()):
                        phase_err_deg = float(dt_m['phase_err']) * 180.0 / math.pi
                        line = (
                            f"\n             {dt_name:>20s} (n={dt_m['n_samples']:d}): "
                            f"r_p95={dt_m['ratio_p95']:.3f}, r_max={dt_m['ratio_max']:.3f}, "
                            f"res={dt_m['residual']:.4e}, gap={dt_m['gap_dB']:+.1f}dB, "
                            f"amp_err={dt_m['amp_err']:.4e}, "
                            f"phase_err={dt_m['phase_err']:.4e} ({phase_err_deg:.1f}°), "
                            f"psnr={dt_m['psnr']:.2f}dB"
                        )
                        if "sparam_mag_err" in dt_m:
                            line += f", |S|_err={dt_m['sparam_mag_err']:.4f}, S_ph_err={dt_m['sparam_phase_err_deg']:.1f}°"
                        msg += line
                logger.info(msg)
                if saved_any:
                    logger.info(f"[epoch {epoch:04d}] Saved eval sample(s) to {samples_dir}")

                # Prefer ODE-sampled S-param example (clean fields) over noisy single-step estimate
                _sparam_ex = ode_sparams_example if ode_sparams_example is not None else sparams_example
                if (sparam_weight > 0.0) and (_sparam_ex is not None):
                    S_true_0, S_pred_0, pv0, ip0 = _sparam_ex
                    src_label = "ODE-sampled" if ode_sparams_example is not None else "single-step"
                    logger.info(
                        f"[epoch {epoch:04d}] Sparams ({src_label}) example (true->pred): "
                        f"{_format_sparams_compact(S_true_0, S_pred_0, port_valid_1d=pv0, in_port_idx=ip0)}"
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
                        sample_gt_residual_mean,
                        sample_residual_ratio_vs_gt,
                        sample_residual_gap_dB,
                        sample_amp_err_mean,
                        sample_phase_err_mean,
                        sample_psnr_mean,
                        worst_ratio_p95 if worst_ratio_p95 is not None else "",
                        worst_ratio_max if worst_ratio_max is not None else "",
                        float(getattr(args, "physical_ratio_target", 1.5)),
                        int(bool(all_devices_p95_le_target)) if all_devices_p95_le_target is not None else "",
                        int(bool(all_devices_max_le_target)) if all_devices_max_le_target is not None else "",
                    ])

                if wandb_run is not None:
                    log_dict = {
                        "epoch": epoch,
                        "val/fm_loss": val_fm_loss,
                        "val/residual_loss": val_res_loss,
                        "val/sample_residual": sample_residual_mean,
                        "val/sample_gt_residual": sample_gt_residual_mean,
                        "val/sample_residual_ratio_vs_gt": sample_residual_ratio_vs_gt,
                        "val/sample_residual_gap_dB": sample_residual_gap_dB,
                        "val/sample_amp_err": sample_amp_err_mean,
                        "val/sample_phase_err_w": sample_phase_err_mean,
                        "val/sample_psnr_mean": sample_psnr_mean,
                    }
                    if worst_ratio_p95 is not None and math.isfinite(float(worst_ratio_p95)):
                        log_dict["val/worst_ratio_p95"] = float(worst_ratio_p95)
                        log_dict["val/worst_ratio_max"] = float(worst_ratio_max) if worst_ratio_max is not None else float("nan")
                        log_dict["val/physical_ratio_target"] = float(getattr(args, "physical_ratio_target", 1.5))
                        log_dict["val/all_devices_ratio_p95_le_target"] = int(bool(all_devices_p95_le_target))
                        log_dict["val/all_devices_ratio_max_le_target"] = int(bool(all_devices_max_le_target))
                    if phase_weight > 0.0:
                        log_dict["val/phase_loss"] = val_phase_loss
                        try:
                            log_dict["val/phase_rms_deg"] = math.sqrt(max(0.0, 2.0 * val_phase_loss)) * 180.0 / math.pi
                        except Exception:
                            pass
                    if use_endpoint:
                        log_dict["val/endpoint_loss"] = val_endpoint_loss
                    if phase_grad_weight > 0.0:
                        log_dict["val/phase_grad_loss"] = val_phase_grad_loss
                    if sparam_weight > 0.0:
                        log_dict["val/sparam_loss"] = val_sparam_loss
                    # Per-device metrics to wandb
                    for dt_name, dt_m in sample_metrics_per_device.items():
                        log_dict[f"val_device/{dt_name}/residual"] = dt_m["residual"]
                        log_dict[f"val_device/{dt_name}/residual_gap_dB"] = dt_m["gap_dB"]
                        log_dict[f"val_device/{dt_name}/ratio_p95"] = dt_m["ratio_p95"]
                        log_dict[f"val_device/{dt_name}/ratio_max"] = dt_m["ratio_max"]
                        log_dict[f"val_device/{dt_name}/ratio_std"] = dt_m["ratio_std"]
                        log_dict[f"val_device/{dt_name}/amp_err"] = dt_m["amp_err"]
                        log_dict[f"val_device/{dt_name}/phase_err"] = dt_m["phase_err"]
                        log_dict[f"val_device/{dt_name}/phase_err_deg"] = float(dt_m["phase_err"]) * 180.0 / math.pi
                        log_dict[f"val_device/{dt_name}/psnr"] = dt_m["psnr"]
                        if "sparam_mag_err" in dt_m:
                            log_dict[f"val_device/{dt_name}/sparam_mag_err"] = dt_m["sparam_mag_err"]
                            log_dict[f"val_device/{dt_name}/sparam_phase_err_deg"] = dt_m["sparam_phase_err_deg"]
                    # Upload eval sample images to wandb
                    if saved_paths:
                        try:
                            log_dict["val/sample_images"] = [
                                wandb.Image(p, caption=os.path.basename(p))
                                for p in saved_paths
                            ]
                        except Exception:
                            pass
                    # Upload dashboard and training curves plots
                    for plot_name in ["dashboard", "training_curves", "ratio_p95"]:
                        if plot_name == "ratio_p95":
                            plot_path = os.path.join(samples_dir, f"ratio_p95_epoch_{epoch:04d}.png")
                        else:
                            plot_path = os.path.join(samples_dir, f"{plot_name}_epoch_{epoch:04d}.png")
                        if os.path.isfile(plot_path):
                            try:
                                log_dict[f"val/{plot_name}"] = wandb.Image(plot_path)
                            except Exception:
                                pass
                    wandb_run.log(log_dict, step=epoch)

                # Optional: save checkpoint when worst per-device ratio p95 improves (lower is better)
                if (
                    (epoch % int(args.eval_every) == 0)
                    and str(getattr(args, "ckpt_best_metric", "") or "") == "worst_ratio_p95"
                    and worst_ratio_p95 is not None
                    and math.isfinite(float(worst_ratio_p95))
                    and float(worst_ratio_p95) < best_worst_ratio_p95
                ):
                    best_worst_ratio_p95 = float(worst_ratio_p95)
                    ckpt_best_path = os.path.join(ckpt_dir, "best_worst_ratio_p95.pt")
                    torch.save(
                        {
                            "model": _unwrap_model(model).state_dict(),
                            "ema": _unwrap_model(ema).state_dict(),
                            "opt": opt.state_dict(),
                            "args": args,
                            "stats": stats,
                            "epoch": epoch,
                            "worst_ratio_p95": float(worst_ratio_p95),
                        },
                        ckpt_best_path,
                    )
                    logger.info(
                        f"[epoch {epoch:04d}] New best worst_ratio_p95={float(worst_ratio_p95):.4f} "
                        f"-> saved {ckpt_best_path}"
                    )

        # Rejoin all ranks after the rank-0-only sample-eval block so nobody
        # races into the next collective until rank 0 is done.
        ddp_barrier(device)

        # -----------------------
        # Inverse design evaluation (rank0 only, separate frequency)
        # -----------------------
        inv_eval_every = int(getattr(args, "inverse_eval_every", 0))
        # Park non-rank-0 ranks at a barrier while rank 0 (maybe) runs the
        # inverse-design eval block; same rationale as the sample-eval barrier.
        _run_inv_eval = (
            joint_training
            and inv_eval_every > 0
            and epoch % inv_eval_every == 0
            and epoch >= int(args.phaseB_epochs)
        )
        if _run_inv_eval:
            ddp_barrier(device)
        if (
            _run_inv_eval
            and is_rank0()
        ):
            inv_samples = min(int(getattr(args, "inverse_eval_samples", 4)), len(val_ds))
            inv_cfg_scale = float(getattr(args, "inverse_eval_cfg_scale", 3.0))
            inv_steps = int(getattr(args, "inverse_eval_steps", 30))
            eps_core = float(getattr(args, "eps_core", 12.25))
            eps_clad = float(getattr(args, "eps_clad", 2.07))
            eps_binary_thr = (eps_core + eps_clad) / 2.0
            sparam_mode_inv = str(getattr(args, "sparam_mode", "project"))

            # Per-device-type metric accumulators
            inv_metrics_global = defaultdict(list)  # key -> list of floats
            inv_metrics_per_device = defaultdict(lambda: defaultdict(list))
            inv_saved_paths = []

            ema.eval()
            # Epoch-seeded random sample indices for more representative evaluation
            inv_rng = np.random.default_rng(epoch * 137 + 42)
            inv_indices = inv_rng.choice(len(val_ds), size=inv_samples, replace=False).tolist()
            logger.info(f"[epoch {epoch:04d}] Running inverse design eval ({inv_samples} samples, {inv_steps} steps, cfg={inv_cfg_scale})...")

            with torch.no_grad():
                for s_idx, s in enumerate(inv_indices):
                    try:
                        x_full_s, cond_s, aux_s = val_ds[s]
                        x_full_s = x_full_s.unsqueeze(0).to(device, dtype=dtype)
                        cond_s = cond_s.unsqueeze(0).to(device, dtype=dtype)

                        # Extract GT info
                        src_s = x_full_s[:, 3:4]
                        lambda_um_s = cond_s[:, 0:1] * lam_std + lam_mean

                        # Device type
                        device_type_s = aux_s.get("device_type", "") if isinstance(aux_s, dict) else ""
                        if not device_type_s:
                            device_type_s = "unknown"

                        # GT fields (de-normalized)
                        ezr_gt = x_full_s[:, 0:1] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        ezi_gt = x_full_s[:, 1:2] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        mag_gt_s = torch.sqrt(ezr_gt ** 2 + ezi_gt ** 2 + 1e-12)

                        # GT eps (de-normalized)
                        eps_gt_s = x_full_s[:, 2:3]
                        if args.normalize_eps:
                            eps_gt_s = eps_gt_s * float(stats["eps_std"]) + float(stats["eps_mean"])

                        # Run inverse sampling
                        gen_out = sample_inverse(
                            ema,
                            num_steps=inv_steps,
                            src_mask=src_s,
                            cond=cond_s,
                            lambda_um=lambda_um_s,
                            cfg_scale=inv_cfg_scale,
                            sig_min=SIG_MIN,
                            base_cond_dim=1,  # wavelength only (NOT cond_dim which includes S-params)
                        )

                        # De-normalize generated output: [B,3,H,W] = [fields(2), eps(1)]
                        gen_fields_r = gen_out[:, 0:1] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
                        gen_fields_i = gen_out[:, 1:2] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
                        gen_eps = gen_out[:, 2:3]
                        if args.normalize_eps:
                            gen_eps = gen_eps * float(stats["eps_std"]) + float(stats["eps_mean"])

                        mag_gen_s = torch.sqrt(gen_fields_r ** 2 + gen_fields_i ** 2 + 1e-12)

                        # Binarize eps
                        gen_eps_binary = torch.where(gen_eps > eps_binary_thr,
                                                     torch.tensor(eps_core, device=device, dtype=dtype),
                                                     torch.tensor(eps_clad, device=device, dtype=dtype))

                        # Binarization quality: fraction of pixels within 10% of binary values
                        eps_flat = gen_eps.view(-1)
                        near_core = (torch.abs(eps_flat - eps_core) < 0.1 * eps_core).float()
                        near_clad = (torch.abs(eps_flat - eps_clad) < 0.1 * eps_clad).float()
                        binarize_quality = float((near_core + near_clad).clamp(max=1.0).mean().item())

                        # Eps IoU: intersection-over-union between GT and generated binary masks
                        gt_binary = (eps_gt_s > eps_binary_thr).float()
                        gen_binary_mask = (gen_eps > eps_binary_thr).float()
                        intersection = (gt_binary * gen_binary_mask).sum()
                        union = ((gt_binary + gen_binary_mask) > 0).float().sum()
                        eps_iou = float((intersection / union.clamp_min(1.0)).item())

                        inv_metrics_global["binarize_quality"].append(binarize_quality)
                        inv_metrics_global["eps_iou"].append(eps_iou)
                        inv_metrics_per_device[device_type_s]["binarize_quality"].append(binarize_quality)
                        inv_metrics_per_device[device_type_s]["eps_iou"].append(eps_iou)

                        # S-param extraction from generated fields
                        aux_s_dev = _move_aux_to_device(aux_s, device) if isinstance(aux_s, dict) else {}
                        port_masks_s = aux_s_dev.get("port_masks", None)
                        in_port_idx_s = aux_s_dev.get("in_port_idx", None)
                        port_ids_s = aux_s_dev.get("port_ids", None)
                        port_valid_s = aux_s_dev.get("port_valid", None)
                        sparams_true_s = aux_s_dev.get("sparams_true", None)
                        n_ports_s = int(aux_s_dev.get("n_ports", torch.tensor(0)).item()) if aux_s_dev.get("n_ports") is not None else 0

                        sparam_text_lines = ["Port   |S| GT  |S| Gen  Err    Phase GT  Phase Gen  Err"]
                        sparam_text_lines.append("-" * 60)

                        if port_masks_s is not None and n_ports_s > 0 and sparams_true_s is not None:
                            # Build complex E field from generated output
                            E_gen = torch.complex(gen_fields_r[:, 0], gen_fields_i[:, 0])  # [B, H, W]

                            # Unsqueeze aux tensors for single-sample batch
                            pm = port_masks_s.unsqueeze(0) if port_masks_s.dim() == 3 else port_masks_s
                            pid = port_ids_s.unsqueeze(0) if port_ids_s is not None and port_ids_s.dim() == 1 else port_ids_s
                            ipi = in_port_idx_s

                            # Use binarized eps for modal extraction
                            eps_for_modal = gen_eps_binary[:, 0]  # [B, H, W]

                            try:
                                if sparam_mode_inv == "modal":
                                    S_gen = extract_sparams_modal(
                                        E_gen, pm, eps_for_modal,
                                        wavelength_um=lambda_um_s,
                                        dx_um=dx,
                                        in_port_idx=ipi,
                                        port_ids=pid,
                                    )
                                else:
                                    S_gen = extract_sparams(
                                        E_gen, pm,
                                        in_port_idx=ipi,
                                        port_ids=pid,
                                    )
                            except Exception:
                                # Fallback to project method
                                S_gen = extract_sparams(
                                    E_gen, pm,
                                    in_port_idx=ipi,
                                    port_ids=pid,
                                )

                            S_true_1 = sparams_true_s.unsqueeze(0) if sparams_true_s.dim() == 1 else sparams_true_s
                            S_gen_0 = S_gen[0].detach().cpu()
                            S_true_0 = S_true_1[0].detach().cpu()
                            pv = port_valid_s.detach().cpu() if port_valid_s is not None else None

                            # Resolve input port index
                            in_idx_val = 0
                            if ipi is not None:
                                if torch.is_tensor(ipi):
                                    in_idx_val = int(ipi.item()) if ipi.numel() == 1 else int(ipi[0].item())
                                else:
                                    in_idx_val = int(ipi)
                            if in_idx_val < 0:
                                in_idx_val = 0

                            # Resolve port IDs for labeling
                            pid_np = port_ids_s.detach().cpu().numpy() if port_ids_s is not None else np.arange(1, n_ports_s + 1)
                            if pid_np.ndim > 1:
                                pid_np = pid_np[0]
                            in_pid = int(pid_np[in_idx_val]) if in_idx_val < len(pid_np) else 1

                            for p in range(n_ports_s):
                                # Skip input port
                                if p == in_idx_val:
                                    continue
                                if pv is not None and float(pv[p].item() if pv.dim() == 1 else pv[0, p].item()) < 0.5:
                                    continue

                                out_pid = int(pid_np[p]) if p < len(pid_np) else (p + 1)
                                label = f"S{out_pid}{in_pid}"

                                mag_true = float(torch.abs(S_true_0[p]).item())
                                mag_gen = float(torch.abs(S_gen_0[p]).item())
                                mag_err = abs(mag_true - mag_gen)

                                phase_true = float(torch.angle(S_true_0[p]).item())
                                phase_gen = float(torch.angle(S_gen_0[p]).item())
                                phase_diff = float(torch.atan2(
                                    torch.sin(torch.tensor(phase_gen - phase_true)),
                                    torch.cos(torch.tensor(phase_gen - phase_true)),
                                ).item())

                                inv_metrics_global["sparam_mag_err"].append(mag_err)
                                inv_metrics_global["sparam_phase_err"].append(abs(phase_diff))
                                inv_metrics_per_device[device_type_s]["sparam_mag_err"].append(mag_err)
                                inv_metrics_per_device[device_type_s]["sparam_phase_err"].append(abs(phase_diff))

                                sparam_text_lines.append(
                                    f"{label:6s} {mag_true:6.3f}  {mag_gen:6.3f}  {mag_err:5.3f}   "
                                    f"{phase_true:+6.2f}     {phase_gen:+6.2f}     {abs(phase_diff):5.2f}"
                                )

                        sparam_text = "\n".join(sparam_text_lines)

                        # Save PNG for this sample
                        inv_out_path = os.path.join(samples_dir, f"inverse_epoch_{epoch:04d}_idx{s_idx:03d}.png")
                        pm_overlay = None
                        if port_masks_s is not None and n_ports_s > 0:
                            pm_np = port_masks_s.detach().cpu().numpy()
                            pm_overlay = pm_np[:n_ports_s].sum(axis=0)  # [H, W]

                        _save_inverse_eval_png(
                            out_path=inv_out_path,
                            title=f"epoch={epoch:04d} val[{s}]",
                            eps_gt=eps_gt_s[0, 0].detach().float().cpu().numpy(),
                            eps_gen=gen_eps[0, 0].detach().float().cpu().numpy(),
                            eps_gen_binary=gen_eps_binary[0, 0].detach().float().cpu().numpy(),
                            mag_gt=mag_gt_s[0, 0].detach().float().cpu().numpy(),
                            mag_gen=mag_gen_s[0, 0].detach().float().cpu().numpy(),
                            port_masks_overlay=pm_overlay,
                            sparam_text=sparam_text,
                            device_type=device_type_s,
                            ezr_gt=ezr_gt[0, 0].detach().float().cpu().numpy(),
                            ezi_gt=ezi_gt[0, 0].detach().float().cpu().numpy(),
                            ezr_gen=gen_fields_r[0, 0].detach().float().cpu().numpy(),
                            ezi_gen=gen_fields_i[0, 0].detach().float().cpu().numpy(),
                        )
                        if os.path.isfile(inv_out_path):
                            inv_saved_paths.append(inv_out_path)

                    except Exception as exc:
                        logger.warning(f"[epoch {epoch:04d}] Inverse eval sample {s} failed: {exc}")
                        continue

            # Aggregate and log inverse eval metrics
            def _mean_or_zero(lst):
                return float(np.mean(lst)) if lst else 0.0

            g_mag_err = _mean_or_zero(inv_metrics_global.get("sparam_mag_err", []))
            g_phase_err = _mean_or_zero(inv_metrics_global.get("sparam_phase_err", []))
            g_bin_q = _mean_or_zero(inv_metrics_global.get("binarize_quality", []))
            g_iou = _mean_or_zero(inv_metrics_global.get("eps_iou", []))

            msg = (
                f"[epoch {epoch:04d}] INVERSE EVAL: |S| err={g_mag_err:.4f}, "
                f"phase err={g_phase_err:.4f} rad, binarize_q={g_bin_q:.3f}, eps_IoU={g_iou:.3f}"
            )
            for dt_name, dt_metrics in sorted(inv_metrics_per_device.items()):
                dt_mag = _mean_or_zero(dt_metrics.get("sparam_mag_err", []))
                dt_ph = _mean_or_zero(dt_metrics.get("sparam_phase_err", []))
                dt_bq = _mean_or_zero(dt_metrics.get("binarize_quality", []))
                dt_iou = _mean_or_zero(dt_metrics.get("eps_iou", []))
                msg += f"\n  [{dt_name}] |S| err={dt_mag:.4f}, phase err={dt_ph:.4f}, binarize_q={dt_bq:.3f}, eps_IoU={dt_iou:.3f}"
            logger.info(msg)

            if wandb_run is not None:
                inv_log = {
                    "epoch": epoch,
                    "val_inverse/sparam_mag_err": g_mag_err,
                    "val_inverse/sparam_phase_err": g_phase_err,
                    "val_inverse/binarize_quality": g_bin_q,
                    "val_inverse/eps_iou": g_iou,
                }
                for dt_name, dt_metrics in inv_metrics_per_device.items():
                    prefix = f"val_inverse/{dt_name}"
                    inv_log[f"{prefix}/sparam_mag_err"] = _mean_or_zero(dt_metrics.get("sparam_mag_err", []))
                    inv_log[f"{prefix}/sparam_phase_err"] = _mean_or_zero(dt_metrics.get("sparam_phase_err", []))
                    inv_log[f"{prefix}/binarize_quality"] = _mean_or_zero(dt_metrics.get("binarize_quality", []))
                    inv_log[f"{prefix}/eps_iou"] = _mean_or_zero(dt_metrics.get("eps_iou", []))
                if inv_saved_paths:
                    try:
                        inv_log["val_inverse/sample_images"] = [
                            wandb.Image(p, caption=os.path.basename(p))
                            for p in inv_saved_paths
                        ]
                    except Exception:
                        pass
                wandb_run.log(inv_log, step=epoch)

        # Rejoin all ranks after the rank-0-only inverse-design eval block.
        if _run_inv_eval:
            ddp_barrier(device)

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
                msg = f"(epoch={epoch:04d}) fm={fm_mean.item():.4e} res={res_mean.item():.4e}"
                if phase_weight > 0.0:
                    msg += f" ph={ph_mean.item():.4e}"
                    try:
                        ph_rms_deg = (math.sqrt(max(0.0, 2.0 * float(ph_mean.item()))) * 180.0 / math.pi)
                        msg += f"({ph_rms_deg:.1f}°)"
                    except Exception:
                        pass
                if use_endpoint:
                    msg += f" end={end_mean.item():.4e}"
                if phase_grad_weight > 0.0:
                    msg += f" phg={phg_mean.item():.4e}"
                if sparam_weight > 0.0:
                    msg += f" sp={sp_mean.item():.4e}"
                if joint_training and binarize_weight > 0.0:
                    bin_mean_val = running_binarize_loss / max(train_steps_epoch, 1)
                    msg += f" bin={bin_mean_val:.4e}"
                if joint_training and geom_weight > 0.0:
                    geom_mean_val = running_geom_loss / max(train_steps_epoch, 1)
                    msg += f" geom={geom_mean_val:.4e}"
                msg += f" gnorm={running_grad_norm / max(train_steps_epoch, 1):.2f}/{grad_norm_max_epoch:.2f}"
                msg += f" [{sec_per_epoch:.0f}s]"
                # Only print weights when they change from previous epoch
                cur_weights = (residual_weight, phase_weight, phase_grad_weight, sparam_weight, endpoint_weight)
                if cur_weights != _prev_weights:
                    msg += f"\n  [weights] res={residual_weight:.3g} ph={phase_weight:.3g} phg={phase_grad_weight:.3g} sp={sparam_weight:.3g} end={endpoint_weight:.3g}"
                    _prev_weights = cur_weights
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
                    "train/grad_norm_mean": running_grad_norm / max(train_steps_epoch, 1),
                    "train/grad_norm_max": grad_norm_max_epoch,
                }
                if phase_weight > 0.0:
                    wandb_log_dict["train/phase_loss"] = ph_mean.item()
                    try:
                        wandb_log_dict["train/phase_rms_deg"] = math.sqrt(max(0.0, 2.0 * float(ph_mean.item()))) * 180.0 / math.pi
                    except Exception:
                        pass
                if use_endpoint:
                    wandb_log_dict["train/endpoint_loss"] = end_mean.item()
                    wandb_log_dict["train/endpoint_weight"] = endpoint_weight
                if phase_grad_weight > 0.0:
                    wandb_log_dict["train/phase_grad_loss"] = phg_mean.item()
                    wandb_log_dict["train/phase_grad_steps_epoch"] = int(phase_grad_steps_epoch)
                if sparam_weight > 0.0:
                    wandb_log_dict["train/sparam_loss"] = sp_mean.item()
                if joint_training and binarize_weight > 0.0:
                    wandb_log_dict["train/binarize_loss"] = running_binarize_loss / max(train_steps_epoch, 1)
                if joint_training and geom_weight > 0.0:
                    wandb_log_dict["train/geom_loss"] = running_geom_loss / max(train_steps_epoch, 1)

                wandb_run.log(wandb_log_dict, step=epoch)

            # Gradient health plot (rank0 only, during eval epochs)
            if is_rank0() and layer_grad_norms_accum and epoch % int(args.eval_every) == 0:
                # Average accumulated per-layer norms across the epoch
                layer_grad_norms = {k: float(np.mean(v)) for k, v in layer_grad_norms_accum.items()}
                grad_health_path = os.path.join(samples_dir, f"grad_health_epoch_{epoch:04d}.png")
                _save_gradient_health_png(
                    out_path=grad_health_path,
                    title=f"Gradient Health - Epoch {epoch}",
                    layer_grad_norms=layer_grad_norms,
                    epoch=epoch,
                )
                if wandb_run is not None and os.path.isfile(grad_health_path):
                    try:
                        wandb_run.log({"train/grad_health": wandb.Image(grad_health_path)}, step=epoch)
                    except Exception:
                        pass

            start_time = time()
            running_fm_loss = 0.0
            running_residual_loss = 0.0
            running_phase_loss = 0.0
            running_endpoint_loss = 0.0
            running_phase_grad_loss = 0.0
            running_sparam_loss = 0.0
            running_binarize_loss = 0.0
            running_geom_loss = 0.0
            running_grad_norm = 0.0
            grad_norm_max_epoch = 0.0
            layer_grad_norms_accum = defaultdict(list)
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
                    "model": _unwrap_model(model).state_dict(),
                    "ema": _unwrap_model(ema).state_dict(),
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
    parser.add_argument("--include-sweeps", type=str, default="directional_coupler_sweep",
                        help="Comma-separated sweep subdirectory names to include (empty = all)")
    parser.add_argument("--exclude-devices", type=str, default="",
                        help="Comma-separated device types to exclude (e.g. 'straight,mmi')")
    parser.add_argument("--include-wavelengths", type=str, default="",
                        help="Comma-separated wavelengths in µm to include (e.g. '1.55'). Empty = all.")
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

    parser.add_argument("--canvas-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Center-pad all samples to (H, W) for mixed-domain datasets, e.g. --canvas-hw 320 480")

    parser.add_argument("--augment", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Enable D4/D2 augmentation during training (D2 for rectangular canvas)")

    parser.add_argument("--include-sdf", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--normalize-sdf", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sdf-thr-eps", dest="sdf_thr_eps", type=float, default=3.0)
    parser.add_argument("--sdf-feature", type=str, default="raw", choices=["raw", "clip", "exp", "clip_exp"])
    parser.add_argument("--sdf-sigma-px", type=float, default=0.0)
    parser.add_argument("--sdf-sigma-nm", type=float, default=100.0)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Residual blocks per UNet level")
    parser.add_argument("--channel-mult", type=str, default="1,2,4,8", help="Channel multipliers per level (comma-separated)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--attn-resolutions", type=str, default="8", help="Attention at these downsample factors (comma-separated, empty=none)")
    parser.add_argument("--no-attention", action="store_true", help="Disable attention layers to reduce memory")
    parser.add_argument("--model-dropout", type=float, default=0.0, help="Dropout rate in residual blocks")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Max gradient norm for clipping")

    # ConFIG conflict-free gradient combination (uses 'conflictfree' library).
    parser.add_argument("--config", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--config-start-epoch", type=int, default=200)

    parser.add_argument("--unroll-steps", type=int, default=0)
    parser.add_argument("--unroll-phase", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--phase-amp-tau", type=float, default=0.2)

    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"],
                        help="AMP dtype: float16 (needs GradScaler) or bfloat16 (no scaler, more stable)")
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
    parser.add_argument("--endpoint-start-phase", type=str, default="B", choices=["A", "B", "C"])

    parser.add_argument("--lambda-phase-grad", type=float, default=0.0)
    parser.add_argument("--phase-grad-warmup-epochs", type=int, default=100)
    parser.add_argument("--phase-grad-every", type=int, default=4)
    parser.add_argument("--phase-grad-start-phase", type=str, default="C", choices=["A", "B", "C"])

    parser.add_argument("--lambda-sparam", type=float, default=0.0)
    parser.add_argument("--sparam-from-start", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sparam-warmup-epochs", type=int, default=50)
    parser.add_argument("--sparam-every", type=int, default=1)
    parser.add_argument("--sparam-start-phase", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--sparam-mode", type=str, default="project", choices=["project", "modal", "head"])

    parser.add_argument("--normalize-eps", type=bool, default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--physics-features", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--complex-unet", type=bool, default=False, action=argparse.BooleanOptionalAction)

    # Joint training (forward + inverse design)
    parser.add_argument("--joint-training", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--forward-ratio", type=float, default=0.5,
                        help="Probability of forward mask mode per batch")
    parser.add_argument("--inverse-ratio", type=float, default=0.3,
                        help="Probability of inverse mask mode per batch")
    parser.add_argument("--cfg-dropout", type=float, default=0.15,
                        help="Probability of zeroing S-param conditioning (for CFG)")
    parser.add_argument("--eps-core", type=float, default=12.25,
                        help="Core (silicon) permittivity for binarization loss")
    parser.add_argument("--eps-clad", type=float, default=2.07,
                        help="Cladding (oxide) permittivity for binarization loss")
    parser.add_argument("--lambda-binarize", type=float, default=0.01,
                        help="Weight for binarization loss on generated eps")
    parser.add_argument("--binarize-warmup-epochs", type=int, default=100,
                        help="Epochs to linearly ramp binarization loss")
    parser.add_argument("--lambda-geom", type=float, default=0.0,
                        help="Weight for pixel-wise geometry MSE loss on generated eps")
    parser.add_argument("--geom-warmup-epochs", type=int, default=100,
                        help="Epochs to linearly ramp geometry loss")
    parser.add_argument("--include-sparams-cond", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Include S-params in conditioning vector (required for joint/inverse)")

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--sample-eval-limit", type=int, default=16,
                        help="Total sample-eval budget; split across device types unless --eval-samples-per-device is set")
    parser.add_argument(
        "--eval-sample-mode",
        type=str,
        default="fixed",
        choices=["epoch_random", "fixed"],
        help="Sample selection: epoch_random (legacy, changes each eval) or fixed (same stratified draw every epoch)",
    )
    parser.add_argument(
        "--eval-sample-index-seed",
        type=int,
        default=0,
        help="RNG seed for stratified val indices when --eval-sample-mode=fixed (ignored if --eval-sample-indices-json is set)",
    )
    parser.add_argument(
        "--eval-sample-indices-json",
        type=str,
        default="",
        help="Optional JSON map {device_type: [global_val_idx, ...]}; overrides random stratified selection",
    )
    parser.add_argument(
        "--eval-samples-per-device",
        type=int,
        default=0,
        help="If >0, draw this many validation samples per device type; else derive from sample-eval-limit",
    )
    parser.add_argument(
        "--eval-flow-noise-seed",
        type=int,
        default=0,
        help="If >=0, deterministic FM noise x0 per val index (reproducible ODE); if -1, fresh randn each eval (legacy)",
    )
    parser.add_argument(
        "--physical-ratio-target",
        type=float,
        default=1.5,
        help="Target max pred/GT residual ratio for acceptance flags (logged vs per-device p95/max)",
    )
    parser.add_argument(
        "--ckpt-best-metric",
        type=str,
        default="",
        choices=["", "worst_ratio_p95"],
        help="If worst_ratio_p95, save checkpoints/best_worst_ratio_p95.pt when val worst per-device ratio p95 improves",
    )
    parser.add_argument("--save-eval-samples", dest="save_eval_samples", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-eval-samples-limit", dest="save_eval_samples_limit", type=int, default=1)

    # Inverse design evaluation
    parser.add_argument("--inverse-eval-every", type=int, default=50,
                        help="Run inverse design eval every N epochs (0=disable)")
    parser.add_argument("--inverse-eval-samples", type=int, default=4,
                        help="Number of val samples for inverse eval")
    parser.add_argument("--inverse-eval-cfg-scale", type=float, default=3.0,
                        help="CFG scale for inverse design sampling")
    parser.add_argument("--inverse-eval-steps", type=int, default=30,
                        help="ODE integration steps for inverse eval")

    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--auto-resume", action="store_true", default=False,
                        help="Automatically resume from latest checkpoint in ckpt_dir if available")
    parser.add_argument("--reset-lr-on-resume", action="store_true", default=False,
                        help="Reset LR schedule to fresh cosine from --lr on resume")
    parser.add_argument("--reset-optimizer", action="store_true", default=False,
                        help="Skip loading optimizer state on resume (fresh optimizer, keeps model weights)")
    parser.add_argument("--t-physics-min", type=float, default=0.0,
                        help="Min t for physics losses (residual/phase/sparam). FM always at all t.")

    # Time sampling distribution
    parser.add_argument("--t-sample-mode", dest="t_sample_mode", type=str, default="uniform",
                        choices=["uniform", "logit_normal", "mixture"],
                        help="Time sampling distribution for FM training.")
    parser.add_argument("--t-sample-loc", dest="t_sample_loc", type=float, default=0.0,
                        help="Location param for logit-normal time sampling (logit-space mean).")
    parser.add_argument("--t-sample-scale", dest="t_sample_scale", type=float, default=1.0,
                        help="Scale param for logit-normal time sampling (logit-space std).")

    # Residual t-power weighting
    parser.add_argument("--residual-t-power", dest="residual_t_power", type=float, default=0.0,
                        help="If >0, weight per-sample residual loss by t^p (upweights clean high-t reconstructions).")

    # ODE sampler time grid
    parser.add_argument("--time-grid", dest="time_grid", type=str, default="quadratic",
                        choices=["quadratic", "linear"],
                        help="ODE sampler time grid: 'quadratic' (more steps near t=0) or 'linear' (uniform).")

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
    parser.add_argument("--interface-thr", type=float, default=0.0,
                        help="Exclude pixels with |grad(eps)| > thr from Helmholtz residual "
                             "(removes dielectric interface artifacts). 0 = disabled. "
                             "Recommended: 1.0 for physical-unit eps.")

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
