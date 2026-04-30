#!/usr/bin/env python3
"""Full test-set inference and per-family metric histograms.

Sweeps the held-out `test` split of the unified MMI / Y-branch / directional
coupler dataset, runs the FM+phase+residual checkpoint on every sample, and
writes a 1x3 panel of overlaid histograms for the three metrics reported in
the paper:

    - eps_R : Helmholtz-residual compliance percentage (paper Eq. 9).
    - PSNR  : (Re, Im) pair PSNR vs FDTD reference, dB.
    - phase : intensity-weighted wrap-safe cosine distance (paper Eq. 10).

Per-sample metrics are cached to a single .npz so re-running with --plot-only
skips inference entirely.

Usage
-----
    python tools/test_set_histograms.py \
        --data-root Data/unified_sweep_mmi_ybranch_dc_7500_each_1p55um \
        --ckpt /path/to/0000300.pt \
        --num-steps 100 \
        --usetex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Model"
TOOLS_DIR = REPO_ROOT / "tools"
for _p in (str(MODEL_DIR), str(TOOLS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import phase_anchor_mask, phase_anchor_roi  # noqa: E402
from flow_matching import sample as fm_sample  # noqa: E402
from predict_parametric_device import (  # noqa: E402
    _build_cond_maps,
    _build_cond_vector,
    _build_model_from_checkpoint,
    _checkpoint_state_dict,
    _ckpt_get,
)
from _triptych_metrics import residual_metrics  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "Data" / "unified_sweep_mmi_ybranch_dc_7500_each_1p55um"
DEFAULT_CKPT = (
    Path("/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm")
    / "REAL_VS_COMPLEX_real_h56_phase_residual_v100x12"
    / "checkpoints"
    / "0000300.pt"
)
DEVICE_ORDER = ("mmi", "ybranch", "directional_coupler")
DEVICE_LABELS = {
    "mmi": r"$2\times2$ MMI",
    "ybranch": "Y-branch",
    "directional_coupler": "Directional coupler",
}
TRAINED_SOURCE_PORTS = {
    "mmi": (1, 2),
    "ybranch": (1, 2, 3),
    "directional_coupler": (1, 2),
}
DEVICE_COLORS = {
    "mmi": "#1f77b4",
    "ybranch": "#d62728",
    "directional_coupler": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Sample loading (mirrors make_paper_inference_triptychs._load_shard_sample)
# ---------------------------------------------------------------------------

def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _load_index(data_root: Path) -> list[dict[str, Any]]:
    index_path = data_root / "shards" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing shard index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def _slot_prefix(entry: dict[str, Any]) -> str:
    return f"s{int(entry['slot'])}/"


def _load_shard_sample(data_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    shard_path = data_root / "shards" / str(entry["shard"])
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    data = np.load(shard_path, allow_pickle=False)
    p = _slot_prefix(entry)

    return {
        "device": str(_scalar(data, p + "device", entry.get("device", ""))),
        "geometry_id": str(_scalar(data, p + "geometry_id", entry.get("geometry_id", ""))),
        "input_port": int(_scalar(data, p + "input_port")),
        "wavelength_um": float(_scalar(data, p + "wavelength_um")),
        "eps": np.asarray(data[p + "eps"], dtype=np.float32),
        "ez_real": np.asarray(data[p + "Ez_real"], dtype=np.float32),
        "ez_imag": np.asarray(data[p + "Ez_imag"], dtype=np.float32),
        "src_mask": np.asarray(data[p + "src_mask"], dtype=np.float32),
        "dx_um": float(_scalar(data, p + "dx_um", 0.05)),
        "dy_um": float(_scalar(data, p + "dy_um", 0.05)),
    }


def _anchor_reference(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ezr, ezi, eps, src = sample["ez_real"], sample["ez_imag"], sample["eps"], sample["src_mask"]
    ezr2, ezi2, _ = phase_anchor_mask(ezr, ezi, src > 0.5, eps_r=eps, thr_eps=3.0)
    if float((src > 0.5).sum()) < 4.0:
        ezr2, ezi2, _ = phase_anchor_roi(
            ezr, ezi, eps_r=eps, pml_cells=0, margin=2, roi_x=(40, 140), thr_eps=3.0
        )
    return ezr2.astype(np.float32, copy=False), ezi2.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_sample(
    sample: dict[str, Any],
    *,
    ckpt: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    num_steps: int,
    time_grid: str,
    seed: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    eps, src = sample["eps"], sample["src_mask"]
    wavelength_um = float(sample["wavelength_um"])

    cond_maps = _build_cond_maps(
        eps, src, stats=stats, ckpt_args=ckpt_args, device=device,
        dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
    )
    cond = _build_cond_vector(wavelength_um, stats, device=device)
    lambda_um_t = torch.tensor([[wavelength_um]], device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x0 = torch.randn(
        (1, 2, eps.shape[0], eps.shape[1]),
        device=device, dtype=torch.float32, generator=gen,
    )
    if time_grid == "checkpoint":
        time_grid = str(_ckpt_get(ckpt_args, "time_grid", "linear"))
    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            fields_norm = fm_sample(
                model, x0,
                num_steps=int(num_steps),
                use_stoc_samp=False,
                cond_maps=cond_maps,
                cond=cond,
                lambda_um=lambda_um_t,
                phys_gate=1.0,
                phase_gate=1.0,
                time_grid=time_grid,
                progress=False,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    fn = fields_norm.float()
    ezr = fn[0, 0].detach().cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fn[0, 1].detach().cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    return ezr.astype(np.float32, copy=False), ezi.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Metrics: PSNR + intensity-weighted phase error
# ---------------------------------------------------------------------------

def _psnr_db(
    pred_r: np.ndarray, pred_i: np.ndarray,
    fdtd_r: np.ndarray, fdtd_i: np.ndarray,
) -> float:
    """PSNR in dB on the field magnitude |E_z|.

    Matches Model/train.py:_compute_psnr — peak signal is data_range
    (gt.max() - gt.min()) of the magnitude, MSE is over |E|.
    """
    fmag = np.sqrt(fdtd_r.astype(np.float64) ** 2 + fdtd_i.astype(np.float64) ** 2)
    pmag = np.sqrt(pred_r.astype(np.float64) ** 2 + pred_i.astype(np.float64) ** 2)
    mse = float(np.mean((pmag - fmag) ** 2))
    if mse <= 1e-15:
        return 100.0
    data_range = float(fmag.max() - fmag.min())
    if data_range <= 1e-15:
        return float("nan")
    return 10.0 * math.log10(data_range * data_range / mse)


def _phase_error(
    pred_r: np.ndarray, pred_i: np.ndarray,
    fdtd_r: np.ndarray, fdtd_i: np.ndarray,
    eps: np.ndarray,
    *,
    eps_thr: float = 3.0,
    amp_quantile: float = 0.5,
) -> float:
    """Intensity-weighted wrap-safe cosine distance (paper Eq. 10).

    Restricted to high-permittivity pixels (eps > eps_thr) and amplitudes above
    the median of |E_FDTD| within that mask, so we don't average phase noise
    in regions where the field is essentially zero.
    """
    fr = fdtd_r.astype(np.float64)
    fi = fdtd_i.astype(np.float64)
    pr = pred_r.astype(np.float64)
    pi = pred_i.astype(np.float64)
    fmag2 = fr * fr + fi * fi
    pmag2 = pr * pr + pi * pi
    region = eps > float(eps_thr)
    if not np.any(region):
        return float("nan")
    thr = float(np.quantile(np.sqrt(fmag2[region]), amp_quantile))
    valid = region & (np.sqrt(fmag2) >= thr) & (np.sqrt(pmag2) > 0.0)
    if not np.any(valid):
        return float("nan")
    cos_term = (pr * fr + pi * fi) / (np.sqrt(pmag2 * fmag2) + 1e-30)
    weights = fmag2[valid]
    num = float(np.sum((1.0 - cos_term[valid]) * weights))
    den = float(np.sum(weights)) + 1e-30
    return num / den


# ---------------------------------------------------------------------------
# Driver: sweep test split, compute metrics, cache
# ---------------------------------------------------------------------------

def _compute_metrics_for_sample(
    sample: dict[str, Any],
    *,
    ckpt: dict[str, Any],
    model: torch.nn.Module,
    runtime_device: torch.device,
    num_steps: int,
    time_grid: str,
    seed: int,
    amp: bool,
) -> dict[str, float]:
    fdtd_r, fdtd_i = _anchor_reference(sample)
    pred_r, pred_i = _predict_sample(
        sample,
        ckpt=ckpt, model=model, device=runtime_device,
        num_steps=num_steps, time_grid=time_grid, seed=seed, amp=amp,
    )
    res = residual_metrics(
        sample["eps"], sample["src_mask"],
        fdtd_r, fdtd_i, pred_r, pred_i,
        dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
        wavelength_um=float(sample["wavelength_um"]),
    )
    return {
        "eps_R": float(res.get("eps_R_pct", float("nan"))),
        "psnr_db": _psnr_db(pred_r, pred_i, fdtd_r, fdtd_i),
        "phase_err": _phase_error(pred_r, pred_i, fdtd_r, fdtd_i, sample["eps"]),
    }


def _run_inference_sweep(args: argparse.Namespace, runtime_device: torch.device,
                         data_root: Path) -> dict[str, np.ndarray]:
    print(f"[hist] checkpoint: {args.ckpt}")
    print(f"[hist] data_root: {data_root}")
    print(f"[hist] runtime_device: {runtime_device}")
    print(f"[hist] num_steps: {args.num_steps} ({args.time_grid})")

    ckpt = torch.load(args.ckpt, map_location=runtime_device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[hist] loaded weights: {state_key}")

    devices = tuple(d.strip() for d in args.devices.split(",") if d.strip())
    index = _load_index(data_root)
    candidates = [
        e for e in index
        if e.get("device") in devices
        and e.get("split") == args.split
        and e.get("augment", "orig") == "orig"
    ]
    if args.max_per_device > 0:
        per_dev_count: dict[str, int] = {}
        kept = []
        for e in candidates:
            d = e.get("device", "")
            per_dev_count[d] = per_dev_count.get(d, 0) + 1
            if per_dev_count[d] <= int(args.max_per_device):
                kept.append(e)
        candidates = kept

    print(f"[hist] {len(candidates)} {args.split}-split entries across {len(devices)} families")

    families: list[str] = []
    eps_R: list[float] = []
    psnr: list[float] = []
    phase: list[float] = []
    geom_ids: list[str] = []

    t0 = time.perf_counter()
    last_print = t0
    for i, entry in enumerate(candidates):
        sample = _load_shard_sample(data_root, entry)
        if sample["input_port"] not in TRAINED_SOURCE_PORTS.get(sample["device"], ()):
            continue
        m = _compute_metrics_for_sample(
            sample,
            ckpt=ckpt, model=model, runtime_device=runtime_device,
            num_steps=int(args.num_steps), time_grid=str(args.time_grid),
            seed=int(args.seed) + i, amp=bool(args.amp),
        )
        families.append(sample["device"])
        eps_R.append(m["eps_R"])
        psnr.append(m["psnr_db"])
        phase.append(m["phase_err"])
        geom_ids.append(sample["geometry_id"])

        now = time.perf_counter()
        if now - last_print > 5.0 or (i + 1) == len(candidates):
            done = len(eps_R)
            rate = done / max(now - t0, 1e-6)
            remain = (len(candidates) - (i + 1)) / max(rate, 1e-6)
            print(f"[hist] {done}/{len(candidates)}  rate={rate:.2f} sample/s  ETA={remain/60:.1f} min")
            last_print = now

    return {
        "family": np.asarray(families),
        "eps_R": np.asarray(eps_R, dtype=np.float64),
        "psnr_db": np.asarray(psnr, dtype=np.float64),
        "phase_err": np.asarray(phase, dtype=np.float64),
        "geometry_id": np.asarray(geom_ids),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _setup_matplotlib(usetex: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": bool(usetex),
    })


def _plot_panel(
    ax: Any,
    data: dict[str, np.ndarray],
    *,
    metric: str,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    log_x: bool,
) -> None:
    import matplotlib.pyplot as plt  # noqa: F401
    fam = data["family"]
    vals = data[metric]
    keep = np.isfinite(vals)
    fam = fam[keep]
    vals = vals[keep]

    for dev in DEVICE_ORDER:
        sel = vals[fam == dev]
        if sel.size == 0:
            continue
        plot_vals = np.log10(np.maximum(sel, 1e-12)) if log_x else sel
        ax.hist(
            plot_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=DEVICE_COLORS[dev],
            edgecolor=DEVICE_COLORS[dev],
            alpha=0.45,
            linewidth=1.1,
            label=DEVICE_LABELS[dev],
        )

    p95 = float(np.nanpercentile(vals, 95))
    p99 = float(np.nanpercentile(vals, 99))
    p95p = math.log10(max(p95, 1e-12)) if log_x else p95
    p99p = math.log10(max(p99, 1e-12)) if log_x else p99
    ax.axvline(p95p, color="0.25", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(p99p, color="0.25", linestyle=":",  linewidth=0.8, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(p95p, ymax * 0.92, "p95", rotation=90, va="top", ha="right",
            fontsize=7.5, color="0.25")
    ax.text(p99p, ymax * 0.92, "p99", rotation=90, va="top", ha="right",
            fontsize=7.5, color="0.25")

    if log_x:
        ax.set_xlabel(xlabel + r" (log$_{10}$)")
    else:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel("density")


def _plot_histograms(data: dict[str, np.ndarray], *, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), constrained_layout=True)

    eps_finite = data["eps_R"][np.isfinite(data["eps_R"])]
    eps_log = np.log10(np.maximum(eps_finite, 1e-3))
    eps_bins = np.linspace(np.percentile(eps_log, 1), np.percentile(eps_log, 99), 36)
    _plot_panel(
        axes[0], data,
        metric="eps_R", bins=eps_bins,
        xlabel=r"compliance $\varepsilon_R$ (\%)" if plt.rcParams["text.usetex"]
               else r"compliance $\varepsilon_R$ (%)",
        title="Helmholtz compliance",
        log_x=True,
    )

    psnr_finite = data["psnr_db"][np.isfinite(data["psnr_db"])]
    psnr_bins = np.linspace(np.percentile(psnr_finite, 1), np.percentile(psnr_finite, 99), 36)
    _plot_panel(
        axes[1], data,
        metric="psnr_db", bins=psnr_bins,
        xlabel="PSNR (dB)",
        title="Reconstruction PSNR",
        log_x=False,
    )

    ph_finite = data["phase_err"][np.isfinite(data["phase_err"])]
    ph_bins = np.linspace(np.percentile(ph_finite, 1), np.percentile(ph_finite, 99), 36)
    _plot_panel(
        axes[2], data,
        metric="phase_err", bins=ph_bins,
        xlabel=r"phase cosine distance $\mathcal{L}_\phi$",
        title="Phase fidelity",
        log_x=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[hist] saved: {out_path}")
    print(f"[hist] saved: {out_path.with_suffix('.pdf')}")


def _print_summary(data: dict[str, np.ndarray]) -> None:
    fam = data["family"]
    print()
    print(f"  {'family':<22}  {'N':>4}  {'eps_R p50/p95/p99':>22}  "
          f"{'PSNR p50 (dB)':>14}  {'phase p50':>10}")
    for dev in DEVICE_ORDER:
        sel = fam == dev
        n = int(sel.sum())
        if n == 0:
            continue
        e = data["eps_R"][sel]
        p = data["psnr_db"][sel]
        ph = data["phase_err"][sel]
        e = e[np.isfinite(e)]
        p = p[np.isfinite(p)]
        ph = ph[np.isfinite(ph)]
        e_str = f"{np.median(e):.2f} / {np.percentile(e,95):.2f} / {np.percentile(e,99):.2f}"
        print(
            f"  {DEVICE_LABELS[dev]:<22}  {n:>4}  {e_str:>22}  "
            f"{np.median(p):>14.2f}  {np.median(ph):>10.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on the test split and plot per-family metric histograms.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="FM+phase+residual checkpoint.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset root containing shards/index.json.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "test_set_histograms"))
    parser.add_argument("--devices", default=",".join(DEVICE_ORDER), help="Comma-separated families to include.")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="checkpoint")
    parser.add_argument("--max-per-device", type=int, default=0, help="Cap samples per family (0 = unlimited).")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-only", action="store_true", help="Skip inference; load cached metrics.")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cache exists.")
    parser.add_argument("--usetex", action="store_true", help="Use external LaTeX for text rendering.")
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "metrics.npz"

    if args.plot_only or (cache_path.is_file() and not args.force):
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cache not found: {cache_path}. Run without --plot-only first.")
        npz = np.load(cache_path, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
        print(f"[hist] loaded cache: {cache_path} ({len(data['eps_R'])} samples)")
    else:
        data = _run_inference_sweep(args, runtime_device, data_root)
        np.savez_compressed(cache_path, **data)
        print(f"[hist] saved cache: {cache_path}")

    _setup_matplotlib(usetex=bool(args.usetex))
    _plot_histograms(data, out_path=out_dir / "test_set_histograms.png")
    _print_summary(data)


if __name__ == "__main__":
    main()
