#!/usr/bin/env python3
"""
Generate paper-style inference triptychs from held-out dataset samples.

Each selected held-out sample produces three horizontal panels:
  1. geometry/source-port overlay,
  2. FDTD reference |Ez|,
  3. flow-matching model prediction |Ez|.

The script is intended for in-distribution examples from the current ablation
dataset: MMI, Y-branch, and directional coupler at 1.55 um.
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
from _triptych_metrics import residual_metrics, format_metric_label  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "Data" / "unified_sweep_mmi_ybranch_dc_7500_each_1p55um"
DEFAULT_CKPT = (
    Path("/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm")
    / "REAL_VS_COMPLEX_real_h56_phase_residual_v100x12"
    / "checkpoints"
    / "0000300.pt"
)
DEVICE_ORDER = ("mmi", "ybranch", "directional_coupler")
TRAINED_SOURCE_PORTS = {
    "mmi": (1, 2),
    "ybranch": (1, 2, 3),
    "directional_coupler": (1, 2),
}
DEVICE_LABELS = {
    "mmi": "MMI",
    "ybranch": "Y-branch",
    "directional_coupler": "Directional coupler",
}


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

    sample = {
        "entry": entry,
        "shard_path": shard_path,
        "geometry_id": str(_scalar(data, p + "geometry_id", entry.get("geometry_id", ""))),
        "device": str(_scalar(data, p + "device", entry.get("device", ""))),
        "split": str(_scalar(data, p + "split", entry.get("split", ""))),
        "input_port": int(_scalar(data, p + "input_port")),
        "wavelength_um": float(_scalar(data, p + "wavelength_um")),
        "eps": np.asarray(data[p + "eps"], dtype=np.float32),
        "ez_real": np.asarray(data[p + "Ez_real"], dtype=np.float32),
        "ez_imag": np.asarray(data[p + "Ez_imag"], dtype=np.float32),
        "src_mask": np.asarray(data[p + "src_mask"], dtype=np.float32),
        "port_ids": np.asarray(data[p + "port_ids"], dtype=np.int32),
        "port_masks": np.asarray(data[p + "port_masks"], dtype=np.float32),
        "dx_um": float(_scalar(data, p + "dx_um", 0.05)),
        "dy_um": float(_scalar(data, p + "dy_um", 0.05)),
        "Lx_um": float(_scalar(data, p + "Lx_um", np.asarray(data[p + "eps"]).shape[1] * 0.05)),
        "Ly_um": float(_scalar(data, p + "Ly_um", np.asarray(data[p + "eps"]).shape[0] * 0.05)),
    }

    params: dict[str, float] = {}
    for key in data.files:
        if key.startswith(p + "params/"):
            params[key.split("/", 2)[-1]] = float(np.asarray(data[key]).item())
    sample["params"] = params
    return sample


def _choose_samples(
    data_root: Path,
    *,
    split: str,
    devices: tuple[str, ...],
    prefer_port: int | None,
) -> list[dict[str, Any]]:
    index = _load_index(data_root)
    selected: list[dict[str, Any]] = []
    for device in devices:
        candidates = [
            e for e in index
            if e.get("device") == device and e.get("split") == split and e.get("augment", "orig") == "orig"
        ]
        if not candidates:
            raise RuntimeError(f"No {split} candidates found for {device}")

        loaded: list[dict[str, Any]] = []
        for entry in candidates:
            sample = _load_shard_sample(data_root, entry)
            if sample["input_port"] in TRAINED_SOURCE_PORTS[device]:
                loaded.append(sample)
            if prefer_port is not None and loaded and loaded[-1]["input_port"] == prefer_port:
                break

        if not loaded:
            raise RuntimeError(f"No {split} candidates with trained source ports found for {device}")

        if prefer_port is not None:
            preferred = [s for s in loaded if s["input_port"] == prefer_port]
            selected.append(preferred[0] if preferred else loaded[0])
        else:
            selected.append(loaded[0])
    return selected


def _anchor_reference(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ezr = sample["ez_real"]
    ezi = sample["ez_imag"]
    eps = sample["eps"]
    src = sample["src_mask"]
    ezr2, ezi2, _ = phase_anchor_mask(ezr, ezi, src > 0.5, eps_r=eps, thr_eps=3.0)
    if float((src > 0.5).sum()) < 4.0:
        ezr2, ezi2, _ = phase_anchor_roi(ezr, ezi, eps_r=eps, pml_cells=0, margin=2, roi_x=(40, 140), thr_eps=3.0)
    return ezr2.astype(np.float32, copy=False), ezi2.astype(np.float32, copy=False)


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
    progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    eps = sample["eps"]
    src = sample["src_mask"]
    wavelength_um = float(sample["wavelength_um"])

    cond_maps = _build_cond_maps(
        eps,
        src,
        stats=stats,
        ckpt_args=ckpt_args,
        device=device,
        dx_um=float(sample["dx_um"]),
        dy_um=float(sample["dy_um"]),
    )
    cond = _build_cond_vector(wavelength_um, stats, device=device)
    lambda_um_t = torch.tensor([[wavelength_um]], device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x0 = torch.randn((1, 2, eps.shape[0], eps.shape[1]), device=device, dtype=torch.float32, generator=gen)

    if time_grid == "checkpoint":
        time_grid = str(_ckpt_get(ckpt_args, "time_grid", "linear"))
    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            fields_norm = fm_sample(
                model,
                x0,
                num_steps=int(num_steps),
                use_stoc_samp=False,
                cond_maps=cond_maps,
                cond=cond,
                lambda_um=lambda_um_t,
                phys_gate=1.0,
                phase_gate=1.0,
                time_grid=time_grid,
                progress=progress,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    fields_norm = fields_norm.float()
    ezr = fields_norm[0, 0].detach().cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fields_norm[0, 1].detach().cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    return ezr.astype(np.float32, copy=False), ezi.astype(np.float32, copy=False)


def _extent(sample: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -0.5 * float(sample["Lx_um"]),
        0.5 * float(sample["Lx_um"]),
        -0.5 * float(sample["Ly_um"]),
        0.5 * float(sample["Ly_um"]),
    )


def _setup_matplotlib(show: bool, usetex: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "text.usetex": bool(usetex),
    })


def _overlay_source_mask(ax: Any, sample: dict[str, Any]) -> None:
    ext = _extent(sample)
    src = sample["src_mask"]
    if np.any(src > 0.5):
        source_overlay = np.ma.masked_where(src <= 0.5, np.ones_like(src, dtype=np.float32))
        ax.imshow(
            source_overlay,
            origin="lower",
            extent=ext,
            cmap="Greens",
            vmin=0,
            vmax=1,
            alpha=0.55,
            aspect="equal",
            interpolation="nearest",
        )


def _add_colorbar(fig: Any, im: Any, ax: Any, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, shrink=0.74)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _compute_residual_metrics(
    sample: dict[str, Any],
    fdtd_ezr: np.ndarray, fdtd_ezi: np.ndarray,
    pred_ezr: np.ndarray, pred_ezi: np.ndarray,
) -> dict[str, float]:
    return residual_metrics(
        sample["eps"], sample["src_mask"],
        fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi,
        dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
        wavelength_um=float(sample["wavelength_um"]),
    )


def _annotate_metric(ax: Any, metric: dict[str, float]) -> None:
    ax.text(
        0.97, 0.95, format_metric_label(metric),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0),
    )


def _plot_triptych(
    sample: dict[str, Any],
    fdtd_ezr: np.ndarray,
    fdtd_ezi: np.ndarray,
    pred_ezr: np.ndarray,
    pred_ezi: np.ndarray,
    *,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    ext = _extent(sample)
    eps = sample["eps"]
    fdtd_mag = np.abs(fdtd_ezr + 1j * fdtd_ezi)
    pred_mag = np.abs(pred_ezr + 1j * pred_ezi)
    vmax = float(np.percentile(np.concatenate([fdtd_mag.ravel(), pred_mag.ravel()]), 99.5))
    vmax = vmax if vmax > 0 else None
    metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    panels = [
        (eps, r"Geometry $\varepsilon_r$", "viridis", None, r"$\varepsilon_r$"),
        (fdtd_mag, r"FDTD $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
        (pred_mag, r"Model $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
    ]
    for col, (ax, (arr, title, cmap, vmax_i, cbar_label)) in enumerate(zip(axes, panels)):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmin=0 if vmax_i else None, vmax=vmax_i)
        if title != r"Geometry $\varepsilon_r$":
            level = 0.5 * (float(eps.min()) + float(eps.max()))
            ax.contour(eps, levels=[level], colors=["white"], linewidths=0.45, origin="lower", extent=ext, alpha=0.75)
        _overlay_source_mask(ax, sample)
        ax.set_title(title)
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        _add_colorbar(fig, im, ax, cbar_label)
        if col == 2:
            _annotate_metric(ax, metric)

    dev_label = DEVICE_LABELS.get(sample["device"], sample["device"])
    params_text = ", ".join(f"{k}={v:g}" for k, v in sorted(sample["params"].items()))
    fig.suptitle(
        rf"{dev_label} held-out sample, source $P_{{{sample['input_port']}}}$, "
        rf"$\lambda={sample['wavelength_um']:.2f}\,\mu$m"
        + (f" | {params_text}" if params_text else ""),
        y=1.08,
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_combined(
    rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(11.2, 2.65 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])

    for r, (sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi) in enumerate(rows):
        ext = _extent(sample)
        eps = sample["eps"]
        fdtd_mag = np.abs(fdtd_ezr + 1j * fdtd_ezi)
        pred_mag = np.abs(pred_ezr + 1j * pred_ezi)
        vmax = float(np.percentile(np.concatenate([fdtd_mag.ravel(), pred_mag.ravel()]), 99.5))
        vmax = vmax if vmax > 0 else None
        metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)
        panels = [
            (eps, r"Geometry $\varepsilon_r$", "viridis", None, r"$\varepsilon_r$"),
            (fdtd_mag, r"FDTD $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
            (pred_mag, r"Model $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
        ]
        for c, (arr, title, cmap, vmax_i, cbar_label) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmin=0 if vmax_i else None, vmax=vmax_i)
            if c > 0:
                level = 0.5 * (float(eps.min()) + float(eps.max()))
                ax.contour(eps, levels=[level], colors=["white"], linewidths=0.4, origin="lower", extent=ext, alpha=0.75)
            _overlay_source_mask(ax, sample)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                dev_label = DEVICE_LABELS.get(sample["device"], sample["device"])
                ax.set_ylabel(dev_label + "\n" + r"$y$ ($\mu$m)")
            else:
                ax.set_ylabel(r"$y$ ($\mu$m)")
            ax.set_xlabel(r"$x$ ($\mu$m)")
            _add_colorbar(fig, im, ax, cbar_label)
            if c == 2:
                _annotate_metric(ax, metric)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style held-out inference triptychs.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="FM+phase+residual checkpoint.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset root containing shards/index.json.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "paper_inference_triptychs"))
    parser.add_argument("--devices", default=",".join(DEVICE_ORDER), help="Comma-separated device list.")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--prefer-port", type=int, default=1, help="Prefer this trained source port when available.")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="checkpoint")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device-runtime", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--usetex", action="store_true", help="Use external LaTeX for text rendering.")
    args = parser.parse_args()

    _setup_matplotlib(show=bool(args.show), usetex=bool(args.usetex))

    devices = tuple(d.strip() for d in str(args.devices).split(",") if d.strip())
    for d in devices:
        if d not in DEVICE_ORDER:
            raise ValueError(f"Unsupported in-distribution device '{d}'. Expected subset of {DEVICE_ORDER}.")

    runtime_device = _select_device(args.device_runtime)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    print(f"[triptychs] checkpoint: {ckpt_path}")
    print(f"[triptychs] data_root: {data_root}")
    print(f"[triptychs] runtime_device: {runtime_device}")

    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    print(f"[triptychs] loaded weights: {state_key}")

    samples = _choose_samples(
        data_root,
        split=str(args.split),
        devices=devices,
        prefer_port=int(args.prefer_port) if args.prefer_port > 0 else None,
    )

    combined_rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, sample in enumerate(samples):
        print(
            f"[triptychs] {sample['device']}: {sample['geometry_id']} "
            f"split={sample['split']} source_port={sample['input_port']}"
        )
        fdtd_ezr, fdtd_ezi = _anchor_reference(sample)
        pred_ezr, pred_ezi = _predict_sample(
            sample,
            ckpt=ckpt,
            model=model,
            device=runtime_device,
            num_steps=int(args.num_steps),
            time_grid=str(args.time_grid),
            seed=int(args.seed) + i,
            amp=bool(args.amp),
            progress=bool(args.progress),
        )
        combined_rows.append((sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi))

        out_path = out_dir / f"{sample['device']}_{sample['geometry_id']}_p{sample['input_port']}_triptych.png"
        _plot_triptych(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi, out_path=out_path, show=bool(args.show))
        case_metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)
        np.savez_compressed(
            out_path.with_suffix(".npz"),
            eps=sample["eps"].astype(np.float32),
            src_mask=sample["src_mask"].astype(np.float32),
            port_ids=sample["port_ids"].astype(np.int32),
            port_masks=sample["port_masks"].astype(np.float32),
            fdtd_Ez_real=fdtd_ezr.astype(np.float32),
            fdtd_Ez_imag=fdtd_ezi.astype(np.float32),
            pred_Ez_real=pred_ezr.astype(np.float32),
            pred_Ez_imag=pred_ezi.astype(np.float32),
            geometry_id=np.array(sample["geometry_id"]),
            device=np.array(sample["device"]),
            input_port=np.int32(sample["input_port"]),
            wavelength_um=np.float32(sample["wavelength_um"]),
            params_json=np.array(json.dumps(sample["params"], sort_keys=True)),
            residual_metrics_json=np.array(json.dumps(case_metric, sort_keys=True)),
        )
        print(f"[triptychs] residual_ratio={case_metric.get('ratio', float('nan')):.3f} (pred/FDTD)")
        print(f"[triptychs] saved: {out_path}")
        print(f"[triptychs] saved: {out_path.with_suffix('.pdf')}")

    combined_path = out_dir / "heldout_mmi_ybranch_dc_triptychs.png"
    _plot_combined(combined_rows, out_path=combined_path, show=bool(args.show))
    print(f"[triptychs] saved: {combined_path}")
    print(f"[triptychs] saved: {combined_path.with_suffix('.pdf')}")
    print(f"[triptychs] elapsed_s={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
