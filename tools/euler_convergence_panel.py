#!/usr/bin/env python3
"""Visual sampler-convergence figure for the paper.

Runs the FM+phase+residual model on a single held-out directional coupler
across an increasing number of Euler integration steps, holding the random
noise seed fixed so the realization is identical and only step count varies.
Saves a 2x4 panel:

    [ FDTD reference | Euler 5  | Euler 10  | Euler 20 ]
    [ Euler 50       | Euler 100 | Euler 200 | Euler 400 ]

Each model panel is annotated with the per-sample compliance percentage
eps_R (paper Eq. 9) and the GPU wall time. The fields and metrics are also
saved as an .npz so the figure can be re-rendered without re-running.

Usage:
    python tools/euler_convergence_panel.py \
        --ckpt /path/to/0000300.pt \
        --usetex
"""

from __future__ import annotations

import argparse
import json
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

# Step counts to render in order. 8 entries -> 2x4 grid alongside the FDTD ref
# (which goes in the [0, 0] slot, so 7 model panels + 1 FDTD = 8 panels total).
STEP_COUNTS = (5, 10, 20, 50, 100, 200, 400)


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _load_index(data_root: Path) -> list[dict[str, Any]]:
    with open(data_root / "shards" / "index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def _load_shard_sample(data_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    shard_path = data_root / "shards" / str(entry["shard"])
    data = np.load(shard_path, allow_pickle=False)
    p = f"s{int(entry['slot'])}/"
    out: dict[str, Any] = {
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
        "Lx_um": float(_scalar(data, p + "Lx_um", 24.0)),
        "Ly_um": float(_scalar(data, p + "Ly_um", 8.0)),
    }
    params: dict[str, float] = {}
    for key in data.files:
        if key.startswith(p + "params/"):
            params[key.split("/", 2)[-1]] = float(np.asarray(data[key]).item())
    out["params"] = params
    return out


def _pick_test_sample(data_root: Path, device_type: str) -> dict[str, Any]:
    """First test sample for `device_type` with input_port=1.

    Falls back to the first candidate if none have input_port=1.
    """
    candidates = [
        e for e in _load_index(data_root)
        if e.get("device") == device_type
        and e.get("split") == "test"
        and e.get("augment", "orig") == "orig"
    ]
    if not candidates:
        raise RuntimeError(f"No {device_type} test samples in shard index.")
    for e in candidates:
        s = _load_shard_sample(data_root, e)
        if int(s["input_port"]) == 1:
            return s
    return _load_shard_sample(data_root, candidates[0])


def _pick_dc_sample(data_root: Path) -> dict[str, Any]:
    """Backwards-compatible alias for `directional_coupler`."""
    return _pick_test_sample(data_root, "directional_coupler")


def _anchor_reference(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ezr, ezi = sample["ez_real"], sample["ez_imag"]
    eps, src = sample["eps"], sample["src_mask"]
    a, b, _ = phase_anchor_mask(ezr, ezi, src > 0.5, eps_r=eps, thr_eps=3.0)
    if float((src > 0.5).sum()) < 4.0:
        a, b, _ = phase_anchor_roi(ezr, ezi, eps_r=eps, pml_cells=0, margin=2,
                                    roi_x=(40, 140), thr_eps=3.0)
    return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Inference (Euler) at fixed seed, one step count at a time
# ---------------------------------------------------------------------------

def _predict_euler(
    sample: dict[str, Any],
    *,
    ckpt: dict[str, Any],
    model: torch.nn.Module,
    runtime_device: torch.device,
    num_steps: int,
    seed: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Euler N-step on the given sample; return (Ez_real, Ez_imag, wall_s)."""
    stats     = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    eps       = sample["eps"]
    src       = sample["src_mask"]
    wavelength_um = float(sample["wavelength_um"])

    cond_maps = _build_cond_maps(eps, src, stats=stats, ckpt_args=ckpt_args,
                                  device=runtime_device,
                                  dx_um=float(sample["dx_um"]),
                                  dy_um=float(sample["dy_um"]))
    cond      = _build_cond_vector(wavelength_um, stats, device=runtime_device)
    lam_t     = torch.tensor([[wavelength_um]], device=runtime_device, dtype=torch.float32)

    gen = torch.Generator(device=runtime_device); gen.manual_seed(int(seed))
    x0 = torch.randn((1, 2, eps.shape[0], eps.shape[1]),
                     device=runtime_device, dtype=torch.float32, generator=gen)

    use_amp = bool(amp and runtime_device.type == "cuda")
    time_grid = "linear"

    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=runtime_device.type, dtype=torch.float16, enabled=use_amp):
            # NOTE: flow_matching.sample defaults to Heun internally; for this figure
            # we emulate Euler by handing it the legacy use_stoc_samp=False positional
            # and converting from Heun to Euler via the explicit step loop here.
            fields = _euler_sample_explicit(model, x0, cond_maps=cond_maps, cond=cond,
                                            lambda_um=lam_t, num_steps=int(num_steps),
                                            time_grid=time_grid, sig_min=0.0)
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    wall_s = float(time.perf_counter() - t0)

    fn = fields.float()
    ezr = fn[0, 0].cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fn[0, 1].cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    return ezr.astype(np.float32, copy=False), ezi.astype(np.float32, copy=False), wall_s


def _euler_sample_explicit(
    model: torch.nn.Module,
    x_0: torch.Tensor,
    *,
    cond_maps: torch.Tensor,
    cond: torch.Tensor | None,
    lambda_um: torch.Tensor | None,
    num_steps: int,
    time_grid: str,
    sig_min: float,
) -> torch.Tensor:
    """First-order Euler integrator over the FM velocity field.

    Mirrors the Euler sampler used by tools/benchmark_dc_inference_methods.py
    so the timings here line up with Table tab:walltime in the paper.
    """
    device = x_0.device
    dtype  = x_0.dtype
    x = x_0
    if time_grid == "quadratic":
        base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        ts   = base ** 2
    else:
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)

    model_kwargs = dict(lambda_um=lambda_um, phys_gate=1.0, phase_gate=1.0, sig_min=sig_min)
    if cond is not None:
        model_kwargs["cond"] = cond
    for k in range(num_steps):
        t_k = ts[k]
        dt  = ts[k + 1] - t_k
        t_vec = t_k.expand(x.shape[0])
        x_in  = torch.cat([x, cond_maps], dim=1)
        v = model(x_in, t_vec, **model_kwargs)
        x = x + dt * v
    return x


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_matplotlib(usetex: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": bool(usetex),
    })


def _plot_panel(
    ax: Any, mag: np.ndarray, eps: np.ndarray, *,
    extent: tuple[float, float, float, float],
    vmax: float | None,
    title: str,
    annot: str | None,
    show_source: np.ndarray | None = None,
) -> None:
    ax.imshow(mag, origin="lower", cmap="magma", extent=extent,
              vmin=0 if vmax else None, vmax=vmax, aspect="equal",
              interpolation="nearest")
    level = 0.5 * (float(eps.min()) + float(eps.max()))
    ax.contour(eps, levels=[level], colors=["white"], linewidths=0.4,
               origin="lower", extent=extent, alpha=0.75)
    if show_source is not None and np.any(show_source > 0.5):
        overlay = np.ma.masked_where(show_source <= 0.5, np.ones_like(show_source, dtype=np.float32))
        ax.imshow(overlay, origin="lower", extent=extent, cmap="Greens",
                  vmin=0, vmax=1, alpha=0.55, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    if annot is not None:
        ax.text(0.97, 0.93, annot, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.0, color="white",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0))


def _save_figure(
    eps: np.ndarray, src: np.ndarray, extent: tuple[float, float, float, float],
    fdtd_mag: np.ndarray,
    pred_mags: list[np.ndarray],
    metrics: list[dict[str, float]],
    walls_ms: list[float],
    *, out_path: Path, vmax: float,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 2.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # [0,0] = FDTD reference (with translucent green source overlay).
    _plot_panel(axes[0, 0], fdtd_mag, eps, extent=extent, vmax=vmax,
                title=r"FDTD reference",
                annot=r"$\varepsilon_R$ ref", show_source=src)

    # Remaining 7 slots = Euler N-step in row-major order over STEP_COUNTS.
    flat_axes = [axes[0, 1], axes[0, 2], axes[0, 3],
                 axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]
    for ax, n_steps, mag, m, w_ms in zip(flat_axes, STEP_COUNTS, pred_mags, metrics, walls_ms):
        eps_R = m.get("eps_R_pct", float("nan"))
        annot = (rf"$\varepsilon_R={eps_R:.1f}\%$" + "\n" + rf"{w_ms:.0f}\,ms"
                 if plt.rcParams["text.usetex"] else
                 rf"$\varepsilon_R$={eps_R:.1f}%" + "\n" + f"{w_ms:.0f} ms")
        _plot_panel(ax, mag, eps, extent=extent, vmax=vmax,
                    title=f"Euler {n_steps}-step", annot=annot, show_source=None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[euler-conv] saved: {out_path}")
    print(f"[euler-conv] saved: {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render Euler-step convergence panel for the paper.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="FM+phase+residual checkpoint.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "euler_convergence"))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for x0 noise. Held FIXED across step counts so the realization is identical.")
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--usetex", action="store_true")
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir   = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[euler-conv] checkpoint: {args.ckpt}")
    print(f"[euler-conv] data_root: {data_root}")
    print(f"[euler-conv] runtime_device: {runtime_device}")
    print(f"[euler-conv] step_counts: {STEP_COUNTS}")

    sample = _pick_dc_sample(data_root)
    fdtd_r, fdtd_i = _anchor_reference(sample)
    print(f"[euler-conv] geometry_id={sample['geometry_id']}  "
          f"input_port={sample['input_port']}  "
          f"params={sample['params']}")

    ckpt = torch.load(args.ckpt, map_location=runtime_device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True); model.eval()
    print(f"[euler-conv] loaded weights: {state_key}")

    pred_r_list: list[np.ndarray] = []
    pred_i_list: list[np.ndarray] = []
    walls_ms: list[float] = []
    metrics: list[dict[str, float]] = []
    for n_steps in STEP_COUNTS:
        # Warmup once so the first reported time isn't dominated by lazy init.
        if n_steps == STEP_COUNTS[0]:
            _ = _predict_euler(sample, ckpt=ckpt, model=model,
                               runtime_device=runtime_device, num_steps=n_steps,
                               seed=int(args.seed), amp=bool(args.amp))
        ezr, ezi, wall_s = _predict_euler(sample, ckpt=ckpt, model=model,
                                           runtime_device=runtime_device,
                                           num_steps=n_steps,
                                           seed=int(args.seed), amp=bool(args.amp))
        m = residual_metrics(sample["eps"], sample["src_mask"], fdtd_r, fdtd_i, ezr, ezi,
                              dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
                              wavelength_um=float(sample["wavelength_um"]))
        pred_r_list.append(ezr); pred_i_list.append(ezi)
        walls_ms.append(wall_s * 1e3)
        metrics.append(m)
        print(f"[euler-conv]   N={n_steps:>3d}  wall={wall_s*1e3:6.1f} ms  "
              f"eps_R={m['eps_R_pct']:.2f}%")

    # Plot
    _setup_matplotlib(usetex=bool(args.usetex))
    eps   = sample["eps"]
    src   = sample["src_mask"]
    Lx, Ly = float(sample["Lx_um"]), float(sample["Ly_um"])
    extent = (-0.5 * Lx, 0.5 * Lx, -0.5 * Ly, 0.5 * Ly)

    fdtd_mag = np.abs(fdtd_r + 1j * fdtd_i)
    pred_mags = [np.abs(r + 1j * i) for r, i in zip(pred_r_list, pred_i_list)]
    vmax_basis = np.concatenate([fdtd_mag.ravel()] + [m.ravel() for m in pred_mags])
    vmax = float(np.percentile(vmax_basis, 99.5))

    _save_figure(eps, src, extent, fdtd_mag, pred_mags, metrics, walls_ms,
                 out_path=out_dir / "euler_convergence_panel.png", vmax=vmax)

    # Cache fields + metrics for re-rendering.
    np.savez_compressed(
        out_dir / "fields.npz",
        eps=eps.astype(np.float32),
        src_mask=src.astype(np.float32),
        fdtd_Ez_real=fdtd_r.astype(np.float32),
        fdtd_Ez_imag=fdtd_i.astype(np.float32),
        pred_Ez_real=np.stack(pred_r_list, axis=0).astype(np.float32),
        pred_Ez_imag=np.stack(pred_i_list, axis=0).astype(np.float32),
        step_counts=np.asarray(STEP_COUNTS, dtype=np.int32),
        walls_ms=np.asarray(walls_ms, dtype=np.float64),
        eps_R_pct=np.asarray([m["eps_R_pct"] for m in metrics], dtype=np.float64),
        geometry_id=np.array(sample["geometry_id"]),
        params_json=np.array(json.dumps(sample["params"], sort_keys=True)),
    )
    print(f"[euler-conv] cache: {out_dir / 'fields.npz'}")


if __name__ == "__main__":
    main()
