#!/usr/bin/env python3
"""Denoising trajectory: animate + still-panel for a single device.

Runs a single 100-step Euler integration on one held-out directional coupler
and captures the field state $x_t$ at every integrator step. Produces:

  - outputs/denoising/trajectory.gif        : animated |x_t| over t in [0, 1]
  - outputs/denoising/trajectory_panel.png  : static frames at step indices
                                              {0, 1, 5, 10, 25, 50, 100} for
                                              the paper
  - outputs/denoising/fields.npz            : cached fields + metadata so the
                                              gif/panel can be re-rendered

The single long integration captures the full noise->field trajectory; the
existing tools/euler_convergence_panel.py answers a different question (what
do *several* short integrations of varying step count converge to).

Usage:
    python tools/denoising_trajectory.py --usetex
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
from predict_parametric_device import (  # noqa: E402
    _build_cond_maps,
    _build_cond_vector,
    _build_model_from_checkpoint,
    _checkpoint_state_dict,
    _ckpt_get,
)
from euler_convergence_panel import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_DATA_ROOT,
    _pick_dc_sample,
    _anchor_reference,
    _select_device,
)


# t-values to render in the static panel. At K=400 these become step indices
# (0, 4, 20, 80, 200, 300, 400). Labeling by t rather than k keeps the panel
# meaningful regardless of integrator step count.
PANEL_T_VALUES = (0.00, 0.01, 0.05, 0.20, 0.50, 0.75, 1.00)
TOTAL_STEPS = 400


# ---------------------------------------------------------------------------
# Inference: capture every intermediate x_t along an Euler trajectory
# ---------------------------------------------------------------------------

def _euler_trajectory(
    model: torch.nn.Module,
    *,
    eps_shape: tuple[int, int],
    cond_maps: torch.Tensor,
    cond: torch.Tensor | None,
    lambda_um: torch.Tensor | None,
    num_steps: int,
    seed: int,
    runtime_device: torch.device,
    amp: bool,
) -> np.ndarray:
    """Run num_steps Euler. Return a stack of (num_steps + 1) field tensors,
    one per integration step (including the initial noise at t=0)."""
    gen = torch.Generator(device=runtime_device); gen.manual_seed(int(seed))
    x = torch.randn((1, 2, eps_shape[0], eps_shape[1]),
                    device=runtime_device, dtype=torch.float32, generator=gen)

    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=runtime_device, dtype=x.dtype)
    use_amp = bool(amp and runtime_device.type == "cuda")
    model_kwargs = dict(lambda_um=lambda_um, phys_gate=1.0, phase_gate=1.0, sig_min=0.0)
    if cond is not None:
        model_kwargs["cond"] = cond

    states: list[np.ndarray] = [x[0].detach().cpu().numpy().copy()]
    with torch.no_grad():
        with torch.autocast(device_type=runtime_device.type, dtype=torch.float16, enabled=use_amp):
            for k in range(num_steps):
                t_k = ts[k]; dt = ts[k + 1] - t_k
                x_in = torch.cat([x, cond_maps], dim=1)
                v = model(x_in, t_k.expand(x.shape[0]), **model_kwargs)
                x = x + dt * v
                states.append(x[0].detach().cpu().float().numpy().copy())
    return np.stack(states, axis=0)  # [num_steps + 1, 2, H, W]


def _denormalize(states_norm: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply (real, imag) channel-wise std + mean. Returns (Re, Im) arrays."""
    er_mean = float(stats["ez_real_mean"]); er_std = float(stats["ez_real_std"])
    ei_mean = float(stats["ez_imag_mean"]); ei_std = float(stats["ez_imag_std"])
    ezr = states_norm[:, 0, :, :] * er_std + er_mean
    ezi = states_norm[:, 1, :, :] * ei_std + ei_mean
    return ezr.astype(np.float32, copy=False), ezi.astype(np.float32, copy=False)


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


def _render_panel_frame(ax, mag, eps, *, vmax, title, source=None, show_contour=True):
    ax.imshow(mag, origin="lower", cmap="magma",
              vmin=0 if vmax else None, vmax=vmax,
              aspect="equal", interpolation="nearest")
    if show_contour:
        level = 0.5 * (float(eps.min()) + float(eps.max()))
        ax.contour(eps, levels=[level], colors=["white"], linewidths=0.4, alpha=0.75)
    if source is not None and np.any(source > 0.5):
        overlay = np.ma.masked_where(source <= 0.5, np.ones_like(source, dtype=np.float32))
        ax.imshow(overlay, origin="lower", cmap="Greens", vmin=0, vmax=1,
                  alpha=0.55, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])


def _save_static_panel(
    eps: np.ndarray, src: np.ndarray,
    fdtd_mag: np.ndarray,
    traj_mags: np.ndarray,            # [num_steps + 1, H, W]
    t_values: tuple[float, ...],
    num_steps: int,
    *, out_path: Path, vmax_global: float,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 2.6), constrained_layout=True)
    flat = axes.flatten()

    for ax, t_val in zip(flat[: len(t_values)], t_values):
        k = int(round(float(t_val) * num_steps))
        k = max(0, min(num_steps, k))
        mag = traj_mags[k]
        if t_val <= 1e-9:
            title = r"$t=0$ (noise)"
        elif abs(t_val - 1.0) < 1e-9:
            title = r"$t=1$ (final)"
        else:
            title = rf"$t={t_val:g}$"
        _render_panel_frame(ax, mag, eps, vmax=vmax_global, title=title,
                            source=(src if t_val <= 1e-9 else None))
    _render_panel_frame(flat[len(t_values)], fdtd_mag, eps,
                        vmax=vmax_global, title="FDTD reference",
                        source=src)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[denoise] saved: {out_path}")
    print(f"[denoise] saved: {out_path.with_suffix('.pdf')}")


def _save_gif(
    eps: np.ndarray, src: np.ndarray,
    traj_mags: np.ndarray,
    fdtd_mag: np.ndarray,
    *, out_path: Path, fps: int = 12, hold_last_frames: int = 12,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 1.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Per-frame normalization on the trajectory panel so structure is visible.
    # Fixed FDTD-scale on the right panel for amplitude reference.
    vmax_fdtd = float(np.percentile(fdtd_mag, 99.5))

    im_traj = axes[0].imshow(traj_mags[0], origin="lower", cmap="magma",
                              aspect="equal", interpolation="nearest")
    level = 0.5 * (float(eps.min()) + float(eps.max()))
    axes[0].contour(eps, levels=[level], colors=["white"], linewidths=0.4, alpha=0.75)
    if np.any(src > 0.5):
        axes[0].imshow(np.ma.masked_where(src <= 0.5, np.ones_like(src, dtype=np.float32)),
                       origin="lower", cmap="Greens", vmin=0, vmax=1, alpha=0.55,
                       aspect="equal", interpolation="nearest")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    title_traj = axes[0].set_title("step 0 / noise")

    axes[1].imshow(fdtd_mag, origin="lower", cmap="magma",
                   vmin=0, vmax=vmax_fdtd, aspect="equal", interpolation="nearest")
    axes[1].contour(eps, levels=[level], colors=["white"], linewidths=0.4, alpha=0.75)
    if np.any(src > 0.5):
        axes[1].imshow(np.ma.masked_where(src <= 0.5, np.ones_like(src, dtype=np.float32)),
                       origin="lower", cmap="Greens", vmin=0, vmax=1, alpha=0.55,
                       aspect="equal", interpolation="nearest")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("FDTD reference")

    n_traj = traj_mags.shape[0]
    total_frames = n_traj + int(hold_last_frames)

    def _update(i: int):
        k = min(i, n_traj - 1)
        m = traj_mags[k]
        v = float(np.percentile(m, 99.5))
        v = v if v > 0 else 1.0
        im_traj.set_data(m); im_traj.set_clim(vmin=0, vmax=v)
        t_val = k / max(n_traj - 1, 1)
        if k == 0:
            title_traj.set_text(r"$t = 0.00$ (noise)")
        elif k == n_traj - 1:
            title_traj.set_text(r"$t = 1.00$ (final)")
        else:
            title_traj.set_text(rf"$t = {t_val:.2f}$")
        return [im_traj, title_traj]

    anim = animation.FuncAnimation(fig, _update, frames=total_frames,
                                    interval=int(1000 / max(fps, 1)), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=int(fps))
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"[denoise] saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and render the FM denoising trajectory.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "denoising"))
    parser.add_argument("--num-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--usetex", action="store_true")
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[denoise] checkpoint: {args.ckpt}")
    print(f"[denoise] data_root: {data_root}")
    print(f"[denoise] runtime_device: {runtime_device}")
    print(f"[denoise] num_steps: {args.num_steps}  seed: {args.seed}")

    sample = _pick_dc_sample(data_root)
    fdtd_r, fdtd_i = _anchor_reference(sample)
    print(f"[denoise] geometry_id={sample['geometry_id']}  input_port={sample['input_port']}")

    ckpt = torch.load(args.ckpt, map_location=runtime_device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True); model.eval()
    print(f"[denoise] loaded weights: {state_key}")

    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    cond_maps = _build_cond_maps(sample["eps"], sample["src_mask"], stats=stats,
                                  ckpt_args=ckpt_args, device=runtime_device,
                                  dx_um=float(sample["dx_um"]),
                                  dy_um=float(sample["dy_um"]))
    cond  = _build_cond_vector(float(sample["wavelength_um"]), stats, device=runtime_device)
    lam_t = torch.tensor([[float(sample["wavelength_um"])]], device=runtime_device, dtype=torch.float32)

    t0 = time.perf_counter()
    states_norm = _euler_trajectory(model,
                                     eps_shape=sample["eps"].shape,
                                     cond_maps=cond_maps, cond=cond, lambda_um=lam_t,
                                     num_steps=int(args.num_steps),
                                     seed=int(args.seed), runtime_device=runtime_device,
                                     amp=bool(args.amp))
    print(f"[denoise] trajectory captured in {time.perf_counter() - t0:.2f} s")

    ezr_traj, ezi_traj = _denormalize(states_norm, stats)
    traj_mags = np.abs(ezr_traj + 1j * ezi_traj)         # [num_steps + 1, H, W]
    fdtd_mag  = np.abs(fdtd_r + 1j * fdtd_i)
    vmax_global = float(np.percentile(np.concatenate([traj_mags[-1].ravel(), fdtd_mag.ravel()]), 99.5))

    _setup_matplotlib(usetex=bool(args.usetex))
    _save_static_panel(sample["eps"], sample["src_mask"], fdtd_mag, traj_mags,
                       t_values=PANEL_T_VALUES,
                       num_steps=int(args.num_steps),
                       out_path=out_dir / "trajectory_panel.png",
                       vmax_global=vmax_global)
    _save_gif(sample["eps"], sample["src_mask"], traj_mags, fdtd_mag,
              out_path=out_dir / "trajectory.gif",
              fps=int(args.gif_fps))

    np.savez_compressed(
        out_dir / "fields.npz",
        eps=sample["eps"].astype(np.float32),
        src_mask=sample["src_mask"].astype(np.float32),
        fdtd_Ez_real=fdtd_r.astype(np.float32),
        fdtd_Ez_imag=fdtd_i.astype(np.float32),
        traj_Ez_real=ezr_traj.astype(np.float32),
        traj_Ez_imag=ezi_traj.astype(np.float32),
        num_steps=np.int32(args.num_steps),
        panel_t_values=np.asarray(PANEL_T_VALUES, dtype=np.float32),
        geometry_id=np.array(sample["geometry_id"]),
        params_json=np.array(json.dumps(sample["params"], sort_keys=True)),
    )
    print(f"[denoise] cache: {out_dir / 'fields.npz'}")


if __name__ == "__main__":
    main()
