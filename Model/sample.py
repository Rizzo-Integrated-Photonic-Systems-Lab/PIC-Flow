"""
Sample a trained PhysicsUNet (flow-matching) model on a single device folder.

Expected device folder contents (as produced by your FDTD data generation):
  - eps.npy
  - Ez_real.npy
  - Ez_imag.npy
  - grid_meta.npz
  - sparams.npz   (must contain 'wavelength_um')

This script:
  - loads eps + metadata wavelength
  - normalizes eps and wavelength conditioning using checkpoint stats
  - runs the FM sampler to generate Ez fields
  - saves a visualization comparing GT vs prediction
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import torch.nn.functional as F

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    print(
        "[sample.py] ERROR: PyTorch ('torch') is not available in this Python environment.\n"
        "Activate the same environment you used to train the model (where torch is installed),\n"
        "then re-run this script."
    )
    raise

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = lambda iterable, **kwargs: iterable

# Headless plotting (HPC-safe)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure Model/ is on the import path no matter where this is launched from.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from physics_unet import PhysicsUNet
from flow_matching import sample as fm_sample
from dataset import phase_anchor_roi


def _load_wavelength_um(device_dir: Path) -> float:
    sp_path = device_dir / "sparams.npz"
    if not sp_path.exists():
        raise FileNotFoundError(f"Missing {sp_path}")
    sp = np.load(sp_path, allow_pickle=True)
    if "wavelength_um" not in sp:
        raise KeyError(f"{sp_path} missing key 'wavelength_um'. Keys: {list(sp.keys())}")
    return float(sp["wavelength_um"])


def _load_grid_spacing_um(device_dir: Path) -> tuple[float, float] | None:
    """Optional: read dx/dy from grid_meta.npz if present (for reporting only)."""
    gm_path = device_dir / "grid_meta.npz"
    if not gm_path.exists():
        return None
    gm = np.load(gm_path, allow_pickle=True)
    if "dx" in gm and "dy" in gm:
        return float(gm["dx"]), float(gm["dy"])
    return None


def _build_model_from_ckpt_args(ckpt_args, *, device: torch.device) -> PhysicsUNet:
    # Mirrors Model/train.py construction exactly.
    dx = float(getattr(ckpt_args, "dx"))
    lam0 = float(getattr(ckpt_args, "lambda_um"))
    omega = 2.0 * np.pi / lam0

    model = PhysicsUNet(
        in_channels=3,
        out_channels=2,
        model_channels=int(getattr(ckpt_args, "hidden_size")),
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
    return model

def pml_mask_t(H, W, pml_cells=30, margin=2, device="cpu", dtype=torch.float32):
    p2 = min(pml_cells + margin, max(0, H // 2 - 1), max(0, W // 2 - 1))
    m = torch.ones((1, 1, H, W), device=device, dtype=dtype)
    if p2 > 0:
        m[:, :, :p2, :] = 0
        m[:, :, -p2:, :] = 0
        m[:, :, :, :p2] = 0
        m[:, :, :, -p2:] = 0
    return m

def device_mask_from_eps_t(eps, thr=3.0, dilate=15):
    # eps: [1,1,H,W]
    m = (eps > thr).to(eps.dtype)
    if dilate and dilate > 1:
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate // 2)
    return m

def align_global_phase_t(E_pred, E_true, w, eps=1e-8):
    # E_pred/E_true: complex [H,W], w: real [H,W]
    dot = (w * torch.conj(E_pred) * E_true).sum()
    mag = torch.abs(dot)
    rot = dot / mag.clamp_min(eps)
    return E_pred * rot, rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device_dir",
        type=str,
        help="Path to a single device folder containing eps.npy/Ez_real.npy/Ez_imag.npy/grid_meta.npz/sparams.npz",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a training checkpoint .pt (contains ema/model/stats/args).",
    )
    parser.add_argument("--out", type=str, default="", help="Output image path (default: <device_dir>/sample_pred.png)")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of FM integration steps.")
    parser.add_argument(
        "--use-stoc-samp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stochastic refresh at early times (matches train.py flag).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise fields.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, sample with checkpoint['ema']; else use checkpoint['model'].",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, print a timing breakdown (I/O, checkpoint load, sampling, model inference time).",
    )
    parser.add_argument(
        "--normalize-eps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether eps is normalized; default uses checkpoint args.normalize_eps.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory to write outputs (default: <device_dir>/outputs).",
    )
    parser.add_argument(
        "--ez-real-out",
        type=str,
        default="",
        help="Optional path to save a native-aspect Ez_real comparison (GT vs pred).",
    )
    parser.add_argument(
        "--phase-anchor-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If True, apply the same global phase anchoring used in training (see Model/dataset.py) "
            "to the GT fields before computing |err|. This typically makes |err| much smaller when "
            "the only mismatch is a global complex phase."
        ),
    )
    parser.add_argument(
        "--unanchor-pred",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True, rotate the prediction by +phi (phi computed from the raw GT) to match the "
            "raw/unanchored phasor convention before computing |err|. Useful if you want to compare "
            "directly against the raw saved Ez_real/Ez_imag."
        ),
    )
    parser.add_argument(
        "--align-global-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, rotate prediction by a single complex phase to best match GT before computing |err|."
    )
    parser.add_argument("--eps-thr", type=float, default=3.0)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--pml-cells", type=int, default=30)
    parser.add_argument("--pml-margin", type=int, default=2)

    args = parser.parse_args()
    num_steps = int(args.num_steps)

    device_dir = Path(args.device_dir).expanduser().resolve()
    if not device_dir.is_dir():
        raise NotADirectoryError(f"device_dir {device_dir} is not a directory")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Default: write to a repo-top-level outputs/ folder (sibling of Model/, Data/, etc.)
    repo_root = THIS_DIR.parent
    default_out_dir = repo_root / "outputs"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # If explicit file paths are not provided, default into out_dir/
    device_tag = device_dir.name
    out_path = Path(args.out).expanduser().resolve() if args.out else (out_dir / f"{device_tag}_sample_pred.png")
    ez_real_out_path = Path(args.ez_real_out).expanduser().resolve() if args.ez_real_out else (out_dir / f"{device_tag}_sample_ez_real.png")

    timings: dict[str, float] = {}
    t_start_total = time.perf_counter()

    # -----------------------
    # Load device data + metadata
    # -----------------------
    t0 = time.perf_counter()
    eps_np = np.load(device_dir / "eps.npy").astype(np.float32)  # [H,W]
    ezr_gt = np.load(device_dir / "Ez_real.npy").astype(np.float32)
    ezi_gt = np.load(device_dir / "Ez_imag.npy").astype(np.float32)
    lam_um = _load_wavelength_um(device_dir)
    grid_dxdy = _load_grid_spacing_um(device_dir)
    timings["io_device_files_s"] = time.perf_counter() - t0

    # Training uses a per-sample global phase anchor (see Model/dataset.py).
    # Compute the anchored GT + phi here so evaluation can be consistent.
    ezr_gt_anch, ezi_gt_anch, phi_gt = phase_anchor_roi(
        ezr_gt, ezi_gt,
        eps_r=eps_np,
        pml_cells=30,
        margin=2,
        roi_x=(40, 140),
        thr_eps=3.0,
    )

    # -----------------------
    # Load checkpoint (stats + args + weights)
    # -----------------------
    device = torch.device(args.device)
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    timings["io_checkpoint_load_s"] = time.perf_counter() - t0
    stats = ckpt["stats"]
    ckpt_args = ckpt["args"]

    normalize_eps = bool(getattr(ckpt_args, "normalize_eps", True)) if args.normalize_eps is None else bool(args.normalize_eps)

    t0 = time.perf_counter()
    model = _build_model_from_ckpt_args(ckpt_args, device=device)
    model.set_normalization_stats(stats, normalize_eps=normalize_eps)
    model.eval()

    state_key = "ema" if args.use_ema else "model"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint missing key '{state_key}'. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt[state_key], strict=True)
    timings["model_build_and_load_s"] = time.perf_counter() - t0

    # -----------------------
    # Prepare normalized conditioning
    # -----------------------
    if normalize_eps:
        eps_norm = (eps_np - float(stats["eps_mean"])) / float(stats["eps_std"])
    else:
        eps_norm = eps_np

    lam_norm = (lam_um - float(stats["lambda_um_mean"])) / float(stats["lambda_um_std"])

    # torch tensors
    H, W = eps_np.shape
    cond_eps = torch.from_numpy(eps_norm)[None, None, :, :].to(device=device, dtype=torch.float32)  # [1,1,H,W]
    cond = torch.tensor([[lam_norm]], device=device, dtype=torch.float32)  # [1,1]
    lambda_um_t = torch.tensor([[lam_um]], device=device, dtype=torch.float32)  # [1,1] physical

    # -----------------------
    # Sample fields
    # -----------------------
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))
    x0 = torch.randn((1, 2, H, W), device=device, dtype=torch.float32, generator=g)

    # Profiling helper (mirrors flow_matching.sample but records inference/update timing)
    def _maybe_sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def _fm_sample_profiled():
        B = x0.shape[0]
        dtype = x0.dtype

        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        x_new = torch.clone(x0)

        forward_s = 0.0
        update_s = 0.0

        step_range = range(num_steps)
        if args.profile:
            step_range = tqdm(step_range, desc="FM steps", leave=False)
        for k in step_range:
            t0_step = time_steps[k]
            t1_step = time_steps[k + 1]
            t_vec = t0_step.expand(B)

            x_in = torch.cat([x_new, cond_eps], dim=1)  # [B,3,H,W]

            if cond is None:
                net = lambda x: model(x, t_vec, lambda_um=lambda_um_t)
            else:
                net = lambda x: model(x, t_vec, cond=cond, lambda_um=lambda_um_t)

            _maybe_sync()
            tf0 = time.perf_counter()
            v_t = net(x_in)  # [B,2,H,W]
            _maybe_sync()
            forward_s += time.perf_counter() - tf0

            _maybe_sync()
            tu0 = time.perf_counter()
            if (t0_step < 0.2) and bool(args.use_stoc_samp):
                # predictor step (matches flow_matching.sample)
                x_new = x_new + (1.0 - t0_step) * v_t
                noise = torch.randn_like(x_new)
                x_new = (1.0 - t1_step) * noise + t1_step * x_new
            else:
                x_new = x_new + (t1_step - t0_step) * v_t
            _maybe_sync()
            update_s += time.perf_counter() - tu0

        return x_new, forward_s, update_s

    with torch.no_grad():
        t0 = time.perf_counter()
        fields_norm = fm_sample(
            model,
            x0,
            num_steps=num_steps,
            use_stoc_samp=bool(args.use_stoc_samp),
            cond_eps=cond_eps,
            cond=cond,
            lambda_um=lambda_um_t,
        )
        _maybe_sync()
        timings["sampling_total_s"] = time.perf_counter() - t0

    # De-normalize predicted fields to physical units
    ezr_pred = fields_norm[0, 0].cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi_pred = fields_norm[0, 1].cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])

    # -----------------------
    # Visualize
    # -----------------------
    # Choose which convention to compare in:
    # - If --phase-anchor-gt (default): compare against anchored GT (same as training).
    # - Else: compare against raw GT.
    if args.phase_anchor_gt:
        ezr_gt_cmp, ezi_gt_cmp = ezr_gt_anch, ezi_gt_anch
        gt_suffix = " (anch)"
    else:
        ezr_gt_cmp, ezi_gt_cmp = ezr_gt, ezi_gt
        gt_suffix = ""

    # Optionally rotate prediction back into the raw GT phasor convention.
    # Dataset phase_anchor applies a -phi rotation to the raw fields; so to unanchor,
    # multiply by exp(+i*phi).
    if args.unanchor_pred:
        c = float(np.cos(phi_gt))
        s = float(np.sin(phi_gt))
        ezr_pred_cmp = c * ezr_pred - s * ezi_pred
        ezi_pred_cmp = s * ezr_pred + c * ezi_pred
        pred_suffix = " (unanch)"
    else:
        ezr_pred_cmp, ezi_pred_cmp = ezr_pred, ezi_pred
        pred_suffix = ""
    
    if args.align_global_phase:
        # Build masks/weights like training (device region, exclude PML)
        eps_t = torch.from_numpy(eps_np)[None, None].to(dtype=torch.float32, device="cpu")  # [1,1,H,W]
        pml_m = pml_mask_t(H, W, pml_cells=args.pml_cells, margin=args.pml_margin, device="cpu")
        dev_m = device_mask_from_eps_t(eps_t, thr=args.eps_thr, dilate=args.dilate)
        m_focus = (pml_m * dev_m)[0, 0]  # [H,W]

        E_true = torch.complex(
            torch.from_numpy(ezr_gt_cmp).to(torch.float32),
            torch.from_numpy(ezi_gt_cmp).to(torch.float32),
        )
        E_pred = torch.complex(
            torch.from_numpy(ezr_pred_cmp).to(torch.float32),
            torch.from_numpy(ezi_pred_cmp).to(torch.float32),
        )

        w_align = m_focus * (torch.abs(E_true) ** 2)
        E_pred_aligned, rot = align_global_phase_t(E_pred, E_true, w_align)

        ezr_pred_cmp = E_pred_aligned.real.numpy()
        ezi_pred_cmp = E_pred_aligned.imag.numpy()
        pred_suffix = pred_suffix + " (aligned)"

    err_r = ezr_pred_cmp - ezr_gt_cmp
    err_i = ezi_pred_cmp - ezi_gt_cmp
    mag_gt = np.sqrt(ezr_gt_cmp**2 + ezi_gt_cmp**2)
    mag_pred = np.sqrt(ezr_pred_cmp**2 + ezi_pred_cmp**2)
    mag_err = np.sqrt(err_r**2 + err_i**2)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.reshape(2, 4)

    title_extra = f"λ={lam_um:.4f} µm"
    if grid_dxdy is not None:
        dx, dy = grid_dxdy
        title_extra += f", dx={dx:.5f} µm"
    fig.suptitle(f"{device_dir.name} | {title_extra}", fontsize=11)

    def im(ax, arr, title, cmap="magma", vmin=None, vmax=None):
        h = ax.imshow(
            arr,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=(0, arr.shape[1], 0, arr.shape[0]),
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, arr.shape[1]])
        ax.set_yticks([0, arr.shape[0]])
        ax.set_aspect(arr.shape[1] / max(arr.shape[0], 1))
        return h

    # Row 0: inputs + GT
    im0 = im(axes[0, 0], eps_np, "eps (input)", cmap="viridis")
    im1 = im(axes[0, 1], ezr_gt_cmp, f"Ez_real (GT){gt_suffix}")
    im2 = im(axes[0, 2], ezi_gt_cmp, f"Ez_imag (GT){gt_suffix}")
    im3 = im(axes[0, 3], mag_gt, f"|Ez| (GT){gt_suffix}", cmap="magma")

    # Row 1: predictions + errors
    im4 = im(axes[1, 0], ezr_pred_cmp, f"Ez_real (pred){pred_suffix}")
    im5 = im(axes[1, 1], ezi_pred_cmp, f"Ez_imag (pred){pred_suffix}")
    im6 = im(axes[1, 2], mag_pred, f"|Ez| (pred){pred_suffix}", cmap="magma")
    im7 = im(axes[1, 3], mag_err, "|err|", cmap="magma")

    for ax in axes.ravel():
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")

    # colorbars (one per panel to keep it simple/clear)
    for ax, h in zip(axes.ravel(), [im0, im1, im2, im3, im4, im5, im6, im7]):
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # -----------------------
    # Save high-res Ez_real comparison (native aspect)
    # -----------------------
    extent = None
    x_label = "x (grid)"
    y_label = "y (grid)"
    if grid_dxdy is not None:
        dx, dy = grid_dxdy
        extent = (0, W * dx, 0, H * dy)
        x_label = "x (µm)"
        y_label = "y (µm)"
    else:
        extent = (0, W, 0, H)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = ["Ez_real (GT)", "Ez_real (pred)"]
    data = [ezr_gt_cmp, ezr_pred_cmp]

    for ax, arr, title in zip(axes2, data, titles):
        im_h = ax.imshow(
            arr,
            cmap="RdBu",
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig2.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout()
    ez_real_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(ez_real_out_path, dpi=200)
    plt.close(fig2)
    timings["viz_save_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - t_start_total

    print(f"[sample.py] device_dir: {device_dir}")
    print(f"[sample.py] ckpt:       {ckpt_path} ({state_key})")
    print(f"[sample.py] wavelength:  {lam_um} um  (cond normalized={lam_norm:.4f})")
    print(f"[sample.py] normalize_eps={normalize_eps}")
    print(f"[sample.py] phase_anchor_gt={bool(args.phase_anchor_gt)} phi_gt(rad)={phi_gt:+.6f} unanchor_pred={bool(args.unanchor_pred)}")
    print(f"[sample.py] saved:      {out_path}")
    if args.profile:
        n = num_steps
        fwd = timings.get("sampling_forward_s", 0.0)
        if n > 0 and fwd > 0:
            print(f"[sample.py] timing: sampling_forward_s={fwd:.4f}s (avg {1e3*fwd/n:.2f} ms/step, steps={n})")
        upd = timings.get("sampling_update_s", 0.0)
        if n > 0 and upd > 0:
            print(f"[sample.py] timing: sampling_update_s={upd:.4f}s (avg {1e3*upd/n:.2f} ms/step, steps={n})")
        # high-level breakdown
        for k in [
            "io_device_files_s",
            "io_checkpoint_load_s",
            "model_build_and_load_s",
            "sampling_total_s",
            "viz_save_s",
            "total_s",
        ]:
            if k in timings:
                print(f"[sample.py] timing: {k}={timings[k]:.4f}s")


if __name__ == "__main__":
    main()


