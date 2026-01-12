# sample.py

"""
Sample a trained PhysicsUNet (flow-matching) model on a single device folder.

Expected device folder contents:
  - eps.npy
  - Ez_real.npy
  - Ez_imag.npy
  - grid_meta.npz        (optional)
  - sparams.npz          (must contain 'wavelength_um')
  - src_mask.npy         (optional)

This script:
  - loads eps + metadata wavelength
  - normalizes eps and conditioning using checkpoint stats
  - runs the FM sampler to generate Ez fields
  - saves a visualization comparing GT vs prediction
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage  # type: ignore
except ModuleNotFoundError:
    _scipy_ndimage = None

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, **kwargs: iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from physics_unet import PhysicsUNet
from complex_physics_unet import ComplexPhysicsUNet
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
    gm_path = device_dir / "grid_meta.npz"
    if not gm_path.exists():
        return None
    gm = np.load(gm_path, allow_pickle=True)
    if "dx" in gm and "dy" in gm:
        return float(gm["dx"]), float(gm["dy"])
    return None


def _infer_device_type_from_path(device_dir: Path) -> str:
    p = str(device_dir).lower()
    if "coupler" in p:
        return "coupler"
    if "y_branch" in p or "ybranch" in p:
        return "y_branch"
    if "euler_bend" in p or "eulerbend" in p:
        return "euler_bend"
    return ""


def _build_model_from_ckpt_args(ckpt_args, stats: dict | None, *, device: torch.device):
    dx = float(getattr(ckpt_args, "dx"))
    lam0 = float(getattr(ckpt_args, "lambda_um"))
    omega = 2.0 * np.pi / lam0
    pml_cells = int(getattr(ckpt_args, "pml_cells", 0))

    in_channels = int(getattr(ckpt_args, "in_channels", 0) or 0)
    if (in_channels <= 0) and (stats is not None):
        in_channels = int(stats.get("x_channels", 4))
    if in_channels <= 0:
        include_sdf = bool(getattr(ckpt_args, "include_sdf", False))
        in_channels = 4 + (1 if include_sdf else 0)

    enable_physics = bool(getattr(ckpt_args, "physics_features", True))
    use_complex = bool(getattr(ckpt_args, "complex_unet", False))

    model_kwargs = dict(
        in_channels=in_channels,
        out_channels=2,
        model_channels=int(getattr(ckpt_args, "hidden_size")),
        num_res_blocks=4,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(8, 16),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        num_heads=4,
        cond_dim=int(getattr(ckpt_args, "cond_dim", 1)),
        dx=dx,
        dy=dx,
        omega=omega,
        pml_cells=pml_cells,
        enable_physics_features=enable_physics,
    )

    if use_complex:
        return ComplexPhysicsUNet(**model_kwargs).to(device)
    return PhysicsUNet(**model_kwargs).to(device)


def _sdf_nm_from_eps(eps_phys: np.ndarray, *, dx_um: float, dy_um: float, thr_eps: float = 3.0) -> np.ndarray:
    dx_nm = 1000.0 * float(dx_um)
    dy_nm = 1000.0 * float(dy_um)
    inside = (eps_phys > float(thr_eps))
    if _scipy_ndimage is None:
        raise RuntimeError("SciPy is required to compute SDF during sampling (missing module 'scipy').")
    d_in = _scipy_ndimage.distance_transform_edt(inside, sampling=(dy_nm, dx_nm)).astype(np.float32)
    d_out = _scipy_ndimage.distance_transform_edt(~inside, sampling=(dy_nm, dx_nm)).astype(np.float32)
    return (d_out - d_in).astype(np.float32, copy=False)


def _sdf_feature_from_phi_nm(phi_nm: np.ndarray, *, feature: str, sigma_nm: float) -> np.ndarray:
    feature = str(feature).strip().lower()
    s = float(sigma_nm) if float(sigma_nm) > 0 else 1.0
    if feature == "raw":
        return phi_nm.astype(np.float32, copy=False)[None, ...]
    if feature == "clip":
        return np.clip(phi_nm / s, -1.0, 1.0).astype(np.float32, copy=False)[None, ...]
    if feature == "exp":
        return (np.sign(phi_nm) * np.exp(-np.abs(phi_nm) / s)).astype(np.float32, copy=False)[None, ...]
    if feature == "clip_exp":
        clip = np.clip(phi_nm / s, -1.0, 1.0).astype(np.float32, copy=False)
        exp = (np.sign(phi_nm) * np.exp(-np.abs(phi_nm) / s)).astype(np.float32, copy=False)
        return np.stack([clip, exp], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown sdf_feature: {feature}")


def _build_cond_vector_from_sparams(
    device_dir: Path,
    *,
    stats: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    sp_path = device_dir / "sparams.npz"
    if not sp_path.is_file():
        raise FileNotFoundError(f"Missing {sp_path}")
    sp = np.load(sp_path, allow_pickle=True)

    lam_um = float(sp["wavelength_um"])
    lam_norm = (lam_um - float(stats["lambda_um_mean"])) / float(stats["lambda_um_std"])

    param_names = list(stats.get("cond_param_names", []))
    dev_types = list(stats.get("cond_device_type_names", []))
    dev = _infer_device_type_from_path(device_dir)

    p_vals: list[float] = []
    p_masks: list[float] = []
    for name in param_names:
        if name not in sp:
            p_vals.append(0.0)
            p_masks.append(0.0)
            continue
        v = float(np.array(sp[name]).item())
        if not np.isfinite(v):
            p_vals.append(0.0)
            p_masks.append(0.0)
            continue
        mean = float(stats.get(f"cond_param_{name}_mean", 0.0))
        std = float(stats.get(f"cond_param_{name}_std", 1.0))
        std = std if std > 0 else 1.0
        p_vals.append((v - mean) / std)
        p_masks.append(1.0)

    onehot = [0.0] * len(dev_types)
    for i, tname in enumerate(dev_types):
        if dev == tname:
            onehot[i] = 1.0

    cond_1d = torch.tensor([lam_norm] + p_vals + p_masks + onehot, device=device, dtype=dtype)
    cond = cond_1d[None, :]
    lambda_um_t = torch.tensor([[lam_um]], device=device, dtype=dtype)
    return cond, lambda_um_t, lam_norm


def pml_mask_t(H, W, pml_cells=0, margin=2, device="cpu", dtype=torch.float32):
    p2 = min(pml_cells + margin, max(0, H // 2 - 1), max(0, W // 2 - 1))
    m = torch.ones((1, 1, H, W), device=device, dtype=dtype)
    if p2 > 0:
        m[:, :, :p2, :] = 0
        m[:, :, -p2:, :] = 0
        m[:, :, :, :p2] = 0
        m[:, :, :, -p2:] = 0
    return m


def device_mask_from_eps_t(eps, thr=3.0, dilate=15):
    m = (eps > thr).to(eps.dtype)
    if dilate and dilate > 1:
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate // 2)
    return m


def align_global_phase_t(E_pred, E_true, w, eps=1e-8):
    dot = (w * torch.conj(E_pred) * E_true).sum()
    mag = torch.abs(dot)
    rot = dot / mag.clamp_min(eps)
    return E_pred * rot, rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_dir", type=str)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--use-stoc-samp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--normalize-eps", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--ez-real-out", type=str, default="")

    parser.add_argument("--phase-anchor-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unanchor-pred", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--align-global-phase", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eps-thr", type=float, default=3.0)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--pml-cells", type=int, default=None)   # default: ckpt args
    parser.add_argument("--pml-margin", type=int, default=2)

    args = parser.parse_args()
    num_steps = int(args.num_steps)

    device_dir = Path(args.device_dir).expanduser().resolve()
    if not device_dir.is_dir():
        raise NotADirectoryError(f"device_dir {device_dir} is not a directory")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    repo_root = THIS_DIR.parent
    default_out_dir = repo_root / "outputs"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device_tag = device_dir.name
    out_path = Path(args.out).expanduser().resolve() if args.out else (out_dir / f"{device_tag}_sample_pred.png")
    ez_real_out_path = Path(args.ez_real_out).expanduser().resolve() if args.ez_real_out else (out_dir / f"{device_tag}_sample_ez_real.png")

    timings: dict[str, float] = {}
    t_start_total = time.perf_counter()

    # -----------------------
    # Load device data
    # -----------------------
    t0 = time.perf_counter()
    eps_np = np.load(device_dir / "eps.npy").astype(np.float32)
    ezr_gt_raw = np.load(device_dir / "Ez_real.npy").astype(np.float32)
    ezi_gt_raw = np.load(device_dir / "Ez_imag.npy").astype(np.float32)

    src_path = device_dir / "src_mask.npy"
    if src_path.is_file():
        src_np = np.load(src_path).astype(np.float32)
    else:
        src_np = np.zeros_like(eps_np, dtype=np.float32)

    lam_um = _load_wavelength_um(device_dir)
    grid_dxdy = _load_grid_spacing_um(device_dir)
    timings["io_device_files_s"] = time.perf_counter() - t0

    # -----------------------
    # Load checkpoint
    # -----------------------
    device = torch.device(args.device)
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    timings["io_checkpoint_load_s"] = time.perf_counter() - t0
    stats = ckpt["stats"]
    ckpt_args = ckpt["args"]

    normalize_eps = bool(getattr(ckpt_args, "normalize_eps", True)) if args.normalize_eps is None else bool(args.normalize_eps)
    ckpt_pml_cells = int(getattr(ckpt_args, "pml_cells", 0))
    pml_cells = ckpt_pml_cells if args.pml_cells is None else int(args.pml_cells)

    # Build model
    t0 = time.perf_counter()
    model = _build_model_from_ckpt_args(ckpt_args, stats, device=device)
    model.set_normalization_stats(stats, normalize_eps=normalize_eps)
    model.eval()
    state_key = "ema" if args.use_ema else "model"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint missing key '{state_key}'. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt[state_key], strict=True)
    timings["model_build_and_load_s"] = time.perf_counter() - t0

    # -----------------------
    # Phase-anchor GT (optional) using ckpt-consistent pml_cells
    # -----------------------
    ezr_gt_anch, ezi_gt_anch, phi_gt = phase_anchor_roi(
        ezr_gt_raw,
        ezi_gt_raw,
        eps_r=eps_np,
        pml_cells=pml_cells,
        margin=int(args.pml_margin),
        roi_x=(40, 140),
        thr_eps=float(getattr(ckpt_args, "eps_thr", 3.0)),
    )

    # -----------------------
    # Build conditioning tensors
    # -----------------------
    if normalize_eps:
        eps_norm = (eps_np - float(stats["eps_mean"])) / float(stats["eps_std"])
    else:
        eps_norm = eps_np

    src_norm = (src_np > 0.5).astype(np.float32)

    H, W = eps_np.shape
    eps_t = torch.from_numpy(eps_norm)[None, None].to(device=device, dtype=torch.float32)
    src_t = torch.from_numpy(src_norm)[None, None].to(device=device, dtype=torch.float32)

    # Optional SDF
    x_channels = int(stats.get("x_channels", 4))
    if x_channels > 4:
        if grid_dxdy is not None:
            dx_um, dy_um = float(grid_dxdy[0]), float(grid_dxdy[1])
        else:
            dx_um = float(getattr(ckpt_args, "dx", 1.0 / 24.0))
            dy_um = float(getattr(ckpt_args, "dx", 1.0 / 24.0))

        thr = float(stats.get("sdf_thr_eps", getattr(ckpt_args, "sdf_thr_eps", 3.0)))
        phi_nm = _sdf_nm_from_eps(eps_np, dx_um=dx_um, dy_um=dy_um, thr_eps=thr)
        sdf_feature = str(stats.get("sdf_feature", getattr(ckpt_args, "sdf_feature", "raw")))
        sdf_sigma_nm = float(stats.get("sdf_sigma_nm", getattr(ckpt_args, "sdf_sigma_nm", 100.0)))
        phi_feat = _sdf_feature_from_phi_nm(phi_nm, feature=sdf_feature, sigma_nm=sdf_sigma_nm)

        # Normalize SDF features if the ckpt trained that way
        if bool(getattr(ckpt_args, "normalize_sdf", True)):
            feat = str(sdf_feature).strip().lower()
            C = int(phi_feat.shape[0])
            for c in range(C):
                if feat == "raw":
                    mu = float(stats.get("sdf_nm_mean", 0.0))
                    sd = float(stats.get("sdf_nm_std", 1.0))
                else:
                    mu = float(stats.get(f"sdf_feat{c}_mean", stats.get("sdf_feat_mean", 0.0)))
                    sd = float(stats.get(f"sdf_feat{c}_std", stats.get("sdf_feat_std", 1.0)))
                sd = sd if sd > 0 else 1.0
                phi_feat[c] = (phi_feat[c] - mu) / sd

        phi_t = torch.from_numpy(phi_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
        cond_maps = torch.cat([eps_t, src_t, phi_t], dim=1)
    else:
        cond_maps = torch.cat([eps_t, src_t], dim=1)

    cond, lambda_um_t, lam_norm = _build_cond_vector_from_sparams(
        device_dir, stats=stats, device=device, dtype=torch.float32
    )

    # -----------------------
    # Sample
    # -----------------------
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))
    x0 = torch.randn((1, 2, H, W), device=device, dtype=torch.float32, generator=g)

    def _maybe_sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Gates: if the model was trained with physics features, sampling should typically use them fully.
    # (These gates are internal feature scalars, not loss weights.)
    phys_gate = 1.0
    phase_gate = 1.0

    if args.profile:
        # A minimal profiled loop compatible with your flow_matching.sample calling convention.
        t0 = time.perf_counter()
        fields_norm = fm_sample(
            model,
            x0,
            num_steps=num_steps,
            use_stoc_samp=bool(args.use_stoc_samp),
            cond_maps=cond_maps,
            cond=cond,
            lambda_um=lambda_um_t,
            phys_gate=phys_gate,
            phase_gate=phase_gate,
        )
        _maybe_sync()
        timings["sampling_total_s"] = time.perf_counter() - t0
    else:
        with torch.no_grad():
            t0 = time.perf_counter()
            fields_norm = fm_sample(
                model,
                x0,
                num_steps=num_steps,
                use_stoc_samp=bool(args.use_stoc_samp),
                cond_maps=cond_maps,
                cond=cond,
                lambda_um=lambda_um_t,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
            )
            _maybe_sync()
            timings["sampling_total_s"] = time.perf_counter() - t0

    ezr_pred = fields_norm[0, 0].detach().cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi_pred = fields_norm[0, 1].detach().cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])

    # -----------------------
    # Compare conventions
    # -----------------------
    if args.phase_anchor_gt:
        ezr_gt_cmp, ezi_gt_cmp = ezr_gt_anch, ezi_gt_anch
        gt_suffix = " (anch)"
    else:
        ezr_gt_cmp, ezi_gt_cmp = ezr_gt_raw, ezi_gt_raw
        gt_suffix = ""

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
        eps_cpu = torch.from_numpy(eps_np)[None, None].to(dtype=torch.float32, device="cpu")
        pml_m = pml_mask_t(H, W, pml_cells=pml_cells, margin=int(args.pml_margin), device="cpu")
        dev_m = device_mask_from_eps_t(eps_cpu, thr=float(args.eps_thr), dilate=int(args.dilate))
        m_focus = (pml_m * dev_m)[0, 0]

        E_true = torch.complex(torch.from_numpy(ezr_gt_cmp).float(), torch.from_numpy(ezi_gt_cmp).float())
        E_pred = torch.complex(torch.from_numpy(ezr_pred_cmp).float(), torch.from_numpy(ezi_pred_cmp).float())
        w_align = m_focus * (torch.abs(E_true) ** 2)

        E_pred_aligned, _ = align_global_phase_t(E_pred, E_true, w_align)
        ezr_pred_cmp = E_pred_aligned.real.numpy()
        ezi_pred_cmp = E_pred_aligned.imag.numpy()
        pred_suffix = pred_suffix + " (aligned)"

    err_r = ezr_pred_cmp - ezr_gt_cmp
    err_i = ezi_pred_cmp - ezi_gt_cmp
    mag_gt = np.sqrt(ezr_gt_cmp ** 2 + ezi_gt_cmp ** 2)
    mag_pred = np.sqrt(ezr_pred_cmp ** 2 + ezi_pred_cmp ** 2)
    mag_err = np.sqrt(err_r ** 2 + err_i ** 2)

    # -----------------------
    # Visualize
    # -----------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.reshape(2, 4)

    title_extra = f"λ={lam_um:.4f} µm (cond λ_norm={lam_norm:.4f})"
    if grid_dxdy is not None:
        dxg, dyg = grid_dxdy
        title_extra += f", dx={dxg:.5f} µm"
    fig.suptitle(f"{device_dir.name} | {title_extra}", fontsize=11)

    def im(ax, arr, title, cmap="magma", vmin=None, vmax=None):
        h = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, arr.shape[1]])
        ax.set_yticks([0, arr.shape[0]])
        return h

    im0 = im(axes[0, 0], eps_np, "eps (input)", cmap="viridis")
    im1 = im(axes[0, 1], ezr_gt_cmp, f"Ez_real (GT){gt_suffix}", cmap="RdBu")
    im2 = im(axes[0, 2], ezi_gt_cmp, f"Ez_imag (GT){gt_suffix}", cmap="RdBu")
    im3 = im(axes[0, 3], mag_gt, f"|Ez| (GT){gt_suffix}", cmap="magma")

    im4 = im(axes[1, 0], ezr_pred_cmp, f"Ez_real (pred){pred_suffix}", cmap="RdBu")
    im5 = im(axes[1, 1], ezi_pred_cmp, f"Ez_imag (pred){pred_suffix}", cmap="RdBu")
    im6 = im(axes[1, 2], mag_pred, f"|Ez| (pred){pred_suffix}", cmap="magma")
    im7 = im(axes[1, 3], mag_err, "|err|", cmap="magma")

    for ax in axes.ravel():
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")

    for ax, h in zip(axes.ravel(), [im0, im1, im2, im3, im4, im5, im6, im7]):
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # High-res Ez_real comparison
    extent = (0, W, 0, H)
    x_label, y_label = "x (grid)", "y (grid)"
    if grid_dxdy is not None:
        dxg, dyg = grid_dxdy
        extent = (0, W * dxg, 0, H * dyg)
        x_label, y_label = "x (µm)", "y (µm)"

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, arr, title in zip(axes2, [ezr_gt_cmp, ezr_pred_cmp], ["Ez_real (GT)", "Ez_real (pred)"]):
        im_h = ax.imshow(arr, cmap="RdBu", origin="lower", extent=extent, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig2.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    ez_real_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(ez_real_out_path, dpi=200)
    plt.close(fig2)

    timings["total_s"] = time.perf_counter() - t_start_total

    print(f"[sample.py] device_dir:  {device_dir}")
    print(f"[sample.py] ckpt:        {ckpt_path} ({state_key})")
    print(f"[sample.py] wavelength:   {lam_um:.6f} um  (lambda_norm={lam_norm:+.6f})")
    print(f"[sample.py] normalize_eps={normalize_eps}")
    print(f"[sample.py] pml_cells(ckpt)={ckpt_pml_cells} pml_cells(used)={pml_cells}")
    print(f"[sample.py] phase_anchor_gt={bool(args.phase_anchor_gt)} phi_gt(rad)={phi_gt:+.6f} unanchor_pred={bool(args.unanchor_pred)} align_global_phase={bool(args.align_global_phase)}")
    print(f"[sample.py] saved:       {out_path}")
    print(f"[sample.py] saved:       {ez_real_out_path}")
    if args.profile:
        for k in ["io_device_files_s", "io_checkpoint_load_s", "model_build_and_load_s", "sampling_total_s", "total_s"]:
            if k in timings:
                print(f"[sample.py] timing: {k}={timings[k]:.4f}s")


if __name__ == "__main__":
    main()
