#!/usr/bin/env python3
"""
Predict fields for a parameterized FDTD device without running an FDTD solve.

Examples:
  python tools/predict_parametric_device.py \
    --ckpt /path/to/checkpoints/0000300.pt \
    --device directional_coupler \
    --source-port 1 \
    --params '{"wg_width_um":0.45,"gap_um":0.2,"wg_length_um":6.5,"bend_length_um":5.0,"lead_extra_gap_um":1.2}'

  python tools/predict_parametric_device.py \
    --ckpt /path/to/checkpoints/best_worst_ratio_p95.pt \
    --device mmi \
    --source-port 2 \
    --fill-missing midpoint
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

try:
    from scipy import ndimage as _scipy_ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _scipy_ndimage = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Model"
FDTD_DIR = REPO_ROOT / "FDTD"
for _p in (str(MODEL_DIR), str(FDTD_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flow_matching import sample as fm_sample  # noqa: E402
from physics_unet import PhysicsUNet  # noqa: E402
from complex_physics_unet import ComplexPhysicsUNet  # noqa: E402
from unified_sweep import PARAM_RANGES, build_device, get_device_masks  # noqa: E402


TRAINED_DEVICE_TYPES = ("mmi", "ybranch", "directional_coupler")
SUPPORTED_DEVICE_TYPES = ("mmi", "ybranch", "directional_coupler", "sbend", "crossing", "euler_bend", "straight")
SOURCE_PORTS = {
    "directional_coupler": (1, 2, 3, 4),
    "mmi": (1, 2, 3, 4),
    "ybranch": (1, 2, 3),
    "sbend": (1, 2),
    "crossing": (1, 2, 3, 4),
    "euler_bend": (1, 2),
    "straight": (1, 2),
}
DEFAULT_RESOLUTION = 20
DEFAULT_DPML = 1.0
DEFAULT_CROP_X_PX = 480
DEFAULT_CROP_Y_PX = 160
DEFAULT_WAVELENGTH_UM = 1.55


def _parse_csv_ints(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _ckpt_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_params(args: argparse.Namespace) -> dict[str, float]:
    params: dict[str, Any] = {}
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--params-json must contain a JSON object")
        params.update(loaded)
    if args.params:
        loaded = json.loads(args.params)
        if not isinstance(loaded, dict):
            raise ValueError("--params must be a JSON object")
        params.update(loaded)
    return {str(k): float(v) for k, v in params.items()}


def _validated_params(device_type: str, user_params: dict[str, float], fill_missing: str) -> dict[str, float]:
    if device_type not in PARAM_RANGES:
        raise ValueError(f"Unknown device '{device_type}'. Valid devices: {sorted(PARAM_RANGES)}")
    ranges = PARAM_RANGES[device_type]
    unknown = sorted(set(user_params) - set(ranges))
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for {device_type}: {unknown}. "
            f"Valid parameters: {list(ranges)}"
        )

    params: dict[str, float] = {}
    missing = [name for name in ranges if name not in user_params]
    if missing and fill_missing == "none":
        raise ValueError(
            f"Missing parameter(s) for {device_type}: {missing}. "
            "Pass all parameters or use --fill-missing midpoint."
        )

    for name, (lo, hi) in ranges.items():
        if name in user_params:
            val = float(user_params[name])
        else:
            val = 0.5 * (float(lo) + float(hi))
        if val < float(lo) or val > float(hi):
            raise ValueError(f"{name}={val} is outside trained/sweep range [{lo}, {hi}]")
        params[name] = float(val)
    return params


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _checkpoint_state_dict(ckpt: dict[str, Any], use_ema: bool) -> tuple[str, dict[str, torch.Tensor]]:
    preferred = ("ema", "ema_state_dict") if use_ema else ("model", "model_state_dict")
    fallback = ("model", "model_state_dict") if use_ema else ("ema", "ema_state_dict")
    for key in preferred + fallback:
        state = ckpt.get(key)
        if isinstance(state, dict):
            return key, state
    raise KeyError(f"Checkpoint has no model state dict. Keys: {list(ckpt.keys())}")


def _build_model_from_checkpoint(ckpt: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")

    dx = float(_ckpt_get(ckpt_args, "dx", 0.05))
    lambda_um = float(_ckpt_get(ckpt_args, "lambda_um", DEFAULT_WAVELENGTH_UM))
    omega = 2.0 * math.pi / lambda_um

    in_channels = int(_ckpt_get(ckpt_args, "in_channels", stats.get("x_channels", 4)) or stats.get("x_channels", 4))
    out_channels = 3 if bool(_ckpt_get(ckpt_args, "joint_training", False)) else 2

    channel_mult = _parse_csv_ints(_ckpt_get(ckpt_args, "channel_mult", "1,2,4,8"))
    if not channel_mult:
        channel_mult = (1, 2, 4, 8)
    if bool(_ckpt_get(ckpt_args, "no_attention", False)):
        attn_res = ()
    else:
        attn_res = _parse_csv_ints(_ckpt_get(ckpt_args, "attn_resolutions", ""))

    lambda_sparam = float(_ckpt_get(ckpt_args, "lambda_sparam", 0.0))
    sparam_mode = str(_ckpt_get(ckpt_args, "sparam_mode", "project"))

    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=int(_ckpt_get(ckpt_args, "hidden_size", 56)),
        num_res_blocks=int(_ckpt_get(ckpt_args, "num_res_blocks", 3)),
        channel_mult=channel_mult,
        attention_resolutions=attn_res,
        dropout=float(_ckpt_get(ckpt_args, "model_dropout", _ckpt_get(ckpt_args, "dropout", 0.0))),
        dims=2,
        use_checkpoint=False,
        num_heads=int(_ckpt_get(ckpt_args, "num_heads", 8)),
        cond_dim=int(stats.get("cond_dim", _ckpt_get(ckpt_args, "cond_dim", 1))),
        enable_sparam_head=(lambda_sparam > 0.0 and sparam_mode == "head"),
        dx=dx,
        dy=dx,
        omega=omega,
        pml_cells=int(_ckpt_get(ckpt_args, "pml_cells", 0)),
        enable_physics_features=bool(_ckpt_get(ckpt_args, "physics_features", True)),
    )

    if bool(_ckpt_get(ckpt_args, "complex_unet", False)):
        model = ComplexPhysicsUNet(**kwargs)
    else:
        model = PhysicsUNet(**kwargs)

    normalize_eps = bool(_ckpt_get(ckpt_args, "normalize_eps", True))
    model.set_normalization_stats(stats, normalize_eps=normalize_eps)
    return model.to(device).eval()


def _sdf_nm_from_eps(eps_phys: np.ndarray, *, dx_um: float, dy_um: float, thr_eps: float) -> np.ndarray:
    if _scipy_ndimage is None:
        raise RuntimeError("SciPy is required to compute SDF features, but scipy is not installed.")
    inside = eps_phys > float(thr_eps)
    d_in = _scipy_ndimage.distance_transform_edt(inside, sampling=(dy_um * 1000.0, dx_um * 1000.0))
    d_out = _scipy_ndimage.distance_transform_edt(~inside, sampling=(dy_um * 1000.0, dx_um * 1000.0))
    return (d_out - d_in).astype(np.float32, copy=False)


def _sdf_feature(phi_nm: np.ndarray, *, feature: str, sigma_nm: float) -> np.ndarray:
    feature = str(feature).lower()
    sigma = max(float(sigma_nm), 1e-6)
    if feature == "raw":
        return phi_nm.astype(np.float32, copy=False)[None, ...]
    if feature == "clip":
        return np.clip(phi_nm / sigma, -1.0, 1.0).astype(np.float32, copy=False)[None, ...]
    if feature == "exp":
        return (np.sign(phi_nm) * np.exp(-np.abs(phi_nm) / sigma)).astype(np.float32, copy=False)[None, ...]
    if feature == "clip_exp":
        clip = np.clip(phi_nm / sigma, -1.0, 1.0).astype(np.float32, copy=False)
        exp = (np.sign(phi_nm) * np.exp(-np.abs(phi_nm) / sigma)).astype(np.float32, copy=False)
        return np.stack([clip, exp], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown SDF feature '{feature}'")


def _build_cond_maps(
    eps: np.ndarray,
    src_mask: np.ndarray,
    *,
    stats: dict[str, Any],
    ckpt_args: Any,
    device: torch.device,
    dx_um: float,
    dy_um: float,
) -> torch.Tensor:
    normalize_eps = bool(_ckpt_get(ckpt_args, "normalize_eps", True))
    if normalize_eps:
        eps_norm = (eps - float(stats["eps_mean"])) / float(stats["eps_std"])
    else:
        eps_norm = eps

    chans = [
        torch.from_numpy(eps_norm.astype(np.float32, copy=False))[None, None],
        torch.from_numpy((src_mask > 0).astype(np.float32, copy=False))[None, None],
    ]

    x_channels = int(stats.get("x_channels", 4))
    if x_channels > 4:
        thr = float(stats.get("sdf_thr_eps", _ckpt_get(ckpt_args, "sdf_thr_eps", 3.0)))
        phi_nm = _sdf_nm_from_eps(eps, dx_um=dx_um, dy_um=dy_um, thr_eps=thr)
        feature = str(stats.get("sdf_feature", _ckpt_get(ckpt_args, "sdf_feature", "raw")))
        sigma_nm = float(stats.get("sdf_sigma_nm", _ckpt_get(ckpt_args, "sdf_sigma_nm", 100.0)))
        phi_feat = _sdf_feature(phi_nm, feature=feature, sigma_nm=sigma_nm)

        if bool(_ckpt_get(ckpt_args, "normalize_sdf", True)):
            for c in range(phi_feat.shape[0]):
                if feature == "raw":
                    mu = float(stats.get("sdf_nm_mean", 0.0))
                    sd = float(stats.get("sdf_nm_std", 1.0))
                else:
                    mu = float(stats.get(f"sdf_feat{c}_mean", stats.get("sdf_feat_mean", 0.0)))
                    sd = float(stats.get(f"sdf_feat{c}_std", stats.get("sdf_feat_std", 1.0)))
                phi_feat[c] = (phi_feat[c] - mu) / max(sd, 1e-8)

        chans.append(torch.from_numpy(phi_feat.astype(np.float32, copy=False))[None])

    return torch.cat(chans, dim=1).to(device=device, dtype=torch.float32)


def _build_cond_vector(wavelength_um: float, stats: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    cond_dim = int(stats.get("cond_dim", 1))
    lam_std = float(stats.get("lambda_um_std", 1.0))
    lam_norm = (float(wavelength_um) - float(stats.get("lambda_um_mean", DEFAULT_WAVELENGTH_UM))) / max(lam_std, 1e-8)
    cond = torch.zeros((1, cond_dim), device=device, dtype=torch.float32)
    if cond_dim > 0:
        cond[0, 0] = float(lam_norm)
    return cond


def _get_eps_and_cell(dev: Any, *, crop_pml: bool = True) -> tuple[np.ndarray, tuple[float, float]]:
    if hasattr(dev, "get_eps_and_cell"):
        eps, cell_size = dev.get_eps_and_cell(crop_pml=crop_pml)
        return np.asarray(eps, dtype=np.float32), (float(cell_size[0]), float(cell_size[1]))

    import meep as mp

    sim = mp.Simulation(
        cell_size=dev.cell,
        resolution=int(dev.resolution),
        boundary_layers=[mp.PML(float(dev.dpml))],
        geometry=dev.geometry,
        default_material=dev.clad_medium,
        sources=[],
    )
    sim.init_sim()
    eps_mid = sim.get_epsilon().T
    sim.reset_meep()

    cell_x = float(dev.cell_x)
    cell_y = float(dev.cell_y)
    if not crop_pml:
        return eps_mid.astype(np.float32, copy=False), (cell_x, cell_y)

    p = int(getattr(dev, "pml_px", round(float(dev.dpml) * float(dev.resolution))))
    if p <= 0:
        return eps_mid.astype(np.float32, copy=False), (cell_x, cell_y)
    return (
        eps_mid[p:-p, p:-p].astype(np.float32, copy=False),
        (cell_x - 2.0 * float(dev.dpml), cell_y - 2.0 * float(dev.dpml)),
    )


def _make_device_and_arrays(
    device_type: str,
    params: dict[str, float],
    source_port: int,
    wavelength_um: float,
    resolution: int,
    dpml: float,
    crop_x_px: int,
    crop_y_px: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * float(dpml)
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * float(dpml)
    dev = build_device(device_type, params, wavelength_um, resolution, dpml, cell_x, cell_y, crop_x_px, crop_y_px)
    eps, cell_size = _get_eps_and_cell(dev, crop_pml=True)
    eps = np.asarray(eps, dtype=np.float32)
    ny, nx = eps.shape
    src_mask, port_ids, port_masks = get_device_masks(
        dev,
        device_type,
        int(source_port),
        cell_size[0] + 2.0 * float(dpml),
        cell_size[1] + 2.0 * float(dpml),
        float(dpml),
        int(resolution),
        ny,
        nx,
        thickness_px=3,
    )
    return dev, eps, src_mask, port_ids, port_masks, (float(cell_size[0]), float(cell_size[1]))


def _extent(cell_size: tuple[float, float]) -> tuple[float, float, float, float]:
    sx, sy = cell_size
    return (-0.5 * sx, 0.5 * sx, -0.5 * sy, 0.5 * sy)


def _overlay_masks(ax: Any, src_mask: np.ndarray, port_masks: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    ax.contour(src_mask, levels=[0.5], colors=["white"], linewidths=1.6, origin="lower", extent=extent)
    for i, pm in enumerate(port_masks, start=1):
        ax.contour(pm, levels=[0.5], linewidths=0.8, origin="lower", extent=extent)
        ys, xs = np.where(pm > 0.5)
        if len(xs) > 0:
            x = extent[0] + (float(xs.mean()) + 0.5) * (extent[1] - extent[0]) / pm.shape[1]
            y = extent[2] + (float(ys.mean()) + 0.5) * (extent[3] - extent[2]) / pm.shape[0]
            ax.text(x, y, f"P{i}", color="white", fontsize=8, ha="center", va="center")


def _save_plots(
    eps: np.ndarray,
    src_mask: np.ndarray,
    port_masks: np.ndarray,
    ezr: np.ndarray,
    ezi: np.ndarray,
    *,
    cell_size: tuple[float, float],
    device_type: str,
    params: dict[str, float],
    source_port: int,
    wavelength_um: float,
    out_prefix: Path,
    show: bool,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    ext = _extent(cell_size)
    geom_path = out_prefix.with_suffix(".geometry.png")
    field_path = out_prefix.with_suffix(".prediction.png")

    fig1, ax1 = plt.subplots(figsize=(9.5, 3.6))
    h = ax1.imshow(eps, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    _overlay_masks(ax1, src_mask, port_masks, ext)
    ax1.set_title(f"{device_type} geometry, source port {source_port}, lambda={wavelength_um:.3f} um")
    ax1.set_xlabel("x (um)")
    ax1.set_ylabel("y (um)")
    fig1.colorbar(h, ax=ax1, label="epsilon")
    fig1.tight_layout()
    fig1.savefig(geom_path, dpi=180)

    ez = ezr + 1j * ezi
    mag = np.abs(ez)
    phase = np.angle(ez)
    vmax_re = float(np.percentile(np.abs(np.concatenate([ezr.ravel(), ezi.ravel()])), 99.5))
    vmax_re = vmax_re if vmax_re > 0 else None

    fig2, axes = plt.subplots(2, 2, figsize=(12, 5.8), constrained_layout=True)
    panels = [
        (eps, "Geometry epsilon", "viridis", None, None),
        (ezr, "Predicted Re(Ez)", "RdBu_r", -vmax_re if vmax_re else None, vmax_re),
        (mag, "Predicted |Ez|", "magma", None, None),
        (phase, "Predicted phase(Ez)", "twilight", -math.pi, math.pi),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        if title != "Geometry epsilon":
            ax.contour(eps, levels=[0.5 * (float(eps.min()) + float(eps.max()))], colors=["white"], linewidths=0.5, origin="lower", extent=ext)
        _overlay_masks(ax, src_mask, port_masks, ext)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.suptitle(
        f"{device_type}, source P{source_port}, lambda={wavelength_um:.3f} um | "
        + ", ".join(f"{k}={v:g}" for k, v in params.items()),
        fontsize=10,
    )
    fig2.savefig(field_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    return geom_path, field_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EM fields from parameterized FDTD device geometry.")
    parser.add_argument("--ckpt", required=True, help="Training checkpoint (.pt).")
    parser.add_argument("--device", required=True, choices=sorted(SUPPORTED_DEVICE_TYPES), help="Device family.")
    parser.add_argument("--params", default="", help="JSON object of device parameters.")
    parser.add_argument("--params-json", default="", help="Path to JSON file containing device parameters.")
    parser.add_argument("--fill-missing", choices=("none", "midpoint"), default="none")
    parser.add_argument("--source-port", type=int, required=True)
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-runtime", dest="runtime_device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA weights.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast on CUDA.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress for sampling.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--dpml", type=float, default=DEFAULT_DPML)
    parser.add_argument("--crop-x-px", type=int, default=DEFAULT_CROP_X_PX)
    parser.add_argument("--crop-y-px", type=int, default=DEFAULT_CROP_Y_PX)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "parametric_inference"))
    parser.add_argument("--name", default="", help="Optional output filename prefix.")
    parser.add_argument("--show", action="store_true", help="Open matplotlib windows in addition to saving figures.")
    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    t0 = time.perf_counter()
    device_type = str(args.device)
    if device_type not in TRAINED_DEVICE_TYPES:
        print(
            f"[predict_parametric_device] warning: {device_type} is OOD for this checkpoint; "
            f"trained families are {TRAINED_DEVICE_TYPES}.",
            file=sys.stderr,
        )
    valid_source_ports = SOURCE_PORTS[device_type]
    if int(args.source_port) not in valid_source_ports:
        raise ValueError(f"source-port must be one of {valid_source_ports} for {device_type}")

    params = _validated_params(device_type, _parse_params(args), args.fill_missing)
    runtime_device = _select_device(args.runtime_device)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")

    dev, eps, src_mask, port_ids, port_masks, cell_size = _make_device_and_arrays(
        device_type,
        params,
        int(args.source_port),
        float(args.wavelength_um),
        int(args.resolution),
        float(args.dpml),
        int(args.crop_x_px),
        int(args.crop_y_px),
    )

    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)

    dx_um = 1.0 / float(args.resolution)
    cond_maps = _build_cond_maps(eps, src_mask, stats=stats, ckpt_args=ckpt_args, device=runtime_device, dx_um=dx_um, dy_um=dx_um)
    cond = _build_cond_vector(float(args.wavelength_um), stats, device=runtime_device)
    lambda_um_t = torch.tensor([[float(args.wavelength_um)]], device=runtime_device, dtype=torch.float32)

    gen = torch.Generator(device=runtime_device)
    gen.manual_seed(int(args.seed))
    x0 = torch.randn((1, 2, eps.shape[0], eps.shape[1]), device=runtime_device, dtype=torch.float32, generator=gen)

    time_grid = str(_ckpt_get(ckpt_args, "time_grid", "linear")) if args.time_grid == "checkpoint" else str(args.time_grid)
    use_amp = bool(args.amp and runtime_device.type == "cuda")
    with torch.no_grad():
        with torch.autocast(device_type=runtime_device.type, dtype=torch.float16, enabled=use_amp):
            fields_norm = fm_sample(
                model,
                x0,
                num_steps=int(args.num_steps),
                use_stoc_samp=False,
                cond_maps=cond_maps,
                cond=cond,
                lambda_um=lambda_um_t,
                phys_gate=1.0,
                phase_gate=1.0,
                time_grid=time_grid,
                progress=bool(args.progress),
            )
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)

    fields_norm = fields_norm.float()
    ezr = fields_norm[0, 0].detach().cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fields_norm[0, 1].detach().cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{device_type}_p{int(args.source_port)}_seed{int(args.seed)}"
    out_prefix = out_dir / stem
    npz_path = out_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        eps=eps.astype(np.float32),
        src_mask=src_mask.astype(np.float32),
        port_ids=np.asarray(port_ids, dtype=np.int32),
        port_masks=port_masks.astype(np.float32),
        Ez_real_pred=ezr.astype(np.float32),
        Ez_imag_pred=ezi.astype(np.float32),
        wavelength_um=np.float32(args.wavelength_um),
        source_port=np.int32(args.source_port),
        device=np.array(device_type),
        params_json=np.array(json.dumps(params, sort_keys=True)),
        cell_size_um=np.asarray(cell_size, dtype=np.float32),
    )

    geom_path, field_path = _save_plots(
        eps,
        src_mask,
        port_masks,
        ezr,
        ezi,
        cell_size=cell_size,
        device_type=device_type,
        params=params,
        source_port=int(args.source_port),
        wavelength_um=float(args.wavelength_um),
        out_prefix=out_prefix,
        show=bool(args.show),
    )

    elapsed = time.perf_counter() - t0
    print(f"[predict_parametric_device] ckpt: {ckpt_path}")
    print(f"[predict_parametric_device] weights: {state_key}")
    print(f"[predict_parametric_device] runtime_device: {runtime_device}")
    print(f"[predict_parametric_device] device: {device_type}, source_port={args.source_port}, wavelength_um={args.wavelength_um:g}")
    print(f"[predict_parametric_device] params: {json.dumps(params, sort_keys=True)}")
    print(f"[predict_parametric_device] eps_shape: {eps.shape}, cond_maps_shape={tuple(cond_maps.shape)}, cond_shape={tuple(cond.shape)}")
    print(f"[predict_parametric_device] time_grid={time_grid}, num_steps={args.num_steps}, amp={use_amp}, progress={bool(args.progress)}")
    print(f"[predict_parametric_device] saved: {geom_path}")
    print(f"[predict_parametric_device] saved: {field_path}")
    print(f"[predict_parametric_device] saved: {npz_path}")
    print(f"[predict_parametric_device] elapsed_s={elapsed:.3f}")


if __name__ == "__main__":
    main()
