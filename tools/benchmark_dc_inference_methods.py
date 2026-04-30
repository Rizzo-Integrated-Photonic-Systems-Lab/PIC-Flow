#!/usr/bin/env python3
"""
Benchmark FDTD vs flow-matching inference for one test-set DC, sweeping
sampler configurations.

For a single directional coupler from the held-out test split:
  1. Run FDTD (Meep) once, timed. Field saved as reference.
  2. Run inference under each configured (integrator, num_steps, amp) combo,
     timed (warmup + repeat), with the predicted field saved.
  3. For each inference method, compute the compliance percentage εR
     (Helmholtz residual relative to driving term, paper Eq. 9) using the
     FDTD field as the residual reference.
  4. Emit:
       outputs/dc_benchmark/summary.png  - grid plot of FDTD + each method's |Ez|
       outputs/dc_benchmark/summary.txt  - human-readable table
       outputs/dc_benchmark/summary.json - full numerics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Early thread-count setup: OMP_NUM_THREADS / MKL / OpenBLAS must be set
# BEFORE meep is imported (which happens transitively via unified_sweep).
# We scan sys.argv ourselves for --meep-threads so the env vars are populated
# before any of the heavy imports below.
# ---------------------------------------------------------------------------
def _early_meep_thread_count() -> int:
    slurm_cpt = os.environ.get("SLURM_CPUS_PER_TASK", "")
    default = int(slurm_cpt) if slurm_cpt.isdigit() else 16
    for i, arg in enumerate(sys.argv):
        if arg == "--meep-threads" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return default
        if arg.startswith("--meep-threads="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
    return default


_meep_threads = _early_meep_thread_count()
# setdefault so an externally-set OMP_NUM_THREADS still wins.
os.environ.setdefault("OMP_NUM_THREADS", str(_meep_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_meep_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_meep_threads))


import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Model"
FDTD_DIR = REPO_ROOT / "FDTD"
TOOLS_DIR = REPO_ROOT / "tools"
for _p in (str(MODEL_DIR), str(FDTD_DIR), str(TOOLS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flow_matching import sample as fm_sample  # noqa: E402
from predict_parametric_device import (  # noqa: E402
    _build_cond_maps,
    _build_cond_vector,
    _build_model_from_checkpoint,
    _checkpoint_state_dict,
    _ckpt_get,
)
from unified_sweep import (  # noqa: E402
    build_device,
    get_device_masks,
    run_sim_and_extract,
)
from dataset import phase_anchor_mask, phase_anchor_roi  # noqa: E402
from _triptych_metrics import residual_metrics  # noqa: E402
from make_paper_inference_triptychs import (  # noqa: E402
    _load_index,
    _load_shard_sample,
)


DEFAULT_CKPT = (
    Path("/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm")
    / "REAL_VS_COMPLEX_real_h56_phase_residual_v100x12"
    / "checkpoints"
    / "0000300.pt"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "Data" / "unified_sweep_mmi_ybranch_dc_7500_each_1p55um"

DEFAULT_RESOLUTION = 20
DEFAULT_DPML = 1.0
DEFAULT_CROP_X_PX = 480
DEFAULT_CROP_Y_PX = 160
DEFAULT_WAVELENGTH_UM = 1.55


# Inference configurations to benchmark. Each is timed and produces one |Ez|
# panel + one row in the summary table.
CONFIGS = [
    {"name": "Heun 200 (no AMP)",  "integrator": "heun",  "num_steps": 200, "amp": False},
    {"name": "Heun 100 (AMP)",     "integrator": "heun",  "num_steps": 100, "amp": True },
    {"name": "Heun 50 (AMP)",      "integrator": "heun",  "num_steps":  50, "amp": True },
    {"name": "Heun 20 (AMP)",      "integrator": "heun",  "num_steps":  20, "amp": True },
    {"name": "Euler 100 (AMP)",    "integrator": "euler", "num_steps": 100, "amp": True },
    {"name": "Euler 50 (AMP)",     "integrator": "euler", "num_steps":  50, "amp": True },
    {"name": "Euler 20 (AMP)",     "integrator": "euler", "num_steps":  20, "amp": True },
]


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _collect_hw_info() -> dict[str, Any]:
    """CPU model name + GPU model name for the JSON / paper provenance."""
    info: dict[str, Any] = {}
    try:
        out = subprocess.check_output(
            ["lscpu"], stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):") and "NUMA" not in line:
                info["cpu_total_logical"] = line.split(":", 1)[1].strip()
            elif line.startswith("Socket(s):"):
                info["cpu_sockets"] = line.split(":", 1)[1].strip()
            elif line.startswith("Core(s) per socket"):
                info["cpu_cores_per_socket"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
        names = [n.strip() for n in out.splitlines() if n.strip()]
        if names:
            info["gpus"] = names
    except Exception:
        pass
    info["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "")
    info["slurm_cpus_per_task"] = os.environ.get("SLURM_CPUS_PER_TASK", "")
    return info


def _pick_test_dc(data_root: Path) -> dict[str, Any]:
    """Pick the first DC test-split sample with input_port=1."""
    index = _load_index(data_root)
    candidates = [
        e for e in index
        if e.get("device") == "directional_coupler"
        and e.get("split") == "test"
        and e.get("augment", "orig") == "orig"
    ]
    if not candidates:
        raise RuntimeError("No directional_coupler test samples in shard index.")
    for entry in candidates:
        sample = _load_shard_sample(data_root, entry)
        if int(sample["input_port"]) == 1:
            return sample
    return _load_shard_sample(data_root, candidates[0])


def fm_sample_euler(
    ema, x_0: torch.Tensor, *,
    num_steps: int,
    cond_maps: torch.Tensor,
    cond: torch.Tensor | None = None,
    lambda_um: torch.Tensor | None = None,
    time_grid: str = "linear",
) -> torch.Tensor:
    """First-order Euler integrator. Half the model calls of Heun per step."""
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2 if time_grid == "quadratic" else base
    x = x_0.clone()
    n_ch = x.shape[1]
    lambda_um_model = (
        lambda_um.view(B, 1).to(device=device, dtype=dtype)
        if lambda_um is not None else None
    )
    for k in range(num_steps):
        t0 = time_steps[k]
        dt = time_steps[k + 1] - time_steps[k]
        t_vec = t0.expand(B)
        x_in = torch.cat([x, cond_maps], dim=1)
        kwargs = dict(lambda_um=lambda_um_model)
        if cond is not None:
            kwargs["cond"] = cond
        v = ema(x_in, t_vec, **kwargs)[:, :n_ch]
        x = x + dt * v
    return x


def _run_one_inference(
    *,
    model, cond_maps, cond, lambda_um_t,
    field_shape, runtime_device,
    integrator: str, num_steps: int, use_amp: bool,
    seed: int = 0,
    channels_last_active: bool = False,
) -> torch.Tensor:
    """Single fixed-seed inference run. Returns the predicted x in normalized space."""
    ny, nx = field_shape
    gen = torch.Generator(device=runtime_device); gen.manual_seed(int(seed))
    x0 = torch.randn(
        (1, 2, ny, nx),
        device=runtime_device, dtype=torch.float32, generator=gen,
    )
    if channels_last_active:
        x0 = x0.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        with torch.autocast(
            device_type=runtime_device.type, dtype=torch.float16, enabled=use_amp,
        ):
            if integrator == "heun":
                x_pred = fm_sample(
                    model, x0, num_steps=int(num_steps),
                    use_stoc_samp=False,
                    cond_maps=cond_maps, cond=cond, lambda_um=lambda_um_t,
                    phys_gate=1.0, phase_gate=1.0,
                    time_grid="linear", progress=False,
                )
            elif integrator == "euler":
                x_pred = fm_sample_euler(
                    model, x0, num_steps=int(num_steps),
                    cond_maps=cond_maps, cond=cond, lambda_um=lambda_um_t,
                    time_grid="linear",
                )
            else:
                raise ValueError(integrator)
    if runtime_device.type == "cuda":
        torch.cuda.synchronize()
    return x_pred


def _time_inference(
    *, runtime_device, repeat, warmup, **call_kwargs,
) -> dict[str, float]:
    """Time a config: warmup runs (untimed) + repeat runs (timed)."""
    for w in range(warmup):
        _ = _run_one_inference(seed=10_000 + w, runtime_device=runtime_device, **call_kwargs)
    times = []
    for r in range(repeat):
        if runtime_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = _run_one_inference(seed=r, runtime_device=runtime_device, **call_kwargs)
        times.append(time.perf_counter() - t0)
    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "samples_s": [float(t) for t in times],
    }


def _denorm_field(x_pred: torch.Tensor, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    x_pred = x_pred.float()
    ezr = (
        x_pred[0, 0].detach().cpu().numpy()
        * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ).astype(np.float32, copy=False)
    ezi = (
        x_pred[0, 1].detach().cpu().numpy()
        * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    ).astype(np.float32, copy=False)
    return ezr, ezi


def _phase_anchor(ezr, ezi, src_mask, eps):
    ezr2, ezi2, _ = phase_anchor_mask(ezr, ezi, src_mask > 0.5, eps_r=eps, thr_eps=3.0)
    if float((src_mask > 0.5).sum()) < 4.0:
        ezr2, ezi2, _ = phase_anchor_roi(
            ezr, ezi, eps_r=eps, pml_cells=0, margin=2,
            roi_x=(40, 140), thr_eps=3.0,
        )
    return ezr2.astype(np.float32, copy=False), ezi2.astype(np.float32, copy=False)


def _save_summary_plot(
    *, fdtd_mag, configs_results, eps, extent, out_path: Path,
    case_label: str,
):
    """Grid plot: FDTD reference + each inference config's |Ez|, all on the
    same color scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 1 + len(configs_results)  # FDTD + N configs
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    # Common color scale (99.5 percentile across all panels).
    all_mags = [fdtd_mag] + [r["pred_mag"] for r in configs_results]
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in all_mags]), 99.5))
    vmax = vmax if vmax > 0 else None

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.4 * ncols, 2.6 * nrows + 0.4),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    eps_level = 0.5 * (float(eps.min()) + float(eps.max()))

    # FDTD panel
    ax = axes_flat[0]
    ax.imshow(fdtd_mag, origin="lower", extent=extent, cmap="magma",
              aspect="equal", vmin=0, vmax=vmax)
    ax.contour(eps, levels=[eps_level], colors=["white"],
               linewidths=0.4, origin="lower", extent=extent, alpha=0.7)
    ax.set_title("FDTD (reference)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Inference panels
    for i, r in enumerate(configs_results, start=1):
        ax = axes_flat[i]
        ax.imshow(r["pred_mag"], origin="lower", extent=extent, cmap="magma",
                  aspect="equal", vmin=0, vmax=vmax)
        ax.contour(eps, levels=[eps_level], colors=["white"],
                   linewidths=0.4, origin="lower", extent=extent, alpha=0.7)
        title = (
            f"{r['name']}\n"
            f"{r['time_ms']:.0f} ms ({r['speedup']:.1f}×)   "
            rf"$\varepsilon_R$={r['eps_R_pct']:.1f}%"
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide any unused panels.
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(case_label, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FDTD vs FM inference methods on one test-set DC."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "dc_benchmark"))
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--repeat", type=int, default=3,
                        help="GPU inference repetitions per config.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="GPU inference warmup runs per config.")
    parser.add_argument(
        "--meep-threads", type=int, default=_meep_threads,
        help=("OpenMP thread count for Meep (FDTD). Default reads "
              "$SLURM_CPUS_PER_TASK if set, else 16. This must be parsed "
              "BEFORE meep is imported — the script does that early-parse "
              "internally; this flag value just records it for the JSON."),
    )
    parser.add_argument(
        "--include-cpu-inference", action="store_true",
        help=("Also benchmark single-thread CPU inference (off by default). "
              "Other photonics-ML papers compare GPU surrogate vs multi-core "
              "CPU FDTD only; CPU inference is rarely reported."),
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Wrap the GPU model with torch.compile. First call adds "
                             "~30s of compile time; subsequent calls 1.5-3x faster. "
                             "Auto-bumps warmup to >=3 (CUDA graphs need extra warmup).")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False,
                        help="Use channels_last (NHWC) memory format on GPU. ~1.2-1.5x "
                             "speedup on Ampere+ (A100/A5500/A6000), smaller on V100.")
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--dpml", type=float, default=DEFAULT_DPML)
    parser.add_argument("--crop-x-px", type=int, default=DEFAULT_CROP_X_PX)
    parser.add_argument("--crop-y-px", type=int, default=DEFAULT_CROP_Y_PX)
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_info = _collect_hw_info()
    print(f"[bench-dc] runtime: {runtime_device}")
    print(f"[bench-dc] CPU: {hw_info.get('cpu_model', '?')} "
          f"({hw_info.get('cpu_sockets', '?')}× sockets × "
          f"{hw_info.get('cpu_cores_per_socket', '?')} cores)")
    if "gpus" in hw_info:
        print(f"[bench-dc] GPU(s): {', '.join(hw_info['gpus'])}")
    print(f"[bench-dc] OMP_NUM_THREADS = {hw_info.get('omp_num_threads', '?')}  "
          f"(--meep-threads = {args.meep_threads})")
    print(f"[bench-dc] data_root: {args.data_root}")

    # ------------------------------------------------------------------
    # Pick a test-split DC and grab its parameters.
    # ------------------------------------------------------------------
    sample = _pick_test_dc(Path(args.data_root))
    params = sample["params"]
    geometry_id = sample["geometry_id"]
    print(f"[bench-dc] test sample: {geometry_id}, source_port={sample['input_port']}")
    print(f"[bench-dc] params: {params}")

    # ------------------------------------------------------------------
    # FDTD: rebuild the same geometry and run Meep, timed.
    # ------------------------------------------------------------------
    crop_x_px = int(args.crop_x_px); crop_y_px = int(args.crop_y_px)
    resolution = int(args.resolution); dpml = float(args.dpml)
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * dpml
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * dpml

    dev = build_device(
        "directional_coupler", params, float(args.wavelength_um),
        resolution, dpml, cell_x, cell_y, crop_x_px, crop_y_px,
    )
    print("[bench-dc] running FDTD ...")
    t0 = time.perf_counter()
    eps_full, Ez_full, S, cell_size = run_sim_and_extract(
        dev, "directional_coupler",
        input_port=int(sample["input_port"]), decay_tol=1e-5,
    )
    fdtd_time = time.perf_counter() - t0
    print(f"[bench-dc] FDTD time: {fdtd_time:.3f} s")

    pml_px = int(round(dpml * resolution))
    eps = eps_full[pml_px:-pml_px, pml_px:-pml_px] if pml_px > 0 else eps_full
    Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px] if pml_px > 0 else Ez_full
    src_mask, port_ids, port_masks = get_device_masks(
        dev, "directional_coupler", int(sample["input_port"]),
        cell_size[0], cell_size[1], dpml, resolution,
        eps.shape[0], eps.shape[1], thickness_px=3,
    )

    # Phase anchor the FDTD field (matches paper triptych protocol).
    fdtd_ezr, fdtd_ezi = _phase_anchor(Ez.real.copy(), Ez.imag.copy(), src_mask, eps)
    fdtd_mag = np.sqrt(fdtd_ezr ** 2 + fdtd_ezi ** 2)

    extent = (
        -0.5 * float(sample["Lx_um"]),
        +0.5 * float(sample["Lx_um"]),
        -0.5 * float(sample["Ly_um"]),
        +0.5 * float(sample["Ly_um"]),
    )

    # ------------------------------------------------------------------
    # Load model + cond_maps once.
    # ------------------------------------------------------------------
    print("[bench-dc] loading model ...")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    stats = ckpt["stats"]; ckpt_args = ckpt.get("args")
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[bench-dc] loaded weights: {state_key}")

    dx_um = 1.0 / float(resolution)
    cond_maps = _build_cond_maps(
        eps, src_mask, stats=stats, ckpt_args=ckpt_args,
        device=runtime_device, dx_um=dx_um, dy_um=dx_um,
    )
    cond_vec = _build_cond_vector(float(args.wavelength_um), stats, device=runtime_device)
    lambda_um_t = torch.tensor(
        [[float(args.wavelength_um)]],
        device=runtime_device, dtype=torch.float32,
    )

    # Apply GPU-only optimization flags (after weights load, before any sampling).
    channels_last_active = bool(args.channels_last) and runtime_device.type == "cuda"
    if channels_last_active:
        model = model.to(memory_format=torch.channels_last)
        cond_maps = cond_maps.contiguous(memory_format=torch.channels_last)
        print("[bench-dc] enabled channels_last memory format on GPU")

    if bool(args.compile) and runtime_device.type == "cuda":
        print("[bench-dc] compiling GPU model with torch.compile (mode='reduce-overhead'); "
              "first warmup call will be slow ...")
        model = torch.compile(model, mode="reduce-overhead")
        # CUDA graph capture inside reduce-overhead mode needs a couple of warmup
        # calls before timing kicks in.
        if int(args.warmup) < 3:
            print(f"[bench-dc]   bumping --warmup from {args.warmup} to 3 for CUDA-graph warmup")
            args.warmup = 3

    # ------------------------------------------------------------------
    # Sweep configs: time + sample one field for plotting + compute εR.
    # ------------------------------------------------------------------
    field_shape = (eps.shape[0], eps.shape[1])
    configs_results = []
    for cfg in CONFIGS:
        print(f"\n[bench-dc] === {cfg['name']} ===")
        # Time it
        timing = _time_inference(
            runtime_device=runtime_device,
            repeat=int(args.repeat), warmup=int(args.warmup),
            model=model, cond_maps=cond_maps, cond=cond_vec, lambda_um_t=lambda_um_t,
            field_shape=field_shape,
            integrator=cfg["integrator"], num_steps=cfg["num_steps"],
            use_amp=bool(cfg["amp"]),
            channels_last_active=channels_last_active,
        )
        # Get one field for plotting + compliance metric (use the warmup seed
        # so all configs see the same noise → comparable fields).
        x_pred = _run_one_inference(
            model=model, cond_maps=cond_maps, cond=cond_vec, lambda_um_t=lambda_um_t,
            field_shape=field_shape, runtime_device=runtime_device,
            integrator=cfg["integrator"], num_steps=cfg["num_steps"],
            use_amp=bool(cfg["amp"]),
            seed=0,
            channels_last_active=channels_last_active,
        )
        ezr_pred, ezi_pred = _denorm_field(x_pred, stats)
        # Phase anchor predicted (so amplitude is in same frame as FDTD).
        ezr_pred, ezi_pred = _phase_anchor(ezr_pred, ezi_pred, src_mask, eps)
        pred_mag = np.sqrt(ezr_pred ** 2 + ezi_pred ** 2)

        # Compliance metric vs FDTD reference.
        rmet = residual_metrics(
            eps, src_mask,
            fdtd_ezr, fdtd_ezi, ezr_pred, ezi_pred,
            dx_um=dx_um, dy_um=dx_um,
            wavelength_um=float(args.wavelength_um),
        )

        speedup = fdtd_time / max(timing["mean_s"], 1e-9)
        result = {
            "name": cfg["name"],
            "integrator": cfg["integrator"],
            "num_steps": int(cfg["num_steps"]),
            "amp": bool(cfg["amp"]),
            "time_s_mean": timing["mean_s"],
            "time_s_std": timing["std_s"],
            "time_s_min": timing["min_s"],
            "time_ms": timing["mean_s"] * 1000.0,
            "speedup": speedup,
            "eps_R_pct": float(rmet["eps_R_pct"]),
            "L_res_pred": float(rmet["L_res_pred"]),
            "L_res_fdtd": float(rmet["L_res_fdtd"]),
            "ratio_R2_pred_over_fdtd": float(rmet["ratio"]),
            "pred_mag": pred_mag,  # kept for plotting; stripped from JSON
        }
        configs_results.append(result)
        print(
            f"[bench-dc]   time = {timing['mean_s']*1000:.1f} ± "
            f"{timing['std_s']*1000:.1f} ms   "
            f"speedup = {speedup:.1f}×   "
            f"εR = {result['eps_R_pct']:.2f}%"
        )

    # ------------------------------------------------------------------
    # Optional CPU inference pass (off by default — the field standard is
    # GPU surrogate vs multi-thread CPU FDTD, no CPU-inference column).
    # ------------------------------------------------------------------
    cpu_inference_run = bool(getattr(args, "include_cpu_inference", False))
    if cpu_inference_run:
        cpu_threads = max(1, int(args.meep_threads))
        print(f"\n[bench-dc] === CPU inference pass ({cpu_threads} thread(s)) ===")
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(cpu_threads)
        except RuntimeError:
            pass

        cpu_device = torch.device("cpu")
        model_cpu = _build_model_from_checkpoint(ckpt, device=cpu_device)
        model_cpu.load_state_dict(state, strict=True)
        model_cpu.eval()

        cond_maps_cpu = cond_maps.detach().to(cpu_device)
        cond_vec_cpu = cond_vec.detach().to(cpu_device)
        lambda_um_cpu = lambda_um_t.detach().to(cpu_device)

        for r in configs_results:
            cfg_name = r["name"]
            integrator = r["integrator"]; num_steps = int(r["num_steps"])
            # Skip the heaviest configs (they take many minutes single-threaded
            # and the multi-threaded number is what matters anyway).
            if integrator == "heun" and num_steps >= 50:
                print(f"[bench-dc] {cfg_name:<22} CPU: skipped (Heun >= 50)")
                r["time_cpu_s_mean"] = None
                r["time_cpu_ms"] = None
                r["speedup_gpu_vs_cpu_infer"] = None
                continue
            print(f"[bench-dc] {cfg_name:<22} CPU: timing ...")
            cpu_timing = _time_inference(
                runtime_device=cpu_device,
                repeat=1, warmup=0,
                model=model_cpu, cond_maps=cond_maps_cpu, cond=cond_vec_cpu,
                lambda_um_t=lambda_um_cpu,
                field_shape=field_shape,
                integrator=integrator, num_steps=num_steps,
                use_amp=False,
            )
            r["time_cpu_s_mean"] = float(cpu_timing["mean_s"])
            r["time_cpu_ms"] = float(cpu_timing["mean_s"] * 1000.0)
            r["speedup_gpu_vs_cpu_infer"] = float(
                cpu_timing["mean_s"] / max(r["time_s_mean"], 1e-9)
            )
            print(
                f"[bench-dc] {cfg_name:<22} CPU: "
                f"{cpu_timing['mean_s']:>8.2f} s   "
                f"(GPU is {r['speedup_gpu_vs_cpu_infer']:.1f}× faster)"
            )
    else:
        for r in configs_results:
            r["time_cpu_s_mean"] = None
            r["time_cpu_ms"] = None
            r["speedup_gpu_vs_cpu_infer"] = None

    # ------------------------------------------------------------------
    # Pretty-print + save table
    # ------------------------------------------------------------------
    case_label = (
        f"DC test sample {geometry_id}  |  "
        f"gap={params.get('gap_um', float('nan')):.3f} µm, "
        f"L_c={params.get('wg_length_um', float('nan')):.3f} µm  |  "
        f"FDTD = {fdtd_time:.2f} s reference"
    )

    def _fmt_time(seconds: float | None) -> str:
        if seconds is None:
            return "skipped"
        if seconds >= 1.0:
            return f"{seconds:>7.2f} s"
        return f"{seconds*1000:>5.1f} ms"

    def _fmt_speedup(s: float | None) -> str:
        if s is None:
            return "—"
        return f"{s:>5.1f}×"

    fdtd_label = f"FDTD (Meep, {int(args.meep_threads)}-thr)"
    name_w = max([len(r["name"]) for r in configs_results] + [len(fdtd_label)])
    sep = "─" * (name_w + 60 + (24 if cpu_inference_run else 0))

    # Header
    if cpu_inference_run:
        header = (
            f"  {'Method':<{name_w}}  "
            f"{'CPU time (1 thr)':>16}  "
            f"{'GPU time':>10}  "
            f"{'GPU vs FDTD':>11}  "
            f"{'GPU vs CPU infer':>16}  "
            f"{'εR':>6}"
        )
    else:
        header = (
            f"  {'Method':<{name_w}}  "
            f"{'GPU time':>10}  "
            f"{'GPU vs FDTD':>11}  "
            f"{'εR':>6}"
        )

    table_lines = [
        case_label,
        sep,
        header,
        sep,
    ]

    fdtd_row = f"  {fdtd_label:<{name_w}}  "
    if cpu_inference_run:
        fdtd_row += (
            f"{_fmt_time(fdtd_time):>16}  "  # FDTD time goes in CPU column
            f"{'—':>10}  "
            f"{'1.0×':>11}  "
            f"{'—':>16}  "
            f"{'(ref)':>6}"
        )
    else:
        fdtd_row += (
            f"{_fmt_time(fdtd_time):>10}  "
            f"{'1.0×':>11}  "
            f"{'(ref)':>6}"
        )
    table_lines.append(fdtd_row)
    table_lines.append(sep)

    for r in configs_results:
        gpu_t_str = _fmt_time(r["time_s_mean"])
        cpu_t_str = _fmt_time(r.get("time_cpu_s_mean"))
        speedup_fdtd_str = _fmt_speedup(r["speedup"])
        speedup_cpu_str = _fmt_speedup(r.get("speedup_gpu_vs_cpu_infer"))
        eps_str = f"{r['eps_R_pct']:>5.1f}%"

        if cpu_inference_run:
            table_lines.append(
                f"  {r['name']:<{name_w}}  "
                f"{cpu_t_str:>16}  "
                f"{gpu_t_str:>10}  "
                f"{speedup_fdtd_str:>11}  "
                f"{speedup_cpu_str:>16}  "
                f"{eps_str:>6}"
            )
        else:
            table_lines.append(
                f"  {r['name']:<{name_w}}  "
                f"{gpu_t_str:>10}  "
                f"{speedup_fdtd_str:>11}  "
                f"{eps_str:>6}"
            )
    table_lines.append(sep)
    table_str = "\n".join(table_lines)
    print("\n" + table_str + "\n")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(table_str + "\n")

    # Save numeric JSON (drop the magnitude arrays).
    json_results = [
        {k: v for k, v in r.items() if k != "pred_mag"}
        for r in configs_results
    ]
    summary = {
        "ckpt": str(ckpt_path),
        "data_root": str(args.data_root),
        "geometry_id": str(geometry_id),
        "input_port": int(sample["input_port"]),
        "params": {k: float(v) for k, v in params.items()},
        "wavelength_um": float(args.wavelength_um),
        "resolution": resolution,
        "fdtd_time_s": float(fdtd_time),
        "fdtd_threads": int(args.meep_threads),
        "amp_default": True,
        "torch_compile": bool(args.compile),
        "channels_last": bool(args.channels_last),
        "hardware": hw_info,
        "results": json_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Plot summary grid.
    _save_summary_plot(
        fdtd_mag=fdtd_mag, configs_results=configs_results,
        eps=eps, extent=extent,
        out_path=out_dir / "summary.png",
        case_label=case_label,
    )

    # Also save the raw field arrays for any post-hoc analysis.
    np.savez_compressed(
        out_dir / "fields.npz",
        eps=eps.astype(np.float32),
        src_mask=src_mask.astype(np.float32),
        fdtd_Ez_real=fdtd_ezr.astype(np.float32),
        fdtd_Ez_imag=fdtd_ezi.astype(np.float32),
        **{
            f"pred_{r['name'].replace(' ', '_').replace('(', '').replace(')', '')}": r["pred_mag"].astype(np.float32)
            for r in configs_results
        },
    )

    print(f"[bench-dc] saved: {out_dir / 'summary.png'}")
    print(f"[bench-dc] saved: {out_dir / 'summary.txt'}")
    print(f"[bench-dc] saved: {out_dir / 'summary.json'}")
    print(f"[bench-dc] saved: {out_dir / 'fields.npz'}")


if __name__ == "__main__":
    main()
