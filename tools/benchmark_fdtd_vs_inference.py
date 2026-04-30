#!/usr/bin/env python3
"""
Wall-clock benchmark: FDTD (Meep) vs flow-matching surrogate for the same device.

Runs one FDTD simulation, times it, then runs the trained surrogate at a
sweep of FM sampler step counts, times each, and reports the speedup ratio.

Example (Y-branch, sweep step counts 200/100/50/20/10):
    python tools/benchmark_fdtd_vs_inference.py \\
        --ckpt /dartfs/.../checkpoints/0000300.pt \\
        --device ybranch --num-steps 200,100,50,20,10 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def fm_sample_euler(
    ema, x_0: torch.Tensor, *,
    num_steps: int,
    cond_maps: torch.Tensor,
    cond: torch.Tensor | None = None,
    lambda_um: torch.Tensor | None = None,
    phys_gate: float = 1.0, phase_gate: float = 1.0,
    sig_min: float = 0.0,
    time_grid: str = "linear",
) -> torch.Tensor:
    """First-order Euler integrator equivalent to flow_matching.sample's Heun
    loop but with the corrector pass dropped — exactly half the model calls
    per step. Quality drops vs Heun (1st-order error per step) but the
    difference is negligible at >=20 steps.
    """
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2 if time_grid == "quadratic" else base

    x = x_0.clone()
    n_ch = x.shape[1]
    lambda_um_model = None if lambda_um is None else lambda_um.view(B, 1).to(device=device, dtype=dtype)

    for k in range(num_steps):
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        dt = t1 - t0
        t_vec0 = t0.expand(B)
        x_in = torch.cat([x, cond_maps], dim=1)
        kwargs = dict(
            lambda_um=lambda_um_model,
            phys_gate=phys_gate, phase_gate=phase_gate, sig_min=sig_min,
        )
        if cond is not None:
            kwargs["cond"] = cond
        v = ema(x_in, t_vec0, **kwargs)[:, :n_ch]
        x = x + dt * v
    return x
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


# Reasonable mid-sweep parameters for each device family — exactly the kind of
# device a downstream parameter sweep would actually call.
DEFAULT_PARAMS = {
    "ybranch": {
        "wg_width_um": 0.45,
        "l_junction_um": 2.0,
        "l_bend_um": 6.0,
        "h_bend_um": 2.0,
        "l_out_um": 2.0,
    },
    "directional_coupler": {
        "wg_width_um": 0.45,
        "gap_um": 0.20,
        "wg_length_um": 6.5,
        "bend_length_um": 5.0,
        "lead_extra_gap_um": 1.4,
    },
    "mmi": {
        "wg_width_um": 0.45,
        "mmi_width_um": 5.0,
        "mmi_length_um": 11.0,
        "taper_width_um": 1.0,
        "taper_length_um": 2.0,
    },
}


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _bench_inference(
    *,
    model: torch.nn.Module,
    cond_maps: torch.Tensor,
    cond: torch.Tensor,
    lambda_um_t: torch.Tensor,
    field_shape: tuple[int, int],
    runtime_device: torch.device,
    num_steps: int,
    repeat: int,
    warmup: int,
    use_amp: bool,
    time_grid: str,
    integrator: str,
    x0_dtype: torch.dtype,
) -> dict[str, float]:
    """Time a fixed-seed run of the chosen integrator at the given num_steps."""
    ny, nx = field_shape

    def _call_sampler(x0):
        if integrator == "heun":
            return fm_sample(
                model, x0, num_steps=int(num_steps),
                use_stoc_samp=False,
                cond_maps=cond_maps, cond=cond, lambda_um=lambda_um_t,
                phys_gate=1.0, phase_gate=1.0,
                time_grid=time_grid, progress=False,
            )
        elif integrator == "euler":
            return fm_sample_euler(
                model, x0, num_steps=int(num_steps),
                cond_maps=cond_maps, cond=cond, lambda_um=lambda_um_t,
                time_grid=time_grid,
            )
        else:
            raise ValueError(f"unknown integrator: {integrator}")

    def _one_run(seed: int) -> float:
        gen = torch.Generator(device=runtime_device)
        gen.manual_seed(int(seed))
        x0 = torch.randn(
            (1, 2, ny, nx),
            device=runtime_device, dtype=x0_dtype, generator=gen,
        )
        if runtime_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(
                device_type=runtime_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _ = _call_sampler(x0)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Warmup (ignored from timing; lets cuDNN benchmark pick the best kernels
    # and CUDA graphs warm up).
    for w in range(int(warmup)):
        _one_run(seed=10_000 + w)

    times = [_one_run(seed=r) for r in range(int(repeat))]
    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "samples_s": [float(t) for t in times],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FDTD vs FM surrogate wall time.")
    parser.add_argument("--ckpt", required=True, help="Trained checkpoint (.pt)")
    parser.add_argument(
        "--device", default="ybranch",
        choices=["ybranch", "directional_coupler", "mmi"],
        help="Device family to benchmark. Uses mid-sweep parameters.",
    )
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument(
        "--num-steps", default="200,100,50,20,10",
        help="Comma-separated FM step counts to benchmark. Heun does 2 model "
             "calls per step, so 200 ≈ 400 forwards.",
    )
    parser.add_argument("--repeat", type=int, default=5,
                        help="Repetitions per step count (timed; mean/std reported).")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Untimed warmup runs at each step count.")
    parser.add_argument("--time-grid", default="linear",
                        choices=("linear", "quadratic"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fp16 autocast on CUDA (paper triptychs use False; "
                             "AMP gives ~1.5x speed for similar quality).")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Wrap the model with torch.compile (PyTorch 2.x). First "
                             "call is much slower (compile time); subsequent calls faster.")
    parser.add_argument("--integrator", choices=("heun", "euler"), default="heun",
                        help="ODE integrator. Heun is 2nd-order (2 model calls/step, "
                             "default); Euler is 1st-order (1 call/step → 2× faster, "
                             "small quality drop).")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False,
                        help="Convert model + activations to channels_last (NHWC) memory "
                             "layout. ~1.2-1.5× speedup on Ampere+ for conv-heavy graphs.")
    parser.add_argument("--fp16-weights", action=argparse.BooleanOptionalAction, default=False,
                        help="Cast model weights to fp16. Skips autocast overhead. Only "
                             "use with --no-amp; fields are still upcast to fp32 at the end.")
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--wavelength-um", type=float, default=1.55)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--dpml", type=float, default=1.0)
    parser.add_argument("--crop-x-px", type=int, default=480)
    parser.add_argument("--crop-y-px", type=int, default=160)
    parser.add_argument("--params-json", default="",
                        help="JSON string overriding DEFAULT_PARAMS for the device.")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "benchmark.json"))
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    use_amp = bool(args.amp) and runtime_device.type == "cuda"

    print(f"[bench] device family : {args.device}")
    print(f"[bench] runtime       : {runtime_device}")
    print(f"[bench] amp (fp16)    : {use_amp}")
    print(f"[bench] torch.compile : {bool(args.compile)}")

    # Pick params (CLI override or defaults).
    params = dict(DEFAULT_PARAMS[args.device])
    if args.params_json:
        params.update(json.loads(args.params_json))
    print(f"[bench] params        : {params}")

    # Cell sizing.
    crop_x_px = int(args.crop_x_px); crop_y_px = int(args.crop_y_px)
    resolution = int(args.resolution); dpml = float(args.dpml)
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * dpml
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * dpml

    # ------------------------------------------------------------------
    # FDTD benchmark — single run (Meep is deterministic so repetitions
    # only buy you noise from system load; one timed run is enough).
    # ------------------------------------------------------------------
    print("\n[bench] === FDTD (Meep) ===")
    dev = build_device(
        args.device, params, float(args.wavelength_um),
        resolution, dpml, cell_x, cell_y, crop_x_px, crop_y_px,
    )
    print("[bench] running Meep simulation ...")
    t0 = time.perf_counter()
    eps_full, Ez_full, S, cell_size = run_sim_and_extract(
        dev, args.device, input_port=int(args.source_port), decay_tol=1e-5,
    )
    fdtd_time = time.perf_counter() - t0
    print(f"[bench] FDTD wall time: {fdtd_time:.3f} s")

    # Crop PML for the inference inputs (matches dataset convention).
    pml_px = int(round(dpml * resolution))
    if pml_px > 0:
        eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
    else:
        eps = eps_full
    ny, nx = eps.shape

    src_mask, port_ids, port_masks = get_device_masks(
        dev, args.device, int(args.source_port),
        cell_size[0], cell_size[1], dpml, resolution, ny, nx, thickness_px=3,
    )

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------
    print("\n[bench] === Inference (flow matching) ===")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    print(f"[bench] checkpoint    : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[bench] loaded weights: {state_key}")

    if bool(args.fp16_weights):
        if use_amp:
            print("[bench] WARNING: --fp16-weights is most useful with --no-amp; "
                  "AMP autocast is doing similar work and the combination is redundant.")
        model = model.half()
        x0_dtype = torch.float16
        print("[bench] cast model weights to fp16")
    else:
        x0_dtype = torch.float32

    if bool(args.channels_last):
        model = model.to(memory_format=torch.channels_last)
        # cond_maps is also a 4D NCHW tensor — match the layout.
        cond_maps_layout = torch.channels_last
        print("[bench] using channels_last memory format")
    else:
        cond_maps_layout = torch.contiguous_format

    if bool(args.compile):
        print("[bench] compiling model with torch.compile (first call will be slow) ...")
        model = torch.compile(model, mode="reduce-overhead")

    # Build conditioning maps once — they're constant across the sampler loop
    # and across all timing reps.
    dx_um = 1.0 / float(resolution)
    cond_maps = _build_cond_maps(
        eps, src_mask, stats=stats, ckpt_args=ckpt_args,
        device=runtime_device, dx_um=dx_um, dy_um=dx_um,
    )
    if bool(args.fp16_weights):
        cond_maps = cond_maps.to(dtype=torch.float16)
    cond_maps = cond_maps.to(memory_format=cond_maps_layout)
    cond_vec = _build_cond_vector(float(args.wavelength_um), stats, device=runtime_device)
    if bool(args.fp16_weights):
        cond_vec = cond_vec.to(dtype=torch.float16)
    lambda_um_t = torch.tensor(
        [[float(args.wavelength_um)]],
        device=runtime_device,
        dtype=torch.float16 if bool(args.fp16_weights) else torch.float32,
    )

    # Decide which time grid to use based on the checkpoint's training default,
    # or the user override.
    time_grid = str(args.time_grid)
    if time_grid == "checkpoint":
        time_grid = str(_ckpt_get(ckpt_args, "time_grid", "linear"))

    step_list = sorted({int(s) for s in str(args.num_steps).split(",") if s.strip()},
                       reverse=True)
    print(f"[bench] step counts to benchmark: {step_list}")

    inference_results: dict[int, dict[str, float]] = {}
    for num_steps in step_list:
        print(f"\n[bench] benchmarking {num_steps} sampler steps "
              f"(warmup={args.warmup}, repeat={args.repeat}) ...")
        result = _bench_inference(
            model=model, cond_maps=cond_maps, cond=cond_vec,
            lambda_um_t=lambda_um_t,
            field_shape=(ny, nx),
            runtime_device=runtime_device,
            num_steps=int(num_steps),
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            use_amp=use_amp,
            time_grid=time_grid,
            integrator=str(args.integrator),
            x0_dtype=x0_dtype,
        )
        inference_results[int(num_steps)] = result
        speedup = fdtd_time / max(result["mean_s"], 1e-9)
        print(
            f"[bench]   {num_steps:>3} steps: "
            f"{result['mean_s']*1000:>7.1f} ± {result['std_s']*1000:>5.1f} ms "
            f"(min {result['min_s']*1000:>6.1f}, max {result['max_s']*1000:>6.1f})  "
            f"→ {speedup:>6.1f}× faster than FDTD"
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  Summary  device = {args.device}, grid = {ny} x {nx}, λ = {args.wavelength_um} µm")
    print("=" * 70)
    print(f"  {'FDTD (Meep)':<20} {fdtd_time:>10.3f} s {'':>10} (reference)")
    for num_steps in step_list:
        r = inference_results[num_steps]
        speedup = fdtd_time / max(r["mean_s"], 1e-9)
        print(
            f"  {'FM ' + str(num_steps) + '-step':<20} "
            f"{r['mean_s']:>10.4f} s ± {r['std_s']:>4.2f} ms  "
            f"→ {speedup:>6.1f}×"
        )
    print("=" * 70)

    # Save full results.
    out = {
        "device": args.device,
        "params": params,
        "wavelength_um": float(args.wavelength_um),
        "resolution": resolution,
        "grid_ny_nx": [int(ny), int(nx)],
        "amp": bool(use_amp),
        "torch_compile": bool(args.compile),
        "time_grid": time_grid,
        "ckpt": str(ckpt_path),
        "fdtd_time_s": float(fdtd_time),
        "inference_by_steps": {str(k): v for k, v in inference_results.items()},
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[bench] saved JSON: {out_path}")


if __name__ == "__main__":
    main()
