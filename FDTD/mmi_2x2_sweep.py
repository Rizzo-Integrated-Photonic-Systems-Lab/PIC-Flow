# mmi_2x2_sweep.py
"""
Sweep simulations for the 2×2 MMI power splitter.

Outputs per sample (under Data/mmi_2x2_sweep/<tag>/):
  - sparams.npz : S11,S21,S31,S41 (or S12,S22,S32,S42) + scalar parameters
  - Ez_real.npy, Ez_imag.npy : fields (if returned)
  - eps.npy : permittivity grid
  - grid_meta.npz : dx, dy, nx, ny, Lx_um, Ly_um
  - (optional) error.txt on failures
"""

import os
import numpy as np
from pathlib import Path
from scipy.stats import qmc
import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm

from mmi_2x2 import MMI2x2PowerSplitter2D


OUT_DIR = Path("../Data/mmi_2x2_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 30  # pixels per µm

# -----------------------------
# Sweep ranges (edit as desired)
# -----------------------------
wg_width_min, wg_width_max = 0.35, 0.55          # µm
lambda_min, lambda_max = 1.40, 1.60              # µm

w_mmi_min, w_mmi_max = 0.80, 1.60                # µm
l_mmi_min, l_mmi_max = 2.0, 8.0                  # µm
gap_min, gap_max = 0.10, 0.40                    # µm

sb_offset_min, sb_offset_max = 0.60, 1.60        # µm
sb_len_min, sb_len_max = 2.0, 7.0                # µm

l_in_min, l_in_max = 1.0, 4.0                    # µm
l_out_min, l_out_max = 1.0, 6.0                  # µm

N_GEO = 1250  # total random geometries

# LHS sampler over 10D:
# [wg_width, lambda, w_mmi, l_mmi, gap, sb_offset, sb_len, l_in, l_out, dummy]
# (dummy is reserved if you later add another parameter; kept for easy extension)
sampler = qmc.LatinHypercube(d=9, seed=42)
u = sampler.random(N_GEO)


def _map_range(uvals, lo, hi):
    return lo + uvals * (hi - lo)


wg_widths = _map_range(u[:, 0], wg_width_min, wg_width_max)
wavelengths = _map_range(u[:, 1], lambda_min, lambda_max)
w_mmis = _map_range(u[:, 2], w_mmi_min, w_mmi_max)
l_mmis = _map_range(u[:, 3], l_mmi_min, l_mmi_max)
gaps = _map_range(u[:, 4], gap_min, gap_max)
sb_offsets = _map_range(u[:, 5], sb_offset_min, sb_offset_max)
sb_lens = _map_range(u[:, 6], sb_len_min, sb_len_max)
l_ins = _map_range(u[:, 7], l_in_min, l_in_max)
l_outs = _map_range(u[:, 8], l_out_min, l_out_max)


def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


# Quantize width and wavelength to 0.01 µm (like your Y-branch sweep)
wg_widths = _quantize_01(wg_widths, wg_width_min, wg_width_max)
wavelengths = _quantize_01(wavelengths, lambda_min, lambda_max)

# Optional: quantize some other dims (comment out if you want fully continuous)
w_mmis = _quantize_01(w_mmis, w_mmi_min, w_mmi_max)
l_mmis = _quantize_01(l_mmis, l_mmi_min, l_mmi_max)
gaps = _quantize_01(gaps, gap_min, gap_max)
sb_offsets = _quantize_01(sb_offsets, sb_offset_min, sb_offset_max)
sb_lens = _quantize_01(sb_lens, sb_len_min, sb_len_max)
l_ins = _quantize_01(l_ins, l_in_min, l_in_max)
l_outs = _quantize_01(l_outs, l_out_min, l_out_max)

param_list = [
    (
        float(wg_w),
        float(lam),
        float(w_mmi),
        float(l_mmi),
        float(gap),
        float(sb_off),
        float(sb_len),
        float(l_in),
        float(l_out),
    )
    for wg_w, lam, w_mmi, l_mmi, gap, sb_off, sb_len, l_in, l_out in zip(
        wg_widths, wavelengths, w_mmis, l_mmis, gaps, sb_offsets, sb_lens, l_ins, l_outs
    )
]


def run_fdtd_sim(
    wg_width: float,
    wavelength: float,
    w_mmi: float,
    l_mmi: float,
    gap: float,
    sb_offset: float,
    sb_len: float,
    l_in: float,
    l_out: float,
    input_port: int = 1,
):
    mmi = MMI2x2PowerSplitter2D(
        wg_width_um=wg_width,
        wavelength_um=wavelength,
        resolution=RESOLUTION,
        w_mmi_um=w_mmi,
        l_mmi_um=l_mmi,
        gap_um=gap,
        s_bend_offset_um=sb_offset,
        s_bend_length_um=sb_len,
        l_input_um=l_in,
        l_output_um=l_out,
        bend_n_segments=120,
    )

    eps, Ez, Hx, Hy, S, cell = mmi.run_sim(input_port=input_port, decay_tol=1e-5)
    return S, Ez, eps, cell


def worker(args):
    (
        wg_width,
        lam,
        w_mmi,
        l_mmi,
        gap,
        sb_off,
        sb_len,
        l_in,
        l_out,
    ) = args

    # pick which input to excite (default: port 1)
    input_port = 1

    tag = (
        f"wgWidth{wg_width:.3f}"
        f"_lam{lam:.2f}"
        f"_wMMI{w_mmi:.2f}"
        f"_lMMI{l_mmi:.2f}"
        f"_gap{gap:.2f}"
        f"_sbOff{sb_off:.2f}"
        f"_sbLen{sb_len:.2f}"
        f"_lin{l_in:.2f}"
        f"_lout{l_out:.2f}"
        f"_in{input_port}"
    )

    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        S, Ez, eps, cell = run_fdtd_sim(
            wg_width, lam, w_mmi, l_mmi, gap, sb_off, sb_len, l_in, l_out, input_port=input_port
        )

        # Convert dict S to ordered vector [S1in, S2in, S3in, S4in]
        if isinstance(S, dict):
            S_vec = np.array([S[(1, input_port)], S[(2, input_port)], S[(3, input_port)], S[(4, input_port)]],
                             dtype=np.complex128)
        else:
            S_vec = np.asarray(S, dtype=np.complex128)

        np.savez(
            out_dir / "sparams.npz",
            wg_width_um=np.float32(wg_width),
            wavelength_um=np.float32(lam),
            w_mmi_um=np.float32(w_mmi),
            l_mmi_um=np.float32(l_mmi),
            gap_um=np.float32(gap),
            s_bend_offset_um=np.float32(sb_off),
            s_bend_length_um=np.float32(sb_len),
            l_input_um=np.float32(l_in),
            l_output_um=np.float32(l_out),
            input_port=np.int32(input_port),
            S_real=S_vec.real.astype(np.float32),
            S_imag=S_vec.imag.astype(np.float32),
            resolution=np.int32(RESOLUTION),
        )

        if Ez is not None:
            Ez = np.asarray(Ez)
            np.save(out_dir / "Ez_real.npy", Ez.real.astype(np.float32))
            np.save(out_dir / "Ez_imag.npy", Ez.imag.astype(np.float32))

        if eps is not None:
            eps = np.asarray(eps)
            np.save(out_dir / "eps.npy", eps.astype(np.float32))

        ny, nx = eps.shape if eps is not None else (0, 0)
        Lx_um, Ly_um = cell
        dx = 1.0 / RESOLUTION
        dy = 1.0 / RESOLUTION
        np.savez(
            out_dir / "grid_meta.npz",
            dx=np.float32(dx),
            dy=np.float32(dy),
            nx=np.int32(nx),
            ny=np.int32(ny),
            Lx_um=np.float32(Lx_um),
            Ly_um=np.float32(Ly_um),
        )

        return tag

    except Exception as e:
        with open(out_dir / "error.txt", "w") as f:
            f.write(repr(e))
        return f"ERROR::{tag}"


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"Total parameter points: {len(param_list)}")
    print(f"Available CPU cores: {cpu_count()}")

    n_procs = min(4, cpu_count())
    print(f"Running {len(param_list)} sims on {n_procs} processes")

    successes = 0
    failures = 0

    with mp.Pool(processes=n_procs) as pool:
        for tag in tqdm(
            pool.imap_unordered(worker, param_list),
            total=len(param_list),
            desc="MMI 2x2 sims",
        ):
            if isinstance(tag, str) and tag.startswith("ERROR::"):
                failures += 1
            else:
                successes += 1

    print(f"Done. Successes: {successes}, Failures: {failures}")
