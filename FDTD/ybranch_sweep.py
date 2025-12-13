# ybranch_sweep.py
"""
Sweep simulations for the 1×2 Y-branch splitter.

Outputs per sample (under Data/y_branch_sweep/<tag>/):
  - sparams.npz : S11, S21, S31 and scalar parameters
  - Ez_real.npy, Ez_imag.npy : fields (if returned)
  - eps.npy : permittivity grid
  - grid_meta.npz : dx, dy, nx, ny, Lx_um, Ly_um
"""

import numpy as np
from pathlib import Path
from scipy.stats import qmc
import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm
import os

from ybranch import YBranch2D


OUT_DIR = Path("../Data/y_branch_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 30  # pixels per µm

# Sweep ranges
wg_width_min, wg_width_max = 0.38, 0.60          # µm
lambda_min, lambda_max = 1.40, 1.60              # µm
l_j_min, l_j_max = 1.5, 3.5                      # µm
l_bend_min, l_bend_max = 5.0, 8.0                # µm
h_bend_min, h_bend_max = 0.5, 1.2                # µm
l_out_min, l_out_max = 1.0, 15.0                  # µm

N_GEO = 1250  # total random geometries

# LHS sampler over 6D: [wg_width, lambda, l_j, l_bend, h_bend, l_out]
sampler = qmc.LatinHypercube(d=6, seed=42)
u = sampler.random(N_GEO)


def _map_range(uvals, lo, hi):
    return lo + uvals * (hi - lo)


wg_widths = _map_range(u[:, 0], wg_width_min, wg_width_max)
wavelengths = _map_range(u[:, 1], lambda_min, lambda_max)
l_js = _map_range(u[:, 2], l_j_min, l_j_max)
l_bends = _map_range(u[:, 3], l_bend_min, l_bend_max)
h_bends = _map_range(u[:, 4], h_bend_min, h_bend_max)
l_outs = _map_range(u[:, 5], l_out_min, l_out_max)


def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


# Quantize width and wavelength to 0.01 µm
wg_widths = _quantize_01(wg_widths, wg_width_min, wg_width_max)
wavelengths = _quantize_01(wavelengths, lambda_min, lambda_max)

param_list = [
    (
        float(w),
        float(lam),
        float(l_j),
        float(l_bend),
        float(h_bend),
        float(l_out),
    )
    for w, lam, l_j, l_bend, h_bend, l_out in zip(
        wg_widths, wavelengths, l_js, l_bends, h_bends, l_outs
    )
]


def run_fdtd_sim(
    wg_width: float,
    wavelength: float,
    l_j: float,
    l_bend: float,
    h_bend: float,
    l_out: float,
):
    yb = YBranch2D(
        wg_width_um=wg_width,
        wavelength_um=wavelength,
        resolution=RESOLUTION,
        l_junction_um=l_j,
        l_bend_um=l_bend,
        h_bend_um=h_bend,
        l_out_um=l_out,
        bend_n_segments=96,
    )

    eps, Ez, Hx, Hy, S, cell = yb.run_sim(input_port=1, decay_tol=1e-5)
    return S, Ez, eps, cell


def worker(args):
    wg_width, lam, l_j, l_bend, h_bend, l_out = args

    tag = (
        f"wgWidth{wg_width:.3f}"
        f"_lam{lam:.2f}"
        f"_lj{l_j:.2f}"
        f"_lb{l_bend:.2f}"
        f"_hb{h_bend:.2f}"
        f"_lout{l_out:.2f}"
    )
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        S, Ez, eps, cell = run_fdtd_sim(wg_width, lam, l_j, l_bend, h_bend, l_out)

        if isinstance(S, dict):
            S_vec = np.array([S[(1, 1)], S[(2, 1)], S[(3, 1)]], dtype=np.complex128)
        else:
            S_vec = np.asarray(S, dtype=np.complex128)

        np.savez(
            out_dir / "sparams.npz",
            wg_width_um=np.float32(wg_width),
            wavelength_um=np.float32(lam),
            l_junction_um=np.float32(l_j),
            l_bend_um=np.float32(l_bend),
            h_bend_um=np.float32(h_bend),
            l_out_um=np.float32(l_out),
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
        grid_meta = {
            "dx": np.float32(dx),
            "dy": np.float32(dy),
            "nx": np.int32(nx),
            "ny": np.int32(ny),
            "Lx_um": np.float32(Lx_um),
            "Ly_um": np.float32(Ly_um),
        }
        np.savez(out_dir / "grid_meta.npz", **grid_meta)

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
            desc="Y-branch sims",
        ):
            if isinstance(tag, str) and tag.startswith("ERROR::"):
                failures += 1
            else:
                successes += 1

    print(f"Done. Successes: {successes}, Failures: {failures}")

