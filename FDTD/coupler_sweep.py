# coupler_sweep.py
import numpy as np
from pathlib import Path
from scipy.stats import qmc

from directional_coupler import DirectionalCoupler2D

from conditioning_masks import make_source_mask

import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm
import os


OUT_DIR = Path("../Data/coupler_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 30  # Pixels per micron

# Geometry and wavelength ranges
gap_min, gap_max = 0.1, 0.35              # µm
wg_length_min, wg_length_max = 5.0, 15.0   # µm
bend_length_min, bend_length_max = 5.0, 7.0  # µm
wg_width_min, wg_width_max = 0.38, 0.60    # µm
lambda_min, lambda_max = 1.40, 1.60        # µm
lead_gap_min, lead_gap_max = 1.0, 2.5      # µm

N_GEO = 10000

# LHS sampler in 6D: [gap, coupler length, bend length, wg_width, wavelength, lead_extra_gap]
sampler = qmc.LatinHypercube(d=6, seed=42)
u = sampler.random(N_GEO)  # shape (N_GEO, 6), values in [0, 1]

# Map unit cube -> physical ranges
gaps = gap_min + u[:, 0] * (gap_max - gap_min)
wg_lengths = wg_length_min + u[:, 1] * (wg_length_max - wg_length_min)
bend_lengths = bend_length_min + u[:, 2] * (bend_length_max - bend_length_min)
wg_widths = wg_width_min + u[:, 3] * (wg_width_max - wg_width_min)
wavelengths = lambda_min + u[:, 4] * (lambda_max - lambda_min)
lead_gaps = lead_gap_min + u[:, 5] * (lead_gap_max - lead_gap_min)

# Quantize all continuous parameters to 0.01 µm steps for a regularized grid
def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)

# Quantize only wg_width and wavelength to 0.01 µm steps
# gaps, wg_lengths, bend_lengths, lead_gaps remain continuous
wg_widths = _quantize_01(wg_widths, wg_width_min, wg_width_max)
wavelengths = _quantize_01(wavelengths, lambda_min, lambda_max)


# excitation sampling
p_port1 = 0.5
p_port2 = 0.5


rng = np.random.default_rng(42)
exc_ports = rng.choice([1, 2], size=N_GEO, p=[p_port1, p_port2])

param_list = [
    # (wg_width, gap, coupler_length, bend_length, lead_extra_gap, wavelength)
    (float(w), float(g), float(L), float(b), float(lead), float(lam), int(p))
    for w, g, L, b, lead, lam, p in zip(
        wg_widths, gaps, wg_lengths, bend_lengths, lead_gaps, wavelengths, exc_ports
    )
]


def run_fdtd_sim(
    wg_width: float,
    gap: float,
    wg_length: float,
    bend_length: float,
    lead_extra_gap: float,
    wl: float,
    input_port: int,
):
    dc = DirectionalCoupler2D(
        wg_width_um=wg_width,
        gap_um=gap,
        wg_length_um=wg_length,
        wavelength_um=wl,
        resolution=RESOLUTION,
        dpml=1,
        pad_y_um=1.0,
        lead_extra_gap_um=lead_extra_gap,  # sweep IO lead separation
        bend_length_um=bend_length,        # length of each S-bend section (clamped to fit cell)
        bend_n_segments=64,                # smoother S-bend
    )

    if input_port == 0:
        eps, cell = dc.get_eps_and_cell()    # eps is [ny,nx]
        ny, nx = eps.shape
        Lx_um, Ly_um = cell

        src_mask = make_source_mask(
            input_port=0,
            port_centers_um=dc.get_port_centers_um(),
            y_span_um=dc.get_port_y_span_um(),
            Lx_um=Lx_um,
            Ly_um=Ly_um,
            ny=ny,
            nx=nx,
        )
        return None, None, eps, cell, src_mask

    eps, Ez, Hx, Hy, S, cell = dc.run_sim(input_port=input_port, decay_tol=1e-5)
    ny, nx = eps.shape
    Lx_um, Ly_um = cell

    src_mask = make_source_mask(
        input_port=input_port,
        port_centers_um=dc.get_port_centers_um(),
        y_span_um=dc.get_port_y_span_um(),
        Lx_um=Lx_um,
        Ly_um=Ly_um,
        ny=ny,
        nx=nx,
    )
    return S, Ez, eps, cell, src_mask


def worker(args):
    wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, input_port = args

    # Include width + input port in the tag so different runs don't collide
    tag = (
        f"wgWidth{wg_width:.3f}"
        f"_gap{gap:.3f}"
        f"_wgLength{wg_length:.1f}"
        f"_bendLength{bend_length:.1f}"
        f"_leadGap{lead_extra_gap:.2f}"
        f"_lam{lam:.2f}"
        f"_inputPort{input_port}"
    )
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        S, Ez, eps, cell, src_mask = run_fdtd_sim(
            wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, input_port
        )

        # ---- S-params (handle dict + handle "no excitation") ----
        S_vec = None
        if S is not None:
            if isinstance(S, dict):
                # Vector is [S11, S21, S31, S41] but with the *actual* excited port
                # i.e. S(out_port, input_port) for out_port in {1,2,3,4}
                S_vec = np.array(
                    [S[(1, input_port)], S[(2, input_port)], S[(3, input_port)], S[(4, input_port)]],
                    dtype=np.complex128,
                )
            else:
                S_vec = np.asarray(S, dtype=np.complex128)

        # 1) Save S-params + scalar metadata
        # If input_port == 0 (or S is None), we store a source_off flag and omit S arrays.
        if S_vec is not None:
            np.savez(
                out_dir / "sparams.npz",
                input_port=np.int32(input_port),
                source_off=np.int32(0),
                gap_um=np.float32(gap),
                Lc_um=np.float32(wg_length),
                wg_width_um=np.float32(wg_width),
                lead_extra_gap_um=np.float32(lead_extra_gap),
                wavelength_um=np.float32(lam),
                S_real=S_vec.real.astype(np.float32),
                S_imag=S_vec.imag.astype(np.float32),
                resolution=np.int32(RESOLUTION),
            )
        else:
            np.savez(
                out_dir / "sparams.npz",
                input_port=np.int32(input_port),
                source_off=np.int32(1),
                gap_um=np.float32(gap),
                Lc_um=np.float32(wg_length),
                wg_width_um=np.float32(wg_width),
                lead_extra_gap_um=np.float32(lead_extra_gap),
                wavelength_um=np.float32(lam),
                resolution=np.int32(RESOLUTION),
            )

        # 2) Save fields (if returned)
        if Ez is not None:
            Ez = np.asarray(Ez)
            np.save(out_dir / "Ez_real.npy", Ez.real.astype(np.float32))
            np.save(out_dir / "Ez_imag.npy", Ez.imag.astype(np.float32))

        # 3) Save permittivity grid
        if eps is not None:
            eps = np.asarray(eps)
            np.save(out_dir / "eps.npy", eps.astype(np.float32))

        # 4) Save grid metadata from actual array shape
        ny, nx = eps.shape if eps is not None else src_mask.shape
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

        # 5) Save source mask
        np.save(out_dir / "src_mask.npy", np.asarray(src_mask, dtype=np.float32))

        return tag

    except Exception as e:
        with open(out_dir / "error.txt", "w") as f:
            f.write(repr(e))
        return f"ERROR::{tag}"




if __name__ == "__main__":
    # Limit threading inside each worker to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"Total parameter points: {len(param_list)}")
    print(f"Available CPU cores: {cpu_count()}")

    # Adjust n_procs as desired (<= cpu_count())
    n_procs = 4
    print(f"Running {len(param_list)} sims on {n_procs} processes")

    successes = 0
    failures = 0

    with mp.Pool(processes=n_procs) as pool:
        for tag in tqdm(
            pool.imap_unordered(worker, param_list),
            total=len(param_list),
            desc="FDTD sims",
        ):
            if isinstance(tag, str) and tag.startswith("ERROR::"):
                failures += 1
            else:
                successes += 1

    print(f"Done. Successes: {successes}, Failures: {failures}")

