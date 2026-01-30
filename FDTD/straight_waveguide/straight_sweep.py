# straight_sweep.py
# Generate a dataset of 2D straight waveguides (effective-index), cropped to a square non-PML interior.
# Uses shared shard/index utilities from utils.py.

import os
import uuid
import math
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

import meep as meep

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    neff_siwire_from_tables,
    shard_writer_generic,
    quantize_square_cell_from_crop,
    validate_crop_square,
    decode_npz_str,
)

# -----------------------------
# Defaults (Euler-style dataset conventions)
# -----------------------------
RESOLUTION_DEFAULT = 20
CROP_PX_DEFAULT = 384
DPML_DEFAULT = 2.0 / 3.0

N_GEO_DEFAULT = 500
N_PROCS_DEFAULT = 24

# Sweep ranges (um)
WG_WIDTH_MIN, WG_WIDTH_MAX = 0.38, 0.60
LAMBDA_MIN, LAMBDA_MAX = 1.40, 1.60

# Other fixed sim knobs
N_CLAD_DEFAULT = 1.444
PORT_MARGIN_UM = 2.0
SOURCE_SHIFT_UM = 0.5
PORT_Y_PAD_UM = 1.0

# Device-window margin (conditioning/mask only)
FIT_MARGIN_UM = 0.5


# -----------------------------
# Naming utilities
# -----------------------------
def _as_code_um(x_um, scale=1000):
    return int(np.round(float(x_um) * scale))


def geom_tag(wg_width, lam):
    w_i = _as_code_um(wg_width, 1000)
    lam_i = _as_code_um(lam, 1000)
    return f"w{w_i:04d}nm_lam{lam_i:04d}nm__w{wg_width:.3f}_lam{lam:.2f}"


# -----------------------------
# Mask helper (rectangle device window)
# -----------------------------
def rect_window_mask(ny, nx, dx, dy, cx_um, cy_um, wx_um, wy_um):
    # Coordinates centered at (0,0) for cropped non-PML interior.
    x = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx)
    y = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    in_x = np.abs(xx - float(cx_um)) <= (0.5 * float(wx_um))
    in_y = np.abs(yy - float(cy_um)) <= (0.5 * float(wy_um))
    return (in_x & in_y).astype(np.float32)


# -----------------------------
# Sampling: LHS over (wg_width, lambda)
# -----------------------------
def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


def build_param_list(n_geo, seed=42):
    sampler = qmc.LatinHypercube(d=2, seed=int(seed))
    u = sampler.random(int(n_geo))

    widths = WG_WIDTH_MIN + u[:, 0] * (WG_WIDTH_MAX - WG_WIDTH_MIN)
    lams = LAMBDA_MIN + u[:, 1] * (LAMBDA_MAX - LAMBDA_MIN)

    widths = _quantize_01(widths, WG_WIDTH_MIN, WG_WIDTH_MAX)
    lams = _quantize_01(lams, LAMBDA_MIN, LAMBDA_MAX)

    return [(float(w), float(lam)) for w, lam in zip(widths, lams)]


# -----------------------------
# Core sim: straight waveguide, input from left, output on right
# -----------------------------
def run_straight_sim(wg_width_um, wavelength_um, resolution, dpml_um, cell_um,
                     n_clad, port_margin_um, source_shift_um, port_y_pad_um, decay_tol):
    # Square global window: cell_x = cell_y = cell_um
    cell = meep.Vector3(float(cell_um), float(cell_um), 0)

    # Materials
    n_core = neff_siwire_from_tables(float(wg_width_um), float(wavelength_um))
    core = meep.Medium(index=float(n_core))
    clad = meep.Medium(index=float(n_clad))

    # Geometry: infinite-x waveguide
    geometry = [
        meep.Block(
            size=meep.Vector3(meep.inf, float(wg_width_um), meep.inf),
            center=meep.Vector3(0, 0, 0),
            material=core,
        )
    ]

    # Inner (non-PML) region is cell_um - 2*dpml_um
    L_inner = float(cell_um) - 2.0 * float(dpml_um)
    x_left = -0.5 * L_inner
    x_right = 0.5 * L_inner

    x_in = x_left + float(port_margin_um)
    x_out = x_right - float(port_margin_um)

    port_y_span = float(wg_width_um) + float(port_y_pad_um)
    port_size = meep.Vector3(0, port_y_span, 0)

    port_in = meep.Volume(center=meep.Vector3(x_in, 0), size=port_size)
    port_out = meep.Volume(center=meep.Vector3(x_out, 0), size=port_size)
    src_vol = meep.Volume(center=meep.Vector3(x_in - float(source_shift_um), 0), size=port_size)

    fcen = 1.0 / float(wavelength_um)
    fwidth = 0.1 * fcen

    sources = [
        meep.EigenModeSource(
            src=meep.GaussianSource(fcen, fwidth=fwidth),
            volume=src_vol,
            eig_band=1,
            eig_parity=meep.NO_PARITY,
            eig_match_freq=True,
        )
    ]

    sim = meep.Simulation(
        cell_size=cell,
        resolution=int(resolution),
        boundary_layers=[meep.PML(float(dpml_um))],
        geometry=geometry,
        default_material=clad,
        sources=sources,
    )

    m_in = sim.add_mode_monitor(fcen, 0, 1, meep.ModeRegion(volume=port_in))
    m_out = sim.add_mode_monitor(fcen, 0, 1, meep.ModeRegion(volume=port_out))

    full_plane = meep.Volume(center=meep.Vector3(0, 0), size=meep.Vector3(float(cell_um), float(cell_um), 0))
    dft = sim.add_dft_fields([meep.Ez], fcen, 0, 1, center=full_plane.center, size=full_plane.size)

    sim.run(until_after_sources=meep.stop_when_dft_decayed(tol=float(decay_tol)))

    a_in = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=meep.NO_PARITY).alpha[0, 0, 0]
    a_out = sim.get_eigenmode_coefficients(m_out, [1], eig_parity=meep.NO_PARITY).alpha[0, 0, 0]
    S21 = a_out / a_in

    eps_full = sim.get_epsilon().T.astype(np.float32)                   # [ny_full, nx_full]
    Ez_full = sim.get_dft_array(dft, meep.Ez, 0).T.astype(np.complex64) # [ny_full, nx_full]

    return eps_full, Ez_full, S21


# -----------------------------
# Worker: run sim, crop, write ONE temp npz, return path
# -----------------------------
def worker(task):
    (wg_width, lam, resolution, dpml_um, cell_um, crop_px, decay_tol, tmp_dir_str) = task
    tmp_dir = Path(tmp_dir_str)

    base = geom_tag(wg_width, lam)
    tmp_name = f"geom_{base}__{uuid.uuid4().hex}.npz"
    tmp_path = tmp_dir / tmp_name

    try:
        eps_full, Ez_full, S21 = run_straight_sim(
            wg_width_um=wg_width,
            wavelength_um=lam,
            resolution=int(resolution),
            dpml_um=float(dpml_um),
            cell_um=float(cell_um),
            n_clad=float(N_CLAD_DEFAULT),
            port_margin_um=float(PORT_MARGIN_UM),
            source_shift_um=float(SOURCE_SHIFT_UM),
            port_y_pad_um=float(PORT_Y_PAD_UM),
            decay_tol=float(decay_tol),
        )

        # Crop to non-PML interior (dpml already quantized by main)
        pml_px = int(np.round(float(dpml_um) * float(resolution)))
        eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px]

        ny, nx = eps.shape
        crop_px = int(crop_px)
        if (ny, nx) != (crop_px, crop_px):
            raise RuntimeError(f"Expected cropped ({crop_px},{crop_px}) but got {(ny,nx)}")

        # Grid meta (cropped)
        dx = 1.0 / float(resolution)
        dy = 1.0 / float(resolution)
        Lx_um = float(cell_um) - 2.0 * float(dpml_um)
        Ly_um = float(cell_um) - 2.0 * float(dpml_um)

        # Device window: full inner span in x, tight in y around the waveguide
        dev_cx = 0.0
        dev_cy = 0.0
        dev_wx = Lx_um
        dev_wy = float(wg_width) + 2.0 * (float(PORT_Y_PAD_UM) + float(FIT_MARGIN_UM))

        dev_mask = rect_window_mask(ny, nx, dx, dy, dev_cx, dev_cy, dev_wx, dev_wy)

        np.savez_compressed(
            tmp_path,
            base_tag=np.array(base),

            # geometry scalars
            wg_width_um=np.float32(wg_width),
            wavelength_um=np.float32(lam),

            # sim scalars
            resolution=np.int32(resolution),
            dpml_um=np.float32(dpml_um),

            # grid scalars (cropped)
            nx=np.int32(nx),
            ny=np.int32(ny),
            dx=np.float32(dx),
            dy=np.float32(dy),
            Lx_um=np.float32(Lx_um),
            Ly_um=np.float32(Ly_um),
            pml_px=np.int32(pml_px),

            # device window scalars (conditioning/mask only)
            dev_cx_um=np.float32(dev_cx),
            dev_cy_um=np.float32(dev_cy),
            dev_wx_um=np.float32(dev_wx),
            dev_wy_um=np.float32(dev_wy),

            # arrays (cropped)
            eps=eps.astype(np.float32),
            Ez_real=Ez.real.astype(np.float32),
            Ez_imag=Ez.imag.astype(np.float32),
            dev_mask=dev_mask.astype(np.float32),

            # S-parameter
            S21_real=np.float32(np.real(S21)),
            S21_imag=np.float32(np.imag(S21)),
        )

        return ("OK", str(tmp_path))

    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return ("ERR", f"{base} :: {repr(e)}")


# -----------------------------
# Temp NPZ -> shard sample decode (device-specific mapping)
# -----------------------------
def decode_tmp_npz(tmp_path):
    with np.load(tmp_path, allow_pickle=True) as f:
        base_tag = decode_npz_str(f["base_tag"])

        eps = f["eps"].astype(np.float32)
        Ezr = f["Ez_real"].astype(np.float32)
        Ezi = f["Ez_imag"].astype(np.float32)
        dev_mask = f["dev_mask"].astype(np.float32)

        wg_width_um = float(f["wg_width_um"])
        wavelength_um = float(f["wavelength_um"])
        resolution = int(f["resolution"])
        dpml_um = float(f["dpml_um"])

        nx = int(f["nx"])
        ny = int(f["ny"])
        dx = float(f["dx"])
        dy = float(f["dy"])
        Lx_um = float(f["Lx_um"])
        Ly_um = float(f["Ly_um"])
        pml_px = int(f["pml_px"])

        dev_cx_um = float(f["dev_cx_um"])
        dev_cy_um = float(f["dev_cy_um"])
        dev_wx_um = float(f["dev_wx_um"])
        dev_wy_um = float(f["dev_wy_um"])

        S21_real = float(f["S21_real"])
        S21_imag = float(f["S21_imag"])

    tag = f"{base_tag}__inLeft"
    arrays = {
        "eps": eps,
        "Ez_real": Ezr,
        "Ez_imag": Ezi,
        "dev_mask": dev_mask,

        "params/wg_width_um": np.array(wg_width_um, dtype=np.float32),
        "params/wavelength_um": np.array(wavelength_um, dtype=np.float32),

        "sparams/S21_real": np.array(S21_real, dtype=np.float32),
        "sparams/S21_imag": np.array(S21_imag, dtype=np.float32),

        "grid/dx": np.array(dx, dtype=np.float32),
        "grid/dy": np.array(dy, dtype=np.float32),
        "grid/nx": np.array(nx, dtype=np.int32),
        "grid/ny": np.array(ny, dtype=np.int32),
        "grid/Lx_um": np.array(Lx_um, dtype=np.float32),
        "grid/Ly_um": np.array(Ly_um, dtype=np.float32),
        "grid/pml_px": np.array(pml_px, dtype=np.int32),

        "sim/resolution": np.array(resolution, dtype=np.int32),
        "sim/dpml_um": np.array(dpml_um, dtype=np.float32),

        "device_window/cx_um": np.array(dev_cx_um, dtype=np.float32),
        "device_window/cy_um": np.array(dev_cy_um, dtype=np.float32),
        "device_window/wx_um": np.array(dev_wx_um, dtype=np.float32),
        "device_window/wy_um": np.array(dev_wy_um, dtype=np.float32),
    }
    meta = {"tag": tag}  # dataset filled by shard_writer_generic if absent
    return [(arrays, meta)]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None, help="Output root (default: <repo>/Data/straight_sweep)")
    ap.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    ap.add_argument("--crop-px", type=int, default=CROP_PX_DEFAULT, help="Non-PML crop size in pixels (square).")
    ap.add_argument("--dpml", type=float, default=DPML_DEFAULT)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--n-procs", type=int, default=N_PROCS_DEFAULT)
    ap.add_argument("--decay-tol", type=float, default=1e-5)

    ap.add_argument("--shard-size", type=int, default=200, help="Samples per shard.")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--queue-max", type=int, default=64)
    ap.add_argument("--index-every-shard", action="store_true")

    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir is not None else (repo_root / "Data" / "straight_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "tmp_geom"
    shards_dir = out_dir / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    crop_px = int(args.crop_px)
    if crop_px <= 0:
        raise ValueError("--crop-px must be > 0")

    # Quantize dpml/cell to EXACT integer-pixel sizes (Euler-style)
    dpml_um, pml_px, cell_um, _full_px = quantize_square_cell_from_crop(
        crop_px=crop_px,
        resolution=int(args.resolution),
        dpml_um=float(args.dpml),
    )
    validate_crop_square(
        cell_um=cell_um,
        resolution=int(args.resolution),
        pml_px=pml_px,
        crop_px=crop_px,
    )

    params = build_param_list(args.n_geo, seed=42)

    print(f"Unique geometries: {len(params)}")
    print(f"OUT_DIR:   {out_dir}")
    print(f"TMP_DIR:   {tmp_dir}  (bounded by queue_max ~ {args.queue_max})")
    print(f"SHARDS:    {shards_dir}")
    print(f"n_procs:   {args.n_procs}")
    print(f"resolution: {args.resolution}, crop_px: {crop_px}, dpml_um: {dpml_um:.6f}, pml_px: {pml_px}, cell_um: {cell_um:.6f}")
    print(f"shard_size(samples): {args.shard_size}  => ~{math.ceil(len(params)/args.shard_size)} shards")

    q = mp.Queue(maxsize=int(args.queue_max))

    writer_p = mp.Process(
        target=shard_writer_generic,
        args=(
            q,
            str(shards_dir),
            int(args.shard_size),
            bool(args.compress),
            "straight",
            decode_tmp_npz,
            "index.json",
            bool(args.index_every_shard),
        ),
        daemon=True,
    )
    writer_p.start()

    tasks = [
        (w, lam, int(args.resolution), float(dpml_um), float(cell_um), int(crop_px), float(args.decay_tol), str(tmp_dir))
        for (w, lam) in params
    ]

    successes, failures = 0, 0
    with mp.Pool(processes=int(args.n_procs)) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="FDTD -> tmp"):
            if status == "OK":
                q.put(("OK", payload))
                successes += 1
            else:
                failures += 1
                q.put(("ERR", payload))

    q.put(None)
    writer_p.join()

    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    print(f"Done. successes={successes}, failures={failures}")
    print(f"Shards written to: {shards_dir / 'shard_*.npz'}")
    print(f"Index written to:  {shards_dir / 'index.json'}")


if __name__ == "__main__":
    main()

