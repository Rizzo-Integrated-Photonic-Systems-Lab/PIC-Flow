# coupler_sweep.py
from __future__ import annotations

import os
import json
import uuid
import math
import argparse
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from directional_coupler.directional_coupler import DirectionalCoupler2D


# -----------------------------
# Config defaults (Euler-aligned)
# -----------------------------
RESOLUTION_DEFAULT = 20
CROP_X_PX_DEFAULT = 640   # propagation direction (rectangular)
CROP_Y_PX_DEFAULT = 128   # transverse direction  (rectangular)
DPML_DEFAULT = 2.0 / 3.0

N_GEO_DEFAULT = 500
N_PROCS_DEFAULT = 24

# Geometry and wavelength ranges (um) for rectangular 640x128 @ res 20
# X interior = 32 um, Y interior = 6.4 um
gap_min, gap_max = 0.15, 0.35

# Coupling region: up to 12 um captures most coupling physics.
# Max device extent: Lc(12) + 2*bend(6) = 24 um, fits in 25.6 um X interior (128x512 @ res20).
wg_length_min, wg_length_max = 5.0, 12.0

# Bends: with 32 um X interior, bends of 4-6 um are comfortable.
bend_length_min, bend_length_max = 4.0, 6.0

wg_width_min, wg_width_max = 0.375, 0.600
lambda_min, lambda_max = 1.40, 1.60

lead_gap_min, lead_gap_max = 0.8, 2.5

# Stratify gap (oversample small gap where coupling is strongest)
GAP_STRATA = [
    (0.150, 0.200, 0.40),
    (0.200, 0.275, 0.35),
    (0.275, 0.350, 0.25),
]


# -----------------------------
# Naming utilities
# -----------------------------
def _as_code_um(x_um: float, scale: int = 1000) -> int:
    return int(np.round(float(x_um) * scale))


def geom_tag(
    wg_width: float,
    gap: float,
    wg_length: float,
    bend_length: float,
    lead_extra_gap: float,
    lam: float,
) -> str:
    w_i = _as_code_um(wg_width, 1000)
    g_i = _as_code_um(gap, 1000)
    L_i = _as_code_um(wg_length, 1000)
    b_i = _as_code_um(bend_length, 1000)
    lead_i = _as_code_um(lead_extra_gap, 1000)
    lam_i = _as_code_um(lam, 1000)

    return (
        f"w{w_i:04d}nm_gap{g_i:04d}nm_Lc{L_i:05d}nm_b{b_i:05d}nm_lead{lead_i:05d}nm_lam{lam_i:04d}nm"
        f"__w{wg_width:.3f}_g{gap:.3f}_Lc{wg_length:.2f}_b{bend_length:.2f}_lead{lead_extra_gap:.2f}_lam{lam:.2f}"
    )


def flip_y(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=0).copy()


def synthesize_s_for_port2_from_port1(S_port1: np.ndarray) -> np.ndarray:
    # S_port1 = [S11, S21, S31, S41] for input=1
    # By y-mirror symmetry: 1<->2 and 3<->4
    return np.array([S_port1[1], S_port1[0], S_port1[3], S_port1[2]], dtype=np.complex128)


# -----------------------------
# Mask helper
# -----------------------------
def _draw_thick_line_mask(ny: int, nx: int, x0: float, y0: float, x1: float, y1: float, thickness_px: int = 3) -> np.ndarray:
    x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
    vx = x1f - x0f
    vy = y1f - y0f
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        m = np.zeros((ny, nx), dtype=np.float32)
        xi = int(np.clip(round(x0f), 0, nx - 1))
        yi = int(np.clip(round(y0f), 0, ny - 1))
        m[yi, xi] = 1.0
        return m

    yy, xx = np.indices((ny, nx), dtype=np.float32)
    t = ((xx - x0f) * vx + (yy - y0f) * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    px = x0f + t * vx
    py = y0f + t * vy
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    thr2 = float(thickness_px * thickness_px)
    return (d2 <= thr2).astype(np.float32)


# -----------------------------
# Sampling: stratified LHS over gap
# -----------------------------
def _allocate_counts(total: int, fracs: list[float]) -> list[int]:
    counts = [int(round(total * f)) for f in fracs]
    drift = total - sum(counts)
    if drift != 0:
        step = 1 if drift > 0 else -1
        for i in range(abs(drift)):
            counts[i % len(counts)] += step
    return counts


def _quantize_to_grid(x, x_min, x_max, resolution):
    """Quantize to half-pixel steps so each value maps to a distinguishable epsilon map."""
    step = 0.5 / float(resolution)  # half-pixel in um
    xq = np.round(x / step) * step
    return np.clip(xq, x_min, x_max)


def build_geometry_list(N_GEO: int, resolution: int = 20, seed_base: int = 42) -> list[tuple[float, float, float, float, float]]:
    """Sample N_GEO geometries in 5D (no wavelength), quantized to half-pixel grid.
    Returns (wg_width, gap, Lc, bend, lead_gap)."""
    strata_counts = _allocate_counts(N_GEO, [f for (_, _, f) in GAP_STRATA])

    u_list = []
    for si, (gap_lo, gap_hi, _frac) in enumerate(GAP_STRATA):
        n_i = int(strata_counts[si])
        if n_i <= 0:
            continue
        sampler = qmc.LatinHypercube(d=5, seed=seed_base + si)
        u_i = sampler.random(n_i)

        gaps_i = gap_lo + u_i[:, 0] * (gap_hi - gap_lo)
        wg_lengths_i = wg_length_min + u_i[:, 1] * (wg_length_max - wg_length_min)
        bend_lengths_i = bend_length_min + u_i[:, 2] * (bend_length_max - bend_length_min)
        wg_widths_i = wg_width_min + u_i[:, 3] * (wg_width_max - wg_width_min)
        lead_gaps_i = lead_gap_min + u_i[:, 4] * (lead_gap_max - lead_gap_min)

        u_list.append((gaps_i, wg_lengths_i, bend_lengths_i, wg_widths_i, lead_gaps_i))

    gaps = np.concatenate([t[0] for t in u_list], axis=0)
    wg_lengths = np.concatenate([t[1] for t in u_list], axis=0)
    bend_lengths = np.concatenate([t[2] for t in u_list], axis=0)
    wg_widths = np.concatenate([t[3] for t in u_list], axis=0)
    lead_gaps = np.concatenate([t[4] for t in u_list], axis=0)

    assert len(gaps) == N_GEO, f"expected N_GEO={N_GEO}, got {len(gaps)}"

    # Quantize all geometry parameters to half-pixel grid
    gaps = _quantize_to_grid(gaps, gap_min, gap_max, resolution)
    wg_widths = _quantize_to_grid(wg_widths, wg_width_min, wg_width_max, resolution)
    wg_lengths = _quantize_to_grid(wg_lengths, wg_length_min, wg_length_max, resolution)
    bend_lengths = _quantize_to_grid(bend_lengths, bend_length_min, bend_length_max, resolution)
    lead_gaps = _quantize_to_grid(lead_gaps, lead_gap_min, lead_gap_max, resolution)

    return [
        (float(w), float(g), float(L), float(b), float(lead))
        for w, g, L, b, lead in zip(wg_widths, gaps, wg_lengths, bend_lengths, lead_gaps)
    ]


def build_param_list(N_GEO: int, wavelengths: list[float], resolution: int = 20, seed_base: int = 42) -> list[tuple[float, float, float, float, float, float]]:
    """Sample N_GEO geometries in 5D, then cross with wavelength list.
    Returns N_GEO * len(wavelengths) tuples of (wg_width, gap, Lc, bend, lead_gap, wavelength)."""
    geos = build_geometry_list(N_GEO, resolution=resolution, seed_base=seed_base)
    params = []
    for w, g, L, b, lead in geos:
        for lam in wavelengths:
            params.append((w, g, L, b, lead, float(lam)))
    return params


# -----------------------------
# Worker: run port1 sim, write ONE temp npz, return path
# -----------------------------
def run_fdtd_sim_port1(
    wg_width: float,
    gap: float,
    wg_length: float,
    bend_length: float,
    lead_extra_gap: float,
    wl: float,
    RESOLUTION: int,
    dpml_um: float,
    cell_x_um: float,
    cell_y_um: float,
    crop_x_px: int,
    crop_y_px: int,
    decay_tol: float,
):
    dc = DirectionalCoupler2D(
        wg_width_um=wg_width,
        gap_um=gap,
        wg_length_um=wg_length,
        wavelength_um=wl,
        resolution=int(RESOLUTION),
        dpml=float(dpml_um),
        crop_x_px=int(crop_x_px),
        crop_y_px=int(crop_y_px),
        cell_x_um=float(cell_x_um),
        cell_y_um=float(cell_y_um),
        pad_y_um=1.0,
        lead_extra_gap_um=lead_extra_gap,
        bend_length_um=bend_length,
        bend_n_segments=48,
        source_shift_um=0.5,
        quantize_grid=False,  # already quantized in main
        fit_margin_um=0.5,
    )

    eps, Ez, Hx, Hy, S_dict, cell = dc.run_sim(input_port=1, decay_tol=float(decay_tol))
    return dc, S_dict, Ez, eps, cell


def worker(task):
    """
    Returns:
      ("OK", temp_npz_path_str) or ("ERR", err_string)
    """
    (wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, RESOLUTION, dpml_um, cell_x_um, cell_y_um, crop_x_px, crop_y_px, decay_tol, tmp_dir_str) = task
    tmp_dir = Path(tmp_dir_str)

    base = geom_tag(wg_width, gap, wg_length, bend_length, lead_extra_gap, lam)
    tmp_name = f"geom_{base}__{uuid.uuid4().hex}.npz"
    tmp_path = tmp_dir / tmp_name

    try:
        dc, S_dict, Ez1_full, eps_full, cell = run_fdtd_sim_port1(
            wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, RESOLUTION, dpml_um, cell_x_um, cell_y_um, crop_x_px, crop_y_px, decay_tol
        )
        if S_dict is None:
            raise RuntimeError("S_dict is None")

        eps_full = np.asarray(eps_full, dtype=np.float32)
        Ez1_full = np.asarray(Ez1_full, dtype=np.complex64)

        # Crop to non-PML (Euler-style)
        pml_px = int(np.round(float(dpml_um) * float(RESOLUTION)))
        eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez1 = Ez1_full[pml_px:-pml_px, pml_px:-pml_px]

        ny, nx = eps.shape
        if (ny, nx) != (int(crop_y_px), int(crop_x_px)):
            raise RuntimeError(f"Expected cropped ({crop_y_px},{crop_x_px}) but got {(ny,nx)}")

        # Source mask for input_port=1 (cropped coords)
        src_px = dc.get_source_region_px(input_port=1, crop_pml=True)
        src_mask1 = _draw_thick_line_mask(
            ny, nx,
            src_px["line_start_px"][0], src_px["line_start_px"][1],
            src_px["line_end_px"][0], src_px["line_end_px"][1],
            thickness_px=3,
        )

        # Port masks for ports 1..4 (cropped coords)
        port_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        port_masks = []
        for p in port_ids.tolist():
            pr = dc.get_port_region_px(p, crop_pml=True)
            pm = _draw_thick_line_mask(
                ny, nx,
                pr["line_start_px"][0], pr["line_start_px"][1],
                pr["line_end_px"][0], pr["line_end_px"][1],
                thickness_px=3,
            )
            port_masks.append(pm.astype(np.float32))
        port_masks = np.stack(port_masks, axis=0).astype(np.float32)  # [4, ny, nx]

        # S vector for input=1
        S1 = np.array([S_dict[(1, 1)], S_dict[(2, 1)], S_dict[(3, 1)], S_dict[(4, 1)]], dtype=np.complex128)

        # grid meta (cropped)
        dx = 1.0 / float(RESOLUTION)
        dy = 1.0 / float(RESOLUTION)
        Lx_um = float(cell[0]) - 2.0 * float(dpml_um)
        Ly_um = float(cell[1]) - 2.0 * float(dpml_um)

        np.savez_compressed(
            tmp_path,
            # geometry scalars
            wg_width_um=np.float32(wg_width),
            gap_um=np.float32(gap),
            Lc_um=np.float32(wg_length),
            bend_length_um=np.float32(bend_length),
            lead_extra_gap_um=np.float32(lead_extra_gap),
            wavelength_um=np.float32(lam),
            resolution=np.int32(RESOLUTION),
            dpml_um=np.float32(dpml_um),
            # grid scalars (cropped)
            nx=np.int32(nx),
            ny=np.int32(ny),
            dx=np.float32(dx),
            dy=np.float32(dy),
            Lx_um=np.float32(Lx_um),
            Ly_um=np.float32(Ly_um),
            pml_px=np.int32(pml_px),
            # arrays (cropped)
            eps=eps.astype(np.float32),
            Ez_real=Ez1.real.astype(np.float32),
            Ez_imag=Ez1.imag.astype(np.float32),
            src_mask=src_mask1.astype(np.float32),
            port_ids=port_ids.astype(np.int32),
            port_masks=port_masks.astype(np.float32),
            S1_real=S1.real.astype(np.float32),
            S1_imag=S1.imag.astype(np.float32),
            base_tag=np.array(base),
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
# Shard writer process (streaming)
# -----------------------------
def _write_shard_npz(out_path: Path, samples: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]], compress: bool):
    save_dict: Dict[str, np.ndarray] = {}
    for i, (arrays, meta) in enumerate(samples):
        prefix = f"s{i}/"
        for k, v in arrays.items():
            save_dict[prefix + k] = v
        for k, v in meta.items():
            save_dict[prefix + k] = np.array(v)

    # Atomic write to avoid leaving partial/corrupt shards if interrupted.
    # Write a temporary file in the same directory and then replace.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(out_path.parent),
        prefix=out_path.stem + ".",
        suffix=".tmp.npz",
        delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
    try:
        if compress:
            np.savez_compressed(tmp_path, **save_dict)
        else:
            np.savez(tmp_path, **save_dict)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _atomic_write_index(index_path: Path, index_list: List[Dict[str, object]]):
    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(index_list, f, indent=2)
    os.replace(tmp_path, index_path)


def shard_writer(
    q: mp.Queue,
    shards_root_str: str,
    shard_size: int,
    compress: bool,
    dataset_name: str = "coupler",
    index_name: str = "index.json",
    save_index_every_shard: bool = True,
):
    shards_root = Path(shards_root_str)
    shards_root.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, object]] = []
    index_path = shards_root / index_name

    buffer: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]] = []
    shard_id = 0

    def write_one_shard(batch: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]]):
        nonlocal shard_id, index
        shard_name = f"shard_{shard_id:05d}.npz"
        shard_path = shards_root / shard_name
        _write_shard_npz(shard_path, batch, compress=compress)
        for slot, (_, meta) in enumerate(batch):
            index.append({"tag": meta["tag"], "shard": shard_name, "slot": int(slot)})
        shard_id += 1
        if save_index_every_shard:
            _atomic_write_index(index_path, index)

    while True:
        msg = q.get()
        if msg is None:
            break

        status, payload = msg
        if status != "OK":
            continue

        tmp_path = Path(payload)
        try:
            with np.load(tmp_path, allow_pickle=True) as f:
                base_tag = f["base_tag"].item()
                if isinstance(base_tag, bytes):
                    base_tag = base_tag.decode("utf-8")
                base_tag = str(base_tag)

                eps = f["eps"].astype(np.float32)
                Ezr = f["Ez_real"].astype(np.float32)
                Ezi = f["Ez_imag"].astype(np.float32)
                src1 = f["src_mask"].astype(np.float32)
                port_ids = f["port_ids"].astype(np.int32) if "port_ids" in f else None
                port_masks = f["port_masks"].astype(np.float32) if "port_masks" in f else None
                S1 = (f["S1_real"] + 1j * f["S1_imag"]).astype(np.complex128)

                # scalars
                wg_width_um = float(f["wg_width_um"])
                gap_um = float(f["gap_um"])
                Lc_um = float(f["Lc_um"])
                bend_length_um = float(f["bend_length_um"])
                lead_extra_gap_um = float(f["lead_extra_gap_um"])
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

            # sample 1 (inPort1)
            tag1 = f"{base_tag}__inPort1"
            arrays1 = {
                "eps": eps,
                "Ez_real": Ezr,
                "Ez_imag": Ezi,
                "src_mask": src1,
                **(
                    {"ports/ids": port_ids, "ports/masks": port_masks}
                    if (port_ids is not None and port_masks is not None)
                    else {}
                ),
                "sparams/input_port": np.array(1, dtype=np.int32),
                "sparams/source_off": np.array(0, dtype=np.int32),
                "sparams/gap_um": np.array(gap_um, dtype=np.float32),
                "sparams/Lc_um": np.array(Lc_um, dtype=np.float32),
                "sparams/bend_length_um": np.array(bend_length_um, dtype=np.float32),
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/lead_extra_gap_um": np.array(lead_extra_gap_um, dtype=np.float32),
                "sparams/wavelength_um": np.array(wavelength_um, dtype=np.float32),
                "sparams/resolution": np.array(resolution, dtype=np.int32),
                "sparams/dpml_um": np.array(dpml_um, dtype=np.float32),
                "sparams/S_real": S1.real.astype(np.float32),
                "sparams/S_imag": S1.imag.astype(np.float32),
                "grid/dx": np.array(dx, dtype=np.float32),
                "grid/dy": np.array(dy, dtype=np.float32),
                "grid/nx": np.array(nx, dtype=np.int32),
                "grid/ny": np.array(ny, dtype=np.int32),
                "grid/Lx_um": np.array(Lx_um, dtype=np.float32),
                "grid/Ly_um": np.array(Ly_um, dtype=np.float32),
                "grid/pml_px": np.array(pml_px, dtype=np.int32),
            }
            meta1 = {"tag": tag1, "dataset": dataset_name}

            # sample 2 (inPort2 via symmetry)
            tag2 = f"{base_tag}__inPort2"
            Ezr2 = flip_y(Ezr)
            Ezi2 = flip_y(Ezi)
            eps2 = flip_y(eps)
            src2 = flip_y(src1)
            S2 = synthesize_s_for_port2_from_port1(S1)

            if port_masks is not None and port_ids is not None and port_ids.shape[0] == 4:
                m1, m2, m3, m4 = port_masks[0], port_masks[1], port_masks[2], port_masks[3]
                port_masks2 = np.stack([flip_y(m2), flip_y(m1), flip_y(m4), flip_y(m3)], axis=0).astype(np.float32)
                port_ids2 = port_ids.copy()
            elif port_masks is not None and port_ids is not None:
                port_masks2 = flip_y(port_masks).astype(np.float32)
                port_ids2 = port_ids.copy()
            else:
                port_masks2 = None
                port_ids2 = None

            arrays2 = {
                "eps": eps2.astype(np.float32),
                "Ez_real": Ezr2.astype(np.float32),
                "Ez_imag": Ezi2.astype(np.float32),
                "src_mask": src2.astype(np.float32),
                **(
                    {"ports/ids": port_ids2.astype(np.int32), "ports/masks": port_masks2.astype(np.float32)}
                    if (port_ids2 is not None and port_masks2 is not None)
                    else {}
                ),
                "sparams/input_port": np.array(2, dtype=np.int32),
                "sparams/source_off": np.array(0, dtype=np.int32),
                "sparams/gap_um": np.array(gap_um, dtype=np.float32),
                "sparams/Lc_um": np.array(Lc_um, dtype=np.float32),
                "sparams/bend_length_um": np.array(bend_length_um, dtype=np.float32),
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/lead_extra_gap_um": np.array(lead_extra_gap_um, dtype=np.float32),
                "sparams/wavelength_um": np.array(wavelength_um, dtype=np.float32),
                "sparams/resolution": np.array(resolution, dtype=np.int32),
                "sparams/dpml_um": np.array(dpml_um, dtype=np.float32),
                "sparams/S_real": S2.real.astype(np.float32),
                "sparams/S_imag": S2.imag.astype(np.float32),
                "grid/dx": np.array(dx, dtype=np.float32),
                "grid/dy": np.array(dy, dtype=np.float32),
                "grid/nx": np.array(nx, dtype=np.int32),
                "grid/ny": np.array(ny, dtype=np.int32),
                "grid/Lx_um": np.array(Lx_um, dtype=np.float32),
                "grid/Ly_um": np.array(Ly_um, dtype=np.float32),
                "grid/pml_px": np.array(pml_px, dtype=np.int32),
            }
            meta2 = {"tag": tag2, "dataset": dataset_name}

            buffer.append((arrays1, meta1))
            buffer.append((arrays2, meta2))

            while len(buffer) >= shard_size:
                batch = buffer[:shard_size]
                buffer = buffer[shard_size:]
                write_one_shard(batch)

        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    if buffer:
        write_one_shard(buffer)

    if not save_index_every_shard:
        _atomic_write_index(index_path, index)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None, help="Output root (default: <repo>/Data/coupler_sweep)")
    ap.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    ap.add_argument("--crop-x-px", type=int, default=CROP_X_PX_DEFAULT, help="Non-PML crop X pixels (propagation).")
    ap.add_argument("--crop-y-px", type=int, default=CROP_Y_PX_DEFAULT, help="Non-PML crop Y pixels (transverse).")
    ap.add_argument("--dpml", type=float, default=DPML_DEFAULT)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--n-procs", type=int, default=N_PROCS_DEFAULT)
    ap.add_argument("--decay-tol", type=float, default=1e-5)
    ap.add_argument("--wavelengths", type=str, default=None,
                    help="Comma-separated wavelengths in um (e.g. '1.45,1.50,1.55,1.60'). "
                         "If not set, uses --n-wavelengths evenly spaced points.")
    ap.add_argument("--n-wavelengths", type=int, default=1,
                    help="Number of evenly spaced wavelengths in [lambda_min, lambda_max]. "
                         "Ignored if --wavelengths is set.")

    ap.add_argument("--shard-size", type=int, default=100, help="Samples per shard (NOTE: each geometry yields 2 samples).")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--queue-max", type=int, default=64)
    ap.add_argument("--index-every-shard", action="store_true")

    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR = Path(args.out_dir) if args.out_dir is not None else (repo_root / "Data" / "coupler_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = OUT_DIR / "tmp_geom"
    shards_dir = OUT_DIR / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Quantize dpml/cell to EXACT integer-pixel sizes (rectangular grid).
    crop_x_px = int(args.crop_x_px)
    crop_y_px = int(args.crop_y_px)
    if crop_x_px <= 0 or crop_y_px <= 0:
        raise ValueError("--crop-x-px and --crop-y-px must be > 0")

    pml_px = int(np.round(float(args.dpml) * float(args.resolution)))
    dpml_um = float(pml_px) / float(args.resolution)
    full_x_px = int(crop_x_px + 2 * pml_px)
    full_y_px = int(crop_y_px + 2 * pml_px)
    cell_x_um = float(full_x_px) / float(args.resolution)
    cell_y_um = float(full_y_px) / float(args.resolution)

    # Validate cropped shapes
    nx_full = int(np.round(cell_x_um * float(args.resolution)))
    ny_full = int(np.round(cell_y_um * float(args.resolution)))
    if (nx_full - 2 * pml_px) != crop_x_px:
        raise ValueError(f"X crop mismatch: expected {crop_x_px} but got {nx_full - 2*pml_px}.")
    if (ny_full - 2 * pml_px) != crop_y_px:
        raise ValueError(f"Y crop mismatch: expected {crop_y_px} but got {ny_full - 2*pml_px}.")

    # Build wavelength list
    if args.wavelengths is not None:
        wavelengths = [float(x.strip()) for x in args.wavelengths.split(",")]
    else:
        n_wl = max(1, int(args.n_wavelengths))
        if n_wl == 1:
            wavelengths = [0.5 * (lambda_min + lambda_max)]
        else:
            wavelengths = np.linspace(lambda_min, lambda_max, n_wl).tolist()
    wavelengths = sorted(set(round(w, 4) for w in wavelengths))

    params = build_param_list(args.n_geo, wavelengths=wavelengths, resolution=args.resolution, seed_base=42)

    n_sims = len(params)
    n_geos = args.n_geo
    print(f"Unique geometries: {n_geos}")
    print(f"Wavelengths per geometry: {len(wavelengths)}  {wavelengths}")
    print(f"Total FDTD sims: {n_sims}")
    print(f"Total samples produced (with symmetry): {2 * n_sims}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"TMP_DIR:   {tmp_dir}  (bounded by queue_max ~ {args.queue_max})")
    print(f"SHARDS:    {shards_dir}")
    print(f"n_procs:   {args.n_procs}")
    print(f"resolution: {args.resolution}, crop: {crop_x_px}x{crop_y_px}, dpml_um: {dpml_um:.6f}, pml_px: {pml_px}, cell: {cell_x_um:.3f}x{cell_y_um:.3f} um")
    print(f"shard_size(samples): {args.shard_size}  => ~{math.ceil((2*len(params))/args.shard_size)} shards")

    q: mp.Queue = mp.Queue(maxsize=args.queue_max)

    writer_p = mp.Process(
        target=shard_writer,
        args=(
            q,
            str(shards_dir),
            int(args.shard_size),
            bool(args.compress),
            "coupler",
            "index.json",
            bool(args.index_every_shard),
        ),
        # Keep non-daemon so it can finish queued writes on orderly shutdown.
        daemon=False,
    )
    writer_p.start()

    tasks = [
        (w, g, L, b, lead, lam, int(args.resolution), float(dpml_um), float(cell_x_um), float(cell_y_um), int(crop_x_px), int(crop_y_px), float(args.decay_tol), str(tmp_dir))
        for (w, g, L, b, lead, lam) in params
    ]

    successes, failures = 0, 0
    with mp.Pool(processes=args.n_procs) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="FDTD (port1) -> tmp"):
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

    print(f"Done. Geom successes={successes}, failures={failures}")
    print(f"Shards written to: {shards_dir / 'shard_*.npz'}")
    print(f"Index written to:  {shards_dir / 'index.json'}")


if __name__ == "__main__":
    main()
