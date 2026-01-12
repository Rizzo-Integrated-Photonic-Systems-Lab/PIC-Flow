# euler_bend_sweep.py
"""
euler_bend_sweep.py

Coupler-sweep style sharded dataset writer for EulerZigZag2D.

Writes shards compatible with `Model/dataset.py` (use_shards=True):
  - Data/euler_bend_sweep/shards/shard_*.npz
  - Data/euler_bend_sweep/shards/index.json

Each geometry yields ONE sample (input_port=1), with:
  - eps, Ez_real, Ez_imag, src_mask
  - sparams/S_real, sparams/S_imag (vector length 2: [S11, S21])
  - sparams/<conditioning scalars>
  - grid/<scalars>
  - optional ports/ids + ports/masks (2 ports: input/output)

IMPORTANT FIX (2026-01-10):
- Wavelength `lam` is sampled per-geometry in build_param_list_valid(). The task list MUST pass
  this sampled `lam` into the simulator. Previously, tasks incorrectly used args.wavelength_um,
  causing metadata to claim a wavelength sweep while sims were actually run at a fixed lambda.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    # Keep consistent with FDTD/coupler_sweep.py
    from scipy.stats import qmc
except Exception:  # pragma: no cover
    qmc = None
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x

from euler_bend_device import EulerZigZag2D


# -----------------------------
# Config defaults
# -----------------------------
RESOLUTION_DEFAULT = 20
N_GEO_DEFAULT = 5000
N_PROCS_DEFAULT = 24
CROP_PX_DEFAULT = 384

DPML_DEFAULT = 2.0 / 3.0  # -> pml_px = 16 at res=24
WAVELENGTH_DEFAULT = 1.55

# Sweep ranges (match neff table coverage)
wg_width_min, wg_width_max = 0.38, 0.60  # Must match neff_tables range
wavelength_min, wavelength_max = 1.40, 1.60  # Sweep wavelength too
n_zigs_min, n_zigs_max = 1, 3
# allow tighter bends and longer straights for larger crop sizes
R_min_min, R_min_max = 1.2, 3.0
straight_x_min, straight_x_max = 0.5, 3.0
straight_y_min, straight_y_max = 0.2, 1.5
lead_x_min, lead_x_max = 1.0, 4.0
min_straight_in_min, min_straight_in_max = 2.0, 6.0
y0_min, y0_max = -2.5, 2.5

ROT_CHOICES = [0.0, 90.0, 180.0, 270.0]
BEND_CHOICES = [15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0]


# -----------------------------
# Naming utilities
# -----------------------------
def _as_code_um(x_um: float, scale: int = 1000) -> int:
    return int(np.round(float(x_um) * scale))


def geom_tag(
    wg_width: float,
    n_zigs: int,
    start_up: int,
    end_with_turn: int,
    end_turn_deg: float,
    R_min: float,
    sx: float,
    sy: float,
    lead_x: float,
    min_straight_in: float,
    y0: float,
    bend_angle: float,
    rotation: float,
    lam: float,
) -> str:
    w_i = _as_code_um(wg_width, 1000)
    r_i = _as_code_um(R_min, 1000)
    sx_i = _as_code_um(sx, 1000)
    sy_i = _as_code_um(sy, 1000)
    lead_i = _as_code_um(lead_x, 1000)
    ms_i = _as_code_um(min_straight_in, 1000)
    y0_i = _as_code_um(y0, 1000)
    lam_i = _as_code_um(lam, 1000)
    ba_i = int(round(bend_angle))
    ewt_i = int(end_with_turn)
    et_i = int(round(end_turn_deg))
    rot_i = int(round(rotation))
    return (
        f"w{w_i:04d}nm_n{n_zigs}_up{start_up}_R{r_i:05d}nm_sx{sx_i:05d}nm_sy{sy_i:05d}nm"
        f"_lead{lead_i:05d}nm_ms{ms_i:05d}nm_y0{y0_i:+06d}nm"
        f"_ewt{ewt_i}_et{et_i:03d}_ba{ba_i:03d}_rot{rot_i:03d}_lam{lam_i:04d}nm"
    )


# -----------------------------
# Sampling
# -----------------------------
def build_param_list(N_GEO: int, seed: int = 42) -> list[tuple]:
    raise RuntimeError("build_param_list is deprecated; use build_param_list_valid().")


def _pick_choice(u01: float, choices: list[float]) -> float:
    """Map u in [0,1) to a discrete choice deterministically."""
    if not choices:
        raise ValueError("choices must be non-empty")
    k = int(np.floor(float(u01) * len(choices)))
    k = int(np.clip(k, 0, len(choices) - 1))
    return float(choices[k])


def build_param_list_valid(
    N_GEO: int,
    *,
    seed: int,
    resolution: int,
    dpml: float,
    cell_um: float,
    wavelength_um: float,  # kept for API compatibility; NOT used when sweeping wavelength
    crop_px: int,
    max_tries: int = 200000,
    include_turns: bool = False,
    turn_prob: float = 0.5,
    end_turn_deg: float | None = None,
) -> list[tuple]:
    """
    Rejection-sample parameters until we have N_GEO *valid* geometries (i.e. EulerZigZag2D
    can be constructed and fits in the non-PML crop by construction).

    NOTE:
      - This generator currently SWEEPS wavelength using [wavelength_min, wavelength_max].
      - The `wavelength_um` argument is retained for compatibility but is not used.
    """
    if qmc is None:
        raise RuntimeError(
            "scipy is required for LHS sampling (scipy.stats.qmc). "
            "Install scipy or run in the same environment as FDTD/coupler_sweep.py."
        )

    N_GEO = int(N_GEO)
    if N_GEO <= 0:
        return []

    include_turns = bool(include_turns)
    turn_prob = float(turn_prob)
    end_turn_deg = None if end_turn_deg is None else float(end_turn_deg)

    crop_px = int(crop_px)
    if crop_px <= 0:
        raise ValueError("crop_px must be > 0")

    # Keep rejection rates reasonable when generating a smaller pixel crop:
    # scale all length-like sweep ranges approximately linearly with crop size.
    # (Waveguide width range is left unchanged.)
    scale = float(crop_px) / 512.0

    def _s(x: float) -> float:
        return float(x) * scale

    # LHS columns:
    #  0 wg_width, 1 n_zigs, 2 start_up, 3 R_min, 4 sx, 5 sy, 6 lead_x, 7 min_straight_in,
    #  8 y0, 9 rotation, 10 bend_choice, 11 turn_coin, 12 wavelength
    d = 13
    sampler = qmc.LatinHypercube(d=d, seed=int(seed))

    out: list[tuple] = []
    tried = 0

    # Oversample and validate until we have enough (rejection-friendly but stratified).
    while len(out) < N_GEO:
        remaining = N_GEO - len(out)
        # Heuristic oversample factor (turn devices tend to fail fit more often).
        factor = 4 if include_turns else 3
        n_draw = min(max(remaining * factor, 256), 8192)

        u = sampler.random(n_draw)  # [n_draw, d] in [0,1)
        for ur in u:
            tried += 1
            if tried > int(max_tries):
                raise RuntimeError(f"Failed to generate {N_GEO} valid geometries within max_tries={max_tries}.")

            wg_width = float(wg_width_min + float(ur[0]) * (wg_width_max - wg_width_min))
            n_zigs = int(n_zigs_min + np.floor(float(ur[1]) * (n_zigs_max - n_zigs_min + 1)))
            n_zigs = int(np.clip(n_zigs, n_zigs_min, n_zigs_max))
            start_up = int(float(ur[2]) >= 0.5)

            R_min = float(_s(R_min_min) + float(ur[3]) * (_s(R_min_max) - _s(R_min_min)))
            sx = float(_s(straight_x_min) + float(ur[4]) * (_s(straight_x_max) - _s(straight_x_min)))
            sy = float(_s(straight_y_min) + float(ur[5]) * (_s(straight_y_max) - _s(straight_y_min)))
            lead_x = float(_s(lead_x_min) + float(ur[6]) * (_s(lead_x_max) - _s(lead_x_min)))
            min_straight_in = float(_s(min_straight_in_min) + float(ur[7]) * (_s(min_straight_in_max) - _s(min_straight_in_min)))
            y0 = float(_s(y0_min) + float(ur[8]) * (_s(y0_max) - _s(y0_min)))

            rotation = _pick_choice(float(ur[9]), ROT_CHOICES)
            bend_angle = _pick_choice(float(ur[10]), BEND_CHOICES)

            # Sweep wavelength from LHS (ignore fixed wavelength_um arg)
            lam = float(wavelength_min + float(ur[12]) * (wavelength_max - wavelength_min))

            # Optional: net-turn devices (adjacent-edge I/O). Use an LHS-driven "coin".
            ewt = 0
            # Allow the terminating turn angle to vary (if not provided, sample like bend choices).
            et_deg = float(end_turn_deg) if end_turn_deg is not None else _pick_choice(float(ur[10]), BEND_CHOICES)
            if include_turns and (float(ur[11]) < turn_prob):
                ewt = 1
                bend_angle = _pick_choice(float(ur[10]), [30.0, 45.0, 60.0, 90.0])

            try:
                # Validate by constructing geometry (no time stepping)
                EulerZigZag2D(
                    wg_width_um=wg_width,
                    wavelength_um=lam,
                    resolution=int(resolution),
                    dpml=float(dpml),
                    crop_px=int(crop_px),
                    cell_x_um=float(cell_um),
                    cell_y_um=float(cell_um),
                    n_zigs=int(n_zigs),
                    start_up=bool(start_up),
                    end_with_turn=bool(ewt),
                    end_turn_deg=(float(et_deg) if bool(ewt) else None),
                    R_min_um=R_min,
                    straight_x_um=sx,
                    straight_y_um=sy,
                    lead_x_um=lead_x,
                    min_straight_input_um=min_straight_in,
                    bend_angle_deg=bend_angle,
                    lead_extend_through_pml=True,
                    y0_um=y0,
                    fit_margin_um=0.5,
                    path_rotation_deg=rotation,
                )
            except Exception:
                continue

            out.append(
                (
                    wg_width,
                    int(n_zigs),
                    int(start_up),
                    int(ewt),
                    float(et_deg),
                    R_min,
                    sx,
                    sy,
                    lead_x,
                    min_straight_in,
                    y0,
                    bend_angle,
                    rotation,
                    lam,
                )
            )
            if len(out) >= N_GEO:
                break

    return out


# -----------------------------
# Mask helpers (pixel-space line rasterization)
# -----------------------------
def _draw_thick_line_mask(ny: int, nx: int, x0: float, y0: float, x1: float, y1: float, thickness_px: int = 3) -> np.ndarray:
    """
    Draw a thick line segment into a float32 mask using a simple distance-to-segment threshold.
    Coordinates are in pixel space (x right, y up) consistent with imshow(origin='lower').
    """
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
    # projection t in [0,1]
    t = ((xx - x0f) * vx + (yy - y0f) * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    px = x0f + t * vx
    py = y0f + t * vy
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    thr2 = float(thickness_px * thickness_px)
    return (d2 <= thr2).astype(np.float32)


# -----------------------------
# Worker: run sim, write ONE temp npz, return path
# -----------------------------
def run_fdtd_sim(
    wg_width: float,
    n_zigs: int,
    start_up: int,
    end_with_turn: int,
    end_turn_deg: float,
    R_min: float,
    sx: float,
    sy: float,
    lead_x: float,
    min_straight_in: float,
    y0: float,
    bend_angle: float,
    rotation: float,
    lam: float,
    resolution: int,
    dpml: float,
    cell_um: float,
    crop_px: int,
    decay_tol: float,
):
    dev = EulerZigZag2D(
        wg_width_um=wg_width,
        wavelength_um=lam,
        resolution=resolution,
        dpml=dpml,
        crop_px=int(crop_px),
        cell_x_um=cell_um,
        cell_y_um=cell_um,
        n_zigs=int(n_zigs),
        start_up=bool(start_up),
        end_with_turn=bool(end_with_turn),
        end_turn_deg=(float(end_turn_deg) if int(end_with_turn) != 0 else None),
        R_min_um=R_min,
        straight_x_um=sx,
        straight_y_um=sy,
        lead_x_um=lead_x,
        min_straight_input_um=min_straight_in,
        bend_angle_deg=bend_angle,
        lead_extend_through_pml=True,
        y0_um=y0,
        fit_margin_um=0.5,
        path_rotation_deg=rotation,
    )

    # eps_full, Ez_full, S_dict, cell
    eps_full, Ez_full, S, cell = dev.run_sim(decay_tol=decay_tol)
    eps_full = np.asarray(eps_full, dtype=np.float32)
    Ez_full = np.asarray(Ez_full, dtype=np.complex64)

    # Crop to non-PML (target training shape)
    pml_px = int(np.round(float(dpml) * float(resolution)))
    eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
    Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px]

    ny, nx = eps.shape
    crop_px = int(crop_px)
    if (ny, nx) != (crop_px, crop_px):
        raise RuntimeError(f"Expected cropped ({crop_px},{crop_px}) but got {(ny,nx)}")

    # Source + port masks from the device pixel helpers (cropped coords)
    src_px = dev.get_source_region_px(crop_pml=True)
    out_px = dev.get_output_region_px(crop_pml=True)
    src_mask = _draw_thick_line_mask(
        ny, nx,
        src_px["line_start_px"][0], src_px["line_start_px"][1],
        src_px["line_end_px"][0], src_px["line_end_px"][1],
        thickness_px=3
    )
    port1_mask = _draw_thick_line_mask(
        ny, nx,
        src_px["line_start_px"][0], src_px["line_start_px"][1],
        src_px["line_end_px"][0], src_px["line_end_px"][1],
        thickness_px=3
    )
    port2_mask = _draw_thick_line_mask(
        ny, nx,
        out_px["line_start_px"][0], out_px["line_start_px"][1],
        out_px["line_end_px"][0], out_px["line_end_px"][1],
        thickness_px=3
    )
    port_ids = np.array([1, 2], dtype=np.int32)
    port_masks = np.stack([port1_mask, port2_mask], axis=0).astype(np.float32)  # [2, ny, nx]

    # S vector for input=1: [S11, S21]
    S1 = np.array([S[(1, 1)], S[(2, 1)]], dtype=np.complex128)

    # grid meta
    dx = 1.0 / float(resolution)
    dy = 1.0 / float(resolution)
    Lx_um = float(cell[0]) - 2.0 * float(dpml)
    Ly_um = float(cell[1]) - 2.0 * float(dpml)

    return dev, S1, Ez, eps, src_mask, port_ids, port_masks, (nx, ny, dx, dy, Lx_um, Ly_um, pml_px)


def worker(task):
    """
    Returns:
      ("OK", temp_npz_path_str) or ("ERR", err_string)
    """
    (
        wg_width,
        n_zigs,
        start_up,
        end_with_turn,
        end_turn_deg,
        R_min,
        sx,
        sy,
        lead_x,
        min_straight_in,
        y0,
        bend_angle,
        rotation,
        lam,
        RESOLUTION,
        dpml,
        cell_um,
        crop_px,
        decay_tol,
        tmp_dir_str,
    ) = task

    tmp_dir = Path(tmp_dir_str)
    base = geom_tag(
        wg_width, n_zigs, start_up,
        int(end_with_turn), float(end_turn_deg),
        R_min, sx, sy, lead_x, min_straight_in, y0, bend_angle, rotation, lam
    )
    tmp_name = f"geom_{base}__{uuid.uuid4().hex}.npz"
    tmp_path = tmp_dir / tmp_name

    try:
        _dev, S1, Ez, eps, src_mask, port_ids, port_masks, grid = run_fdtd_sim(
            wg_width,
            n_zigs,
            start_up,
            int(end_with_turn),
            float(end_turn_deg),
            R_min,
            sx,
            sy,
            lead_x,
            min_straight_in,
            y0,
            bend_angle,
            rotation,
            lam,
            RESOLUTION,
            dpml,
            cell_um,
            int(crop_px),
            decay_tol,
        )

        nx, ny, dx, dy, Lx_um, Ly_um, pml_px = grid

        np.savez_compressed(
            tmp_path,
            # geometry scalars (for conditioning)
            wg_width_um=np.float32(wg_width),
            n_zigs=np.int32(n_zigs),
            start_up=np.int32(start_up),
            end_with_turn=np.int32(end_with_turn),
            end_turn_deg=np.float32(end_turn_deg),
            R_min_um=np.float32(R_min),
            straight_x_um=np.float32(sx),
            straight_y_um=np.float32(sy),
            lead_x_um=np.float32(lead_x),
            min_straight_input_um=np.float32(min_straight_in),
            y0_um=np.float32(y0),
            bend_angle_deg=np.float32(bend_angle),
            path_rotation_deg=np.float32(rotation),
            wavelength_um=np.float32(lam),
            resolution=np.int32(RESOLUTION),
            dpml_um=np.float32(dpml),
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
            Ez_real=Ez.real.astype(np.float32),
            Ez_imag=Ez.imag.astype(np.float32),
            src_mask=src_mask.astype(np.float32),
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
    if compress:
        np.savez_compressed(out_path, **save_dict)
    else:
        np.savez(out_path, **save_dict)


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
    dataset_name: str = "euler_bend",
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
                src = f["src_mask"].astype(np.float32)
                port_ids = f["port_ids"].astype(np.int32) if "port_ids" in f else None
                port_masks = f["port_masks"].astype(np.float32) if "port_masks" in f else None
                S1 = (f["S1_real"] + 1j * f["S1_imag"]).astype(np.complex128)

                # scalars
                wg_width_um = float(f["wg_width_um"])
                n_zigs = int(f["n_zigs"])
                start_up = int(f["start_up"])
                end_with_turn = int(f["end_with_turn"]) if "end_with_turn" in f else 0
                end_turn_deg = float(f["end_turn_deg"]) if "end_turn_deg" in f else 0.0
                R_min_um = float(f["R_min_um"])
                straight_x_um = float(f["straight_x_um"])
                straight_y_um = float(f["straight_y_um"])
                lead_x_um = float(f["lead_x_um"])
                min_straight_input_um = float(f["min_straight_input_um"])
                y0_um = float(f["y0_um"])
                bend_angle_deg = float(f["bend_angle_deg"])
                path_rotation_deg = float(f["path_rotation_deg"])

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

            tag = f"{base_tag}__inPort1"

            arrays = {
                "eps": eps,
                "Ez_real": Ezr,
                "Ez_imag": Ezi,
                "src_mask": src,
                **(
                    {
                        "ports/ids": port_ids.astype(np.int32),
                        "ports/masks": port_masks.astype(np.float32),
                    }
                    if (port_ids is not None and port_masks is not None)
                    else {}
                ),
                "sparams/input_port": np.array(1, dtype=np.int32),
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/n_zigs": np.array(n_zigs, dtype=np.float32),
                "sparams/start_up": np.array(start_up, dtype=np.float32),
                "sparams/end_with_turn": np.array(end_with_turn, dtype=np.float32),
                "sparams/end_turn_deg": np.array(end_turn_deg, dtype=np.float32),
                "sparams/R_min_um": np.array(R_min_um, dtype=np.float32),
                "sparams/straight_x_um": np.array(straight_x_um, dtype=np.float32),
                "sparams/straight_y_um": np.array(straight_y_um, dtype=np.float32),
                "sparams/y0_um": np.array(y0_um, dtype=np.float32),
                "sparams/lead_x_um": np.array(lead_x_um, dtype=np.float32),
                "sparams/min_straight_input_um": np.array(min_straight_input_um, dtype=np.float32),
                "sparams/bend_angle_deg": np.array(bend_angle_deg, dtype=np.float32),
                "sparams/path_rotation_deg": np.array(path_rotation_deg, dtype=np.float32),
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
            meta = {"tag": tag, "dataset": dataset_name}
            buffer.append((arrays, meta))

            # write full shards
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

    # final partial shard
    if buffer:
        write_one_shard(buffer)
    if not save_index_every_shard:
        _atomic_write_index(index_path, index)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None, help="Output root (default: <repo>/Data/euler_bend_sweep)")
    ap.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    ap.add_argument("--dpml", type=float, default=DPML_DEFAULT)
    ap.add_argument("--crop-px", type=int, default=CROP_PX_DEFAULT, help="Non-PML crop size in pixels (square).")
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--n-procs", type=int, default=N_PROCS_DEFAULT)
    ap.add_argument("--wavelength-um", type=float, default=WAVELENGTH_DEFAULT)
    ap.add_argument("--decay-tol", type=float, default=1e-6)
    ap.add_argument("--include-turns", action="store_true", help="Include net-turn devices (adjacent-edge I/O) in the sweep.")
    ap.add_argument("--turn-prob", type=float, default=0.5, help="Probability a sampled geometry is a net-turn device (only if --include-turns).")
    ap.add_argument("--end-turn-deg", type=float, default=90.0, help="Final terminating turn angle for net-turn devices (degrees).")

    ap.add_argument("--shard-size", type=int, default=100, help="Samples per shard (1 geometry -> 1 sample).")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed for shards (smaller, slower).")
    ap.add_argument("--queue-max", type=int, default=64, help="Max queued temp files (bounds disk usage).")
    ap.add_argument("--index-every-shard", action="store_true", help="Write index.json after every shard (recommended).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if int(args.resolution) < 18:
        raise ValueError("--resolution must be >= 18 (requested by design; too-low resolution harms accuracy).")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR = Path(args.out_dir) if args.out_dir is not None else (repo_root / "Data" / "euler_bend_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = OUT_DIR / "tmp_geom"
    shards_dir = OUT_DIR / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Quantize dpml/cell to EXACT integer-pixel sizes (eliminates Meep rounding warnings).
    # Target crop is crop_px, so full = crop_px + 2*pml_px.
    pml_px = int(np.round(float(args.dpml) * float(args.resolution)))
    dpml_um = float(pml_px) / float(args.resolution)
    crop_px = int(args.crop_px)
    if crop_px <= 0:
        raise ValueError("--crop-px must be > 0")
    nx_full = crop_px + 2 * pml_px
    cell_um = float(nx_full) / float(args.resolution)
    if abs(dpml_um - float(args.dpml)) > (0.5 / float(args.resolution)):
        print(f"[euler_bend_sweep] NOTE: quantized dpml from {args.dpml} to {dpml_um} to align with integer pixels.")
    nx_full = int(np.round(cell_um * float(args.resolution)))
    ny_full = nx_full
    nx_crop = nx_full - 2 * pml_px
    ny_crop = ny_full - 2 * pml_px
    if (nx_crop, ny_crop) != (crop_px, crop_px):
        raise ValueError(
            f"Expected cropped ({crop_px},{crop_px}) but got ({nx_crop},{ny_crop}); check dpml/resolution/crop_px."
        )

    params = build_param_list_valid(
        args.n_geo,
        seed=int(args.seed),
        resolution=int(args.resolution),
        dpml=float(dpml_um),
        cell_um=float(cell_um),
        wavelength_um=float(args.wavelength_um),
        crop_px=int(crop_px),
        include_turns=bool(args.include_turns),
        turn_prob=float(args.turn_prob),
        end_turn_deg=float(args.end_turn_deg),
    )

    print(f"Unique geometries: {len(params)}")
    print(f"Total samples produced: {len(params)}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"TMP_DIR:   {tmp_dir}  (bounded by queue_max ~ {args.queue_max})")
    print(f"SHARDS:    {shards_dir}")
    print(f"n_procs:   {args.n_procs}")
    print(f"shard_size(samples): {args.shard_size}  => ~{math.ceil(len(params)/args.shard_size)} shards")
    print(f"compress:  {bool(args.compress)}")
    print(f"index-every-shard: {bool(args.index_every_shard)}")

    q: mp.Queue = mp.Queue(maxsize=args.queue_max)

    writer_p = mp.Process(
        target=shard_writer,
        args=(
            q,
            str(shards_dir),
            int(args.shard_size),
            bool(args.compress),
            "euler_bend",
            "index.json",
            bool(args.index_every_shard),
        ),
        daemon=True,
    )
    writer_p.start()

    # IMPORTANT FIX: pass sampled lam (from params) into the simulator.
    tasks = [
        (
            wg_width,
            n_zigs,
            start_up,
            end_with_turn,
            end_turn_deg,
            R_min,
            sx,
            sy,
            lead_x,
            min_straight_in,
            y0,
            bend_angle,
            rotation,
            float(lam),              # <-- FIXED (was args.wavelength_um)
            int(args.resolution),
            float(dpml_um),
            float(cell_um),
            int(crop_px),
            float(args.decay_tol),
            str(tmp_dir),
        )
        for (wg_width, n_zigs, start_up, end_with_turn, end_turn_deg, R_min, sx, sy, lead_x, min_straight_in, y0, bend_angle, rotation, lam) in params
    ]

    successes, failures = 0, 0
    with mp.Pool(processes=args.n_procs) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="FDTD -> tmp"):
            if status == "OK":
                q.put(("OK", payload))
                successes += 1
            else:
                failures += 1
                q.put(("ERR", payload))

    q.put(None)
    writer_p.join()

    # Cleanup tmp dir if empty
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
