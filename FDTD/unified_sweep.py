# unified_sweep.py
"""
Unified multi-device sweep for photonic dataset generation.

Generates training data for 9 device types:
  Rectangular domain (480x160 px): straight, taper, sbend, ybranch, directional_coupler, mmi
  Square domain (320x320 px): euler_bend, circular_bend, crossing

Each geometry is run at 5 fixed wavelengths (1.48-1.62 µm, C-band).
For multi-input devices, one input port is randomly selected per geometry.

Split: 80/10/10 by geometry_id (no leakage across wavelengths).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Import all device classes
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from straight_waveguide.straight import StraightWaveguide2D
from mmi.mmi import MMI2x2
from sbend.sbend import EulerSBend2D
from ybranch.ybranch import YBranch2D
from directional_coupler.directional_coupler import DirectionalCoupler2D
from taper.taper import TaperWaveguide2D
from euler_bend.euler_bend import EulerBend2D
from circular_bend.circular_bend import CircularBend2D
from crossing.crossing import UniformCrossing2D

from sweep_utils import latin_hypercube, quantize_01, assign_splits


# =============================================================================
# Mask Drawing Utilities
# =============================================================================
def _draw_thick_line_mask(ny: int, nx: int, x0: float, y0: float, x1: float, y1: float, thickness_px: int = 3) -> np.ndarray:
    """
    Draw a thick line segment mask from (x0, y0) to (x1, y1).
    Returns a float32 [ny, nx] mask with 1.0 where the line is and 0.0 elsewhere.
    """
    x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
    vx = x1f - x0f
    vy = y1f - y0f
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        # Point mask
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


def _meep_coord_to_cropped_px(coord: float, cell_size: float, dpml: float, resolution: int) -> float:
    """
    Convert Meep coordinate (centered at 0) to pixel index in cropped (non-PML) array.

    Meep: coord in [-cell_size/2, cell_size/2]
    Cropped interior: starts at -cell_size/2 + dpml (Meep coords)
    Pixel: (coord - interior_start) * resolution
    """
    interior_start = -cell_size / 2.0 + dpml
    return (coord - interior_start) * resolution


def _volume_to_line_px(vol, cell_x: float, cell_y: float, dpml: float, resolution: int) -> Dict[str, Tuple[float, float]]:
    """
    Convert a meep.Volume (line source or port) to pixel coordinates in the cropped interior.

    Returns dict with 'line_start_px' and 'line_end_px' as (x, y) tuples.
    """
    cx = float(vol.center.x)
    cy = float(vol.center.y)
    sx = float(vol.size.x)
    sy = float(vol.size.y)

    # Volume is a line segment
    if sx > sy:
        # Horizontal line
        x0 = cx - sx / 2.0
        x1 = cx + sx / 2.0
        y0 = y1 = cy
    else:
        # Vertical line
        y0 = cy - sy / 2.0
        y1 = cy + sy / 2.0
        x0 = x1 = cx

    # Convert to cropped pixel coordinates
    px_x0 = _meep_coord_to_cropped_px(x0, cell_x, dpml, resolution)
    px_x1 = _meep_coord_to_cropped_px(x1, cell_x, dpml, resolution)
    px_y0 = _meep_coord_to_cropped_px(y0, cell_y, dpml, resolution)
    px_y1 = _meep_coord_to_cropped_px(y1, cell_y, dpml, resolution)

    return {
        "line_start_px": (px_x0, px_y0),
        "line_end_px": (px_x1, px_y1),
    }


def get_device_masks(dev, device_type: str, input_port: int,
                     cell_x: float, cell_y: float, dpml: float, resolution: int,
                     ny: int, nx: int, thickness_px: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate source mask and port masks for any device type.

    Returns:
        src_mask: [ny, nx] float32 - mask for the active source
        port_ids: [n_ports] int32 - port numbers
        port_masks: [n_ports, ny, nx] float32 - mask for each port
    """

    # Device types with get_source_region_px / get_port_region_px methods
    if device_type in ["directional_coupler", "mmi"]:
        # Has 4 ports, uses crop_pml=True methods
        src_px = dev.get_source_region_px(input_port=input_port, crop_pml=True)
        src_mask = _draw_thick_line_mask(
            ny, nx,
            src_px["line_start_px"][0], src_px["line_start_px"][1],
            src_px["line_end_px"][0], src_px["line_end_px"][1],
            thickness_px=thickness_px,
        )

        port_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        port_masks = []
        for p in port_ids.tolist():
            pr = dev.get_port_region_px(p, crop_pml=True)
            pm = _draw_thick_line_mask(
                ny, nx,
                pr["line_start_px"][0], pr["line_start_px"][1],
                pr["line_end_px"][0], pr["line_end_px"][1],
                thickness_px=thickness_px,
            )
            port_masks.append(pm)
        port_masks = np.stack(port_masks, axis=0).astype(np.float32)

        return src_mask, port_ids, port_masks

    elif device_type == "ybranch":
        # Has 3 ports, uses crop_pml=True methods
        src_px = dev.get_source_region_px(input_port=input_port, crop_pml=True)
        src_mask = _draw_thick_line_mask(
            ny, nx,
            src_px["line_start_px"][0], src_px["line_start_px"][1],
            src_px["line_end_px"][0], src_px["line_end_px"][1],
            thickness_px=thickness_px,
        )

        port_ids = np.array([1, 2, 3], dtype=np.int32)
        port_masks = []
        for p in port_ids.tolist():
            pr = dev.get_port_region_px(p, crop_pml=True)
            pm = _draw_thick_line_mask(
                ny, nx,
                pr["line_start_px"][0], pr["line_start_px"][1],
                pr["line_end_px"][0], pr["line_end_px"][1],
                thickness_px=thickness_px,
            )
            port_masks.append(pm)
        port_masks = np.stack(port_masks, axis=0).astype(np.float32)

        return src_mask, port_ids, port_masks

    elif device_type == "crossing":
        # 4-port device with port_1..port_4 and src_vol_1..src_vol_4
        src_vols = {1: dev.src_vol_1, 2: dev.src_vol_2, 3: dev.src_vol_3, 4: dev.src_vol_4}
        src_vol = src_vols[int(input_port)]
        src_line = _volume_to_line_px(src_vol, cell_x, cell_y, dpml, resolution)
        src_mask = _draw_thick_line_mask(
            ny, nx,
            src_line["line_start_px"][0], src_line["line_start_px"][1],
            src_line["line_end_px"][0], src_line["line_end_px"][1],
            thickness_px=thickness_px,
        )

        port_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        port_vols = {1: dev.port_1, 2: dev.port_2, 3: dev.port_3, 4: dev.port_4}
        port_masks = []
        for p in port_ids.tolist():
            pr = _volume_to_line_px(port_vols[p], cell_x, cell_y, dpml, resolution)
            pm = _draw_thick_line_mask(
                ny, nx,
                pr["line_start_px"][0], pr["line_start_px"][1],
                pr["line_end_px"][0], pr["line_end_px"][1],
                thickness_px=thickness_px,
            )
            port_masks.append(pm)
        port_masks = np.stack(port_masks, axis=0).astype(np.float32)

        return src_mask, port_ids, port_masks

    else:
        # 2-port devices: straight, taper, sbend, euler_bend, circular_bend
        # These have port_in, port_out attributes
        # Source is offset from port by source_shift (typically 0.5 um)
        # Direction of shift is away from device center, along port normal

        source_shift = getattr(dev, 'source_shift', 0.5)  # Default 0.5 um

        if int(input_port) == 1:
            port_vol = dev.port_in
        else:
            port_vol = dev.port_out

        pcx = float(port_vol.center.x)
        pcy = float(port_vol.center.y)
        psx = float(port_vol.size.x)
        psy = float(port_vol.size.y)

        if psy >= psx:
            # Vertical line → port is on left/right edge → shift along x
            # Shift away from center (upstream of port)
            sign = -1.0 if pcx < 0 else +1.0
            src_cx = pcx + sign * source_shift
            src_cy = pcy
            src_half_span = psy / 2.0
            src_x0 = src_x1 = src_cx
            src_y0 = src_cy - src_half_span
            src_y1 = src_cy + src_half_span
        else:
            # Horizontal line → port is on top/bottom edge → shift along y
            sign = -1.0 if pcy < 0 else +1.0
            src_cx = pcx
            src_cy = pcy + sign * source_shift
            src_half_span = psx / 2.0
            src_y0 = src_y1 = src_cy
            src_x0 = src_cx - src_half_span
            src_x1 = src_cx + src_half_span

        # Convert to cropped pixel coordinates
        src_px_x0 = _meep_coord_to_cropped_px(src_x0, cell_x, dpml, resolution)
        src_px_x1 = _meep_coord_to_cropped_px(src_x1, cell_x, dpml, resolution)
        src_px_y0 = _meep_coord_to_cropped_px(src_y0, cell_y, dpml, resolution)
        src_px_y1 = _meep_coord_to_cropped_px(src_y1, cell_y, dpml, resolution)

        src_mask = _draw_thick_line_mask(
            ny, nx,
            src_px_x0, src_px_y0,
            src_px_x1, src_px_y1,
            thickness_px=thickness_px,
        )

        # Port masks for port 1 (port_in) and port 2 (port_out)
        port_ids = np.array([1, 2], dtype=np.int32)
        port_masks = []

        port_in_line = _volume_to_line_px(dev.port_in, cell_x, cell_y, dpml, resolution)
        pm1 = _draw_thick_line_mask(
            ny, nx,
            port_in_line["line_start_px"][0], port_in_line["line_start_px"][1],
            port_in_line["line_end_px"][0], port_in_line["line_end_px"][1],
            thickness_px=thickness_px,
        )
        port_masks.append(pm1)

        port_out_line = _volume_to_line_px(dev.port_out, cell_x, cell_y, dpml, resolution)
        pm2 = _draw_thick_line_mask(
            ny, nx,
            port_out_line["line_start_px"][0], port_out_line["line_start_px"][1],
            port_out_line["line_end_px"][0], port_out_line["line_end_px"][1],
            thickness_px=thickness_px,
        )
        port_masks.append(pm2)

        port_masks = np.stack(port_masks, axis=0).astype(np.float32)

        return src_mask, port_ids, port_masks


# =============================================================================
# Data Augmentation (Dihedral Group D4 - 8 transforms)
# =============================================================================
# The dihedral group D4 has 8 elements: 4 rotations × 2 flip states
# This gives us 8× the training data from each simulation for free.

# D4 group (8 elements) — full augmentation for square grids
AUGMENT_TRANSFORMS_D4 = [
    # (flip_horizontal, rotate_k) where rotate_k is number of 90° CCW rotations
    (False, 0),  # identity
    (False, 1),  # 90° CCW
    (False, 2),  # 180°
    (False, 3),  # 270° CCW
    (True, 0),   # horizontal flip
    (True, 1),   # horizontal flip + 90° CCW
    (True, 2),   # horizontal flip + 180° (= vertical flip)
    (True, 3),   # horizontal flip + 270° CCW
]
AUGMENT_NAMES_D4 = [
    "orig", "rot90", "rot180", "rot270",
    "flip_h", "flip_h_rot90", "flip_v", "flip_h_rot270"
]

# D2 subgroup (4 elements) — safe for rectangular grids (no 90°/270° rotations)
AUGMENT_TRANSFORMS_D2 = [
    (False, 0),  # identity
    (False, 2),  # 180°
    (True, 0),   # horizontal flip
    (True, 2),   # vertical flip (= horizontal flip + 180°)
]
AUGMENT_NAMES_D2 = [
    "orig", "rot180", "flip_h", "flip_v"
]


def apply_transform_2d(arr: np.ndarray, flip_h: bool, rot_k: int) -> np.ndarray:
    """Apply a D4 transform to a 2D array [ny, nx]."""
    if flip_h:
        arr = np.fliplr(arr)
    if rot_k != 0:
        arr = np.rot90(arr, k=rot_k)
    return arr.copy()


def apply_transform_3d(arr: np.ndarray, flip_h: bool, rot_k: int) -> np.ndarray:
    """Apply a D4 transform to a 3D array [n, ny, nx] (e.g., port_masks)."""
    result = []
    for i in range(arr.shape[0]):
        result.append(apply_transform_2d(arr[i], flip_h, rot_k))
    return np.stack(result, axis=0)


def augment_sample(data: Dict[str, np.ndarray], flip_h: bool, rot_k: int) -> Dict[str, np.ndarray]:
    """
    Apply a D4 transform to all spatial arrays in a sample.

    Transforms: eps, Ez_real, Ez_imag, src_mask, port_masks
    Keeps unchanged: port_ids, S-parameters, device params, scalars
    """
    result = {}

    for key, val in data.items():
        if key in ("eps", "Ez_real", "Ez_imag", "src_mask"):
            # 2D spatial arrays [ny, nx]
            result[key] = apply_transform_2d(val, flip_h, rot_k)
        elif key == "port_masks":
            # 3D array [n_ports, ny, nx]
            result[key] = apply_transform_3d(val, flip_h, rot_k)
        elif key in ("nx", "ny", "Lx_um", "Ly_um"):
            # Swap nx/ny and Lx/Ly for 90° and 270° rotations
            if rot_k in (1, 3):
                swap_key = {"nx": "ny", "ny": "nx", "Lx_um": "Ly_um", "Ly_um": "Lx_um"}
                result[key] = data[swap_key[key]]
            else:
                result[key] = val
        else:
            # Keep everything else unchanged (S-params, port_ids, device params, etc.)
            result[key] = val

    return result


# =============================================================================
# Domain Configuration
# =============================================================================
DOMAIN_RECTANGULAR = "rectangular"
DOMAIN_SQUARE = "square"
DEVICE_DOMAIN = {
    "straight": DOMAIN_RECTANGULAR,
    "taper": DOMAIN_RECTANGULAR,
    "sbend": DOMAIN_RECTANGULAR,
    "ybranch": DOMAIN_RECTANGULAR,
    "directional_coupler": DOMAIN_RECTANGULAR,
    "mmi": DOMAIN_RECTANGULAR,
    "euler_bend": DOMAIN_RECTANGULAR,
    "circular_bend": DOMAIN_RECTANGULAR,
    "crossing": DOMAIN_RECTANGULAR,
}


# =============================================================================
# Configuration
# =============================================================================
WAVELENGTHS = [1.48, 1.53, 1.58, 1.62]

# Default counts for ~14,300 total geometries (9 device types).
# Allocation weighted by parameter-space dimensionality and physics complexity.
DEFAULT_COUNTS = {
    "straight": 200,              # 2D param space (baseline only)
    "taper": 800,                 # 3D param space
    "mmi": 2500,                  # 5D param space (rich multimode interference)
    "sbend": 800,                 # 3D param space
    "ybranch": 1500,              # 5D param space
    "directional_coupler": 2500,  # 5D param space
    "euler_bend": 1500,           # 2D param space (square domain, D4)
    "circular_bend": 1500,        # 2D param space (square domain, D4)
    "crossing": 3000,             # 2D param space (4-port, square domain)
}

# Parameter ranges per device (9 device types).
# All values should be multiples of 1/(2*resolution) for pixel-grid alignment.
# At resolution=20: quantum = 1/40 = 0.025 µm.
# Rectangular domain: 480×160 interior at res=20 (24.0 × 8.0 µm).
# Square domain: 320×320 interior at res=20 (16.0 × 16.0 µm).
PARAM_RANGES = {
    "straight": {
        "wg_width_um": (0.40, 0.575),       # 8/20 – 11.5/20 (≥8 px per wg width)
        "dev_length_um": (6.0, 18.0),        # fit in 24 µm X
    },
    "taper": {
        "wg_width_in": (0.40, 0.575),        # narrow end
        "wg_width_out": (0.60, 2.0),         # wide end
        "taper_length_um": (3.0, 15.0),      # fit in 24 µm X
    },
    "mmi": {
        "wg_width_um":    (0.40, 0.575),     # 8/20 – 11.5/20
        "mmi_width_um":   (2.5, 5.5),        # fit in 8 µm Y
        "mmi_length_um":  (8.0, 15.0),       # fit in 24 µm X
        "taper_width_um": (0.575, 1.5),      # 11.5/20 – 1.5
        "taper_length_um":(1.0, 3.0),        # taper length per side
    },
    "sbend": {
        "wg_width_um": (0.40, 0.575),        # 8/20 – 11.5/20
        "lateral_offset_um": (2.0, 5.5),     # fit in 8 µm Y
        "R_min_um": (3.0, 7.0),              # reduce X extent
    },
    "ybranch": {
        "wg_width_um": (0.40, 0.575),        # 8/20 – 11.5/20
        "l_junction_um": (1.0, 3.0),
        "l_bend_um": (4.0, 7.0),
        "h_bend_um": (0.575, 2.5),           # fit in 8 µm Y
        "l_out_um": (1.0, 4.0),              # fit in 24 µm X
    },
    "directional_coupler": {
        "wg_width_um": (0.40, 0.575),        # 8/20 – 11.5/20
        "gap_um": (0.10, 0.35),              # 2/20 – 7/20 (≥2 px across gap)
        "wg_length_um": (5.0, 8.0),          # fit in 24 µm X
        "bend_length_um": (4.0, 6.0),
        "lead_extra_gap_um": (0.825, 2.0),   # fit in 8 µm Y
    },
    "euler_bend": {
        "wg_width": (0.40, 0.575),           # 8/20 – 11.5/20
        "R_min_um": (2.0, 3.5),              # fit in 8 µm rect height
    },
    "circular_bend": {
        "wg_width": (0.40, 0.575),           # 8/20 – 11.5/20
        "bend_radius_um": (2.0, 3.5),        # fit in 8 µm rect height
    },
    "crossing": {
        "wg_width_h": (0.40, 0.575),         # horizontal WG width
        "wg_width_v": (0.40, 0.575),         # vertical WG width
    },
}

# Number of input ports per device type (9 device types)
INPUT_PORTS = {
    "straight": [1, 2],
    "taper": [1, 2],
    "mmi": [1, 2],
    "sbend": [1, 2],
    "ybranch": [1, 2, 3],              # Port 1 = splitter, Ports 2/3 = combiner
    "directional_coupler": [1, 2],     # Symmetric, use 1 or 2
    "euler_bend": [1, 2],
    "circular_bend": [1, 2],
    "crossing": [1, 2, 3, 4],          # 4-port device
}


# =============================================================================
# Sampling utilities
# =============================================================================
def sample_params(device_type: str, n_geo: int, seed: int, resolution: int = 14) -> List[Dict[str, float]]:
    """Sample parameters for a device type using Latin Hypercube Sampling.

    Values are quantized to the half-pixel grid (1/(2*resolution)) so that
    every parameter change maps to a distinct difference in the epsilon map.
    """
    ranges = PARAM_RANGES[device_type]
    n_dims = len(ranges)
    param_names = list(ranges.keys())

    # Half-pixel quantum: ensures each step shifts an edge by ≥ half a pixel,
    # producing a measurably different subpixel-averaged epsilon.
    q = 1.0 / (2.0 * resolution)

    u = latin_hypercube(n_geo, d=n_dims, seed=seed)

    params_list = []
    for i in range(n_geo):
        p = {}
        for j, name in enumerate(param_names):
            lo, hi = ranges[name]
            val = lo + u[i, j] * (hi - lo)
            # Snap to half-pixel grid
            val = round(val / q) * q
            val = max(lo, min(hi, val))
            p[name] = float(val)
        params_list.append(p)

    return params_list


def make_geom_id(device_type: str, seed: Optional[int] = None) -> str:
    """Generate a unique geometry ID.

    If seed is provided, produces a deterministic ID (needed for resume support).
    Otherwise falls back to random UUID.
    """
    if seed is not None:
        import hashlib
        h = hashlib.sha256(f"{device_type}_{seed}".encode()).hexdigest()[:12]
        return f"{device_type}_{h}"
    return f"{device_type}_{uuid.uuid4().hex[:12]}"


# =============================================================================
# Device builders
# =============================================================================
def build_device(device_type: str, params: Dict[str, float], wavelength_um: float,
                 resolution: int, dpml: float, cell_x: float, cell_y: float,
                 crop_x_px: int, crop_y_px: int):
    """Build a device instance based on type and parameters."""

    common_kwargs = {
        "wavelength_um": wavelength_um,
        "resolution": resolution,
    }

    if device_type == "straight":
        return StraightWaveguide2D(
            wg_width=params["wg_width_um"],
            dev_length_um=params["dev_length_um"],
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    elif device_type == "mmi":
        return MMI2x2(
            wg_width_um=params["wg_width_um"],
            mmi_width_um=params["mmi_width_um"],
            mmi_length_um=params["mmi_length_um"],
            taper_width_um=params["taper_width_um"],
            taper_length_um=params["taper_length_um"],
            dpml=dpml,
            crop_x_px=crop_x_px,
            crop_y_px=crop_y_px,
            quantize_grid=True,
            fit_margin_um=0.5,
            **common_kwargs,
        )

    elif device_type == "sbend":
        return EulerSBend2D(
            wg_width=params["wg_width_um"],
            lateral_offset_um=params["lateral_offset_um"],
            R_min_um=params["R_min_um"],
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    elif device_type == "ybranch":
        return YBranch2D(
            wg_width_um=params["wg_width_um"],
            l_junction_um=params["l_junction_um"],
            l_bend_um=params["l_bend_um"],
            h_bend_um=params["h_bend_um"],
            l_out_um=params["l_out_um"],
            dpml=dpml,
            cell_x_um=cell_x,
            cell_y_um=cell_y,
            quantize_grid=False,
            fit_margin_um=0.5,
            **common_kwargs,
        )

    elif device_type == "directional_coupler":
        return DirectionalCoupler2D(
            wg_width_um=params["wg_width_um"],
            gap_um=params["gap_um"],
            wg_length_um=params["wg_length_um"],
            bend_length_um=params["bend_length_um"],
            lead_extra_gap_um=params["lead_extra_gap_um"],
            dpml=dpml,
            crop_x_px=crop_x_px,
            crop_y_px=crop_y_px,
            quantize_grid=True,
            fit_margin_um=0.5,
            **common_kwargs,
        )

    elif device_type == "taper":
        return TaperWaveguide2D(
            wg_width_in=params["wg_width_in"],
            wg_width_out=params["wg_width_out"],
            taper_length_um=params["taper_length_um"],
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    elif device_type == "euler_bend":
        return EulerBend2D(
            wg_width=params["wg_width"],
            R_min_um=params["R_min_um"],
            bend_angle_deg=90.0,
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    elif device_type == "circular_bend":
        return CircularBend2D(
            wg_width=params["wg_width"],
            bend_radius_um=params["bend_radius_um"],
            bend_angle_deg=90.0,
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    elif device_type == "crossing":
        return UniformCrossing2D(
            wg_width_h=params["wg_width_h"],
            wg_width_v=params["wg_width_v"],
            dpml=dpml,
            cell_x=cell_x,
            cell_y=cell_y,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown device type: {device_type}")


# =============================================================================
# Simulation runner
# =============================================================================
def run_sim_and_extract(dev, device_type: str, input_port: int, decay_tol: float):
    """
    Run simulation and extract results.
    Returns: (eps, Ez, S_dict, cell_size)
    """
    import meep as meep_mp
    from utils import get_mode_alpha_2dir, pick_in_out_from_alpha

    if device_type in ["straight", "sbend", "taper", "euler_bend", "circular_bend"]:
        # 2-port devices using _build_sim_single
        toward = {1: +1, 2: -1}
        sim, (m1, m2), dft, _fcen = dev._build_sim_single(input_port=int(input_port), df_frac=0.1)
        sim.run(until_after_sources=meep_mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps = sim.get_epsilon().T.astype(np.float32)
        Ez = sim.get_dft_array(dft, meep_mp.Ez, 0).T.astype(np.complex64)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=meep_mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=meep_mp.NO_PARITY)

        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=0, dir_minus=1)
        a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=0, dir_minus=1)

        if int(input_port) == 1:
            S = {"S11": b1_out / a1_in, "S21": b2_out / a1_in}
        else:
            S = {"S22": b2_out / a2_in, "S12": b1_out / a2_in}

        sim.reset_meep()
        return eps, Ez, S, (dev.cell_x, dev.cell_y)

    elif device_type == "crossing":
        # 4-port device — call run_sim directly (already returns correct format)
        eps, Ez, S, cell_size = dev.run_sim(input_port=input_port, decay_tol=decay_tol)
        return eps, Ez, S, cell_size

    elif device_type == "ybranch":
        # 3-port device: port 1 = splitter, ports 2/3 = combiner
        eps, Ez, _Hx, _Hy, S_raw, cell_size = dev.run_sim(input_port=input_port, decay_tol=decay_tol)
        S = {}
        for p in (1, 2, 3):
            S[f"S{p}{input_port}"] = S_raw[(p, input_port)]
        return eps, Ez, S, cell_size

    elif device_type in ["directional_coupler", "mmi"]:
        # 4-port device
        eps, Ez, _Hx, _Hy, S_raw, cell_size = dev.run_sim(input_port=input_port, decay_tol=decay_tol)
        S = {}
        for p in (1, 2, 3, 4):
            S[f"S{p}{input_port}"] = S_raw[(p, input_port)]
        return eps, Ez, S, cell_size

    else:
        raise ValueError(f"Unknown device type: {device_type}")


# =============================================================================
# Worker function for multiprocessing
# =============================================================================
def worker(task: Tuple) -> Tuple[str, Any]:
    """
    Worker function for parallel simulation.
    Returns: ("OK", result_dict) or ("ERR", error_string)
    """
    (device_type, geom_id, params, input_port, wavelength_um, split,
     resolution, dpml, cell_x, cell_y, crop_x_px, crop_y_px, decay_tol, tmp_dir) = task

    tmp_path = Path(tmp_dir) / f"{geom_id}_lam{wavelength_um:.4f}.npz"

    try:
        dev = build_device(device_type, params, wavelength_um, resolution, dpml, cell_x, cell_y, crop_x_px, crop_y_px)
        eps_full, Ez_full, S, cell_size = run_sim_and_extract(dev, device_type, input_port, decay_tol)

        # Crop PML region - only save the interior (non-PML) window
        pml_px = int(round(float(dpml) * float(resolution)))
        if pml_px > 0:
            eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
            Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px]
            # Update cell size to interior dimensions
            interior_x = cell_size[0] - 2.0 * dpml
            interior_y = cell_size[1] - 2.0 * dpml
        else:
            eps = eps_full
            Ez = Ez_full
            interior_x = cell_size[0]
            interior_y = cell_size[1]

        ny, nx = eps.shape

        # Quality checks
        if np.any(np.isnan(Ez)) or np.any(np.isinf(Ez)):
            return ("ERR", f"{geom_id}_{wavelength_um:.4f}: NaN/Inf in Ez field")
        if np.any(np.isnan(eps)) or np.any(np.isinf(eps)):
            return ("ERR", f"{geom_id}_{wavelength_um:.4f}: NaN/Inf in epsilon")
        ez_max = float(np.max(np.abs(Ez)))
        if ez_max < 1e-6:
            return ("ERR", f"{geom_id}_{wavelength_um:.4f}: near-zero field (|Ez|_max={ez_max:.2e})")
        sparam_power = sum(abs(v)**2 for v in S.values())
        if sparam_power > 1.5:
            print(f"[QC WARNING] {geom_id}_{wavelength_um:.4f}: S-param power={sparam_power:.3f} > 1.0")

        # Generate source and port masks (using cropped interior coordinates)
        src_mask, port_ids, port_masks = get_device_masks(
            dev, device_type, input_port,
            cell_size[0], cell_size[1], dpml, resolution,
            ny, nx, thickness_px=3
        )

        # Save to temp file (non-PML interior only)
        np.savez_compressed(
            tmp_path,
            geometry_id=np.array(geom_id),
            device=np.array(device_type),
            split=np.array(split),
            input_port=np.int32(input_port),
            wavelength_um=np.float32(wavelength_um),
            # Grid info (interior only, no PML)
            nx=np.int32(nx),
            ny=np.int32(ny),
            Lx_um=np.float32(interior_x),
            Ly_um=np.float32(interior_y),
            dx_um=np.float32(1.0 / resolution),
            dy_um=np.float32(1.0 / resolution),
            resolution=np.int32(resolution),
            # Field data (interior only)
            eps=eps.astype(np.float32),
            Ez_real=Ez.real.astype(np.float32),
            Ez_imag=Ez.imag.astype(np.float32),
            # Source and port masks
            src_mask=src_mask.astype(np.float32),
            port_ids=port_ids.astype(np.int32),
            port_masks=port_masks.astype(np.float32),
            # S-parameters
            **{f"sparams/{k}_real": np.float32(np.real(v)) for k, v in S.items()},
            **{f"sparams/{k}_imag": np.float32(np.imag(v)) for k, v in S.items()},
            **{f"params/{k}": np.float32(v) for k, v in params.items()},
        )

        return ("OK", str(tmp_path))

    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return ("ERR", f"{geom_id}_{wavelength_um:.4f}: {repr(e)}")


# =============================================================================
# Shard writer (streaming consolidation with augmentation)
# =============================================================================
def shard_writer_process(q: mp.Queue, shards_dir: str, shard_size: int,
                         compress: bool, index_every_shard: bool,
                         augment: bool = True, resume: bool = False):
    """
    Background process that consolidates temp files into shards.

    If augment=True, applies D4 (8×) for square grids or D2 (4×) for rectangular grids.
    Domain shape is detected per-sample from nx vs ny.
    If resume=True, loads existing index and continues shard numbering.
    """
    shards_path = Path(shards_dir)
    shards_path.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    index_path = shards_path / "index.json"

    # Resume: load existing index and find next shard ID
    shard_id = 0
    if resume and index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # Find max shard ID from existing entries
        for entry in index:
            sname = entry.get("shard", "")
            # Parse shard_XXXXX.npz
            try:
                sid = int(sname.replace("shard_", "").replace(".npz", ""))
                shard_id = max(shard_id, sid + 1)
            except (ValueError, AttributeError):
                pass
        print(f"[shard_writer] Resume: loaded {len(index)} existing entries, next shard_id={shard_id}")

    # Buffer holds (data_dict, meta_dict) tuples ready for writing
    buffer: List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = []

    def write_shard():
        nonlocal shard_id, buffer, index
        if not buffer:
            return

        shard_name = f"shard_{shard_id:05d}.npz"
        shard_path = shards_path / shard_name

        save_dict = {}
        for slot, (data, meta) in enumerate(buffer[:shard_size]):
            prefix = f"s{slot}/"
            for key, val in data.items():
                save_dict[prefix + key] = val

            index.append({
                "geometry_id": meta["geometry_id"],
                "device": meta["device"],
                "split": meta["split"],
                "wavelength_um": meta["wavelength_um"],
                "augment": meta.get("augment", "orig"),
                "shard": shard_name,
                "slot": slot,
            })

        if compress:
            np.savez_compressed(shard_path, **save_dict)
        else:
            np.savez(shard_path, **save_dict)

        buffer = buffer[shard_size:]
        shard_id += 1

        if index_every_shard:
            tmp_idx = index_path.with_suffix(".tmp")
            with open(tmp_idx, "w") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp_idx, index_path)

    while True:
        msg = q.get()
        if msg is None:
            break

        status, payload = msg
        if status != "OK":
            continue

        # Load data from temp file and apply augmentations
        tmp_path = Path(payload)
        try:
            with np.load(tmp_path, allow_pickle=True) as f:
                # Extract base metadata
                base_meta = {
                    "geometry_id": str(f["geometry_id"].item()),
                    "device": str(f["device"].item()),
                    "split": str(f["split"].item()),
                    "wavelength_um": float(f["wavelength_um"]),
                }

                # Load all data into dict
                data = {key: f[key] for key in f.files}

            # Apply augmentations (or just original if augment=False)
            # Detect domain shape per-sample: square grids get D4, rectangular get D2
            sample_nx = int(data["nx"])
            sample_ny = int(data["ny"])
            is_rect = sample_nx != sample_ny
            if augment and is_rect:
                transforms_to_apply = list(zip(AUGMENT_TRANSFORMS_D2, AUGMENT_NAMES_D2))
            elif augment:
                transforms_to_apply = list(zip(AUGMENT_TRANSFORMS_D4, AUGMENT_NAMES_D4))
            else:
                transforms_to_apply = [((False, 0), "orig")]

            for (flip_h, rot_k), aug_name in transforms_to_apply:
                # Apply transform to spatial arrays
                aug_data = augment_sample(data, flip_h, rot_k)

                # Create metadata for this augmented sample
                aug_meta = {**base_meta, "augment": aug_name}

                buffer.append((aug_data, aug_meta))

                if len(buffer) >= shard_size:
                    write_shard()

            # Clean up temp file after all augmentations extracted
            try:
                tmp_path.unlink()
            except Exception:
                pass

        except Exception as e:
            print(f"Error processing {tmp_path}: {e}")
            try:
                tmp_path.unlink()
            except Exception:
                pass

    # Flush remaining
    while buffer:
        write_shard()

    # Final index write
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified multi-device photonic sweep")

    # Output
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <repo>/Data/unified_sweep)")

    # Geometry counts (can scale all at once or individually)
    parser.add_argument("--n-geo", type=int, default=14300,
                        help="Total unique geometries (default: 14300)")
    parser.add_argument("--n-straight", type=int, default=None)
    parser.add_argument("--n-taper", type=int, default=None)
    parser.add_argument("--n-mmi", type=int, default=None)
    parser.add_argument("--n-sbend", type=int, default=None)
    parser.add_argument("--n-ybranch", type=int, default=None)
    parser.add_argument("--n-directional-coupler", type=int, default=None)
    parser.add_argument("--n-euler-bend", type=int, default=None)
    parser.add_argument("--n-circular-bend", type=int, default=None)
    parser.add_argument("--n-crossing", type=int, default=None)

    # Simulation params
    parser.add_argument("--resolution", type=int, default=20,
                        help="Simulation resolution (pixels/µm)")
    parser.add_argument("--dpml", type=float, default=1.0,
                        help="PML thickness (µm)")
    parser.add_argument("--crop-x-px", type=int, default=480,
                        help="Non-PML interior width in pixels for rectangular domain")
    parser.add_argument("--crop-y-px", type=int, default=160,
                        help="Non-PML interior height in pixels for rectangular domain")
    parser.add_argument("--crop-sq-px", type=int, default=320,
                        help="Non-PML interior size in pixels for square domain")
    parser.add_argument("--decay-tol", type=float, default=1e-5,
                        help="DFT decay tolerance")

    # Parallelization
    parser.add_argument("--n-procs", type=int, default=24,
                        help="Number of parallel workers")
    parser.add_argument("--queue-max", type=int, default=128,
                        help="Max queue size for shard writer")

    # Sharding
    parser.add_argument("--shard-size", type=int, default=100,
                        help="Samples per shard")
    parser.add_argument("--compress", action="store_true",
                        help="Use compressed npz")
    parser.add_argument("--index-every-shard", action="store_true",
                        help="Write index after each shard")

    # Augmentation (disabled by default - do augmentation during training instead)
    parser.add_argument("--augment", action="store_true",
                        help="Enable D4 augmentation during generation (default: off, do during training)")

    # Resume support
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing index: skip already-completed geometry+wavelength combos")

    # Seeds
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)

    args = parser.parse_args()

    # Set thread limits
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Compute cell sizes from crop dimensions + PML for both domains
    pml_px = int(round(float(args.dpml) * float(args.resolution)))
    dpml_q = float(pml_px) / float(args.resolution)
    args.dpml = dpml_q  # use quantized dpml

    # Rectangular domain (for elongated devices)
    rect_cell_x = float(args.crop_x_px + 2 * pml_px) / float(args.resolution)
    rect_cell_y = float(args.crop_y_px + 2 * pml_px) / float(args.resolution)

    # Square domain (for bends/crossings)
    sq_cell = float(args.crop_sq_px + 2 * pml_px) / float(args.resolution)

    # Store both domain configs
    domain_configs = {
        DOMAIN_RECTANGULAR: {
            "cell_x": rect_cell_x, "cell_y": rect_cell_y,
            "crop_x_px": args.crop_x_px, "crop_y_px": args.crop_y_px,
        },
        DOMAIN_SQUARE: {
            "cell_x": sq_cell, "cell_y": sq_cell,
            "crop_x_px": args.crop_sq_px, "crop_y_px": args.crop_sq_px,
        },
    }

    # Setup paths
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "Data" / "unified_sweep")
    tmp_dir = out_dir / "tmp"
    shards_dir = out_dir / "shards"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Determine device counts
    total_requested = args.n_geo
    counts = {}

    # Check for individual overrides
    overrides = {
        "straight": args.n_straight,
        "taper": args.n_taper,
        "mmi": args.n_mmi,
        "sbend": args.n_sbend,
        "ybranch": args.n_ybranch,
        "directional_coupler": args.n_directional_coupler,
        "euler_bend": args.n_euler_bend,
        "circular_bend": args.n_circular_bend,
        "crossing": args.n_crossing,
    }

    if any(v is not None for v in overrides.values()):
        # Use explicit counts
        for dtype, val in overrides.items():
            counts[dtype] = val if val is not None else 0
    else:
        # Scale from defaults to match total
        default_total = sum(DEFAULT_COUNTS.values())
        scale = total_requested / default_total
        for dtype, default_n in DEFAULT_COUNTS.items():
            counts[dtype] = max(1, int(round(default_n * scale)))

    total_geo = sum(counts.values())
    total_sims = total_geo * len(WAVELENGTHS)
    augment = args.augment  # Default: off (do augmentation during training instead)

    print(f"=" * 60)
    print(f"Unified Multi-Device Photonic Sweep (9 device types)")
    print(f"=" * 60)
    print(f"Output directory: {out_dir}")
    print(f"Temp directory:   {tmp_dir}")
    print(f"Shards directory: {shards_dir}")
    print()
    print(f"Device counts:")
    for dtype, n in counts.items():
        domain = DEVICE_DOMAIN[dtype]
        tag = "rect" if domain == DOMAIN_RECTANGULAR else "sq"
        print(f"  {dtype:25s}: {n:6d} geometries  [{tag}]")
    print(f"  {'TOTAL':25s}: {total_geo:6d} geometries")
    print()
    print(f"Wavelengths: {WAVELENGTHS}")
    print(f"Total simulations: {total_sims}")
    print(f"Augmentation: {'enabled (D4/D2 per domain)' if augment else 'disabled'}")
    print()
    q_half = 1.0 / (2.0 * args.resolution)
    print(f"Simulation params:")
    print(f"  resolution: {args.resolution} px/µm (dx = {1.0/args.resolution:.4f} µm)")
    print(f"  param quantum: {q_half:.5f} µm (half-pixel)")
    print(f"  dpml:       {args.dpml:.4f} µm ({pml_px} px)")
    print(f"  rect domain: {args.crop_x_px} x {args.crop_y_px} px ({rect_cell_x:.1f} x {rect_cell_y:.1f} µm cell)")
    print(f"  sq domain:   {args.crop_sq_px} x {args.crop_sq_px} px ({sq_cell:.1f} x {sq_cell:.1f} µm cell)")
    print(f"  decay_tol:  {args.decay_tol}")
    print()
    print(f"Parallelization: {args.n_procs} workers")
    print(f"Shard size: {args.shard_size} samples/shard")
    print(f"=" * 60)

    # Generate all tasks
    rng = np.random.default_rng(args.seed)
    all_tasks = []
    geom_meta = []  # For geometries.jsonl

    seed_offset = 0
    for device_type, n_geo in counts.items():
        if n_geo == 0:
            continue

        # Sample parameters (quantized to half-pixel grid)
        params_list = sample_params(device_type, n_geo, seed=args.seed + seed_offset, resolution=args.resolution)
        seed_offset += 1

        # Assign splits (80/10/10)
        splits = assign_splits(n_geo, seed=args.split_seed + seed_offset)
        seed_offset += 1

        for i, params in enumerate(params_list):
            geom_id = make_geom_id(device_type, seed=args.seed * 100000 + seed_offset * 10000 + i)
            split = splits[i]

            # Choose input port for this geometry
            possible_ports = INPUT_PORTS[device_type]
            input_port = int(rng.choice(possible_ports))

            # Store geometry metadata
            geom_meta.append({
                "geometry_id": geom_id,
                "device": device_type,
                "split": split,
                "input_port": input_port,
                **params,
            })

            # Get domain config for this device type
            domain = DEVICE_DOMAIN[device_type]
            dc = domain_configs[domain]

            # Create task for each wavelength
            for lam in WAVELENGTHS:
                task = (
                    device_type, geom_id, params, input_port, lam, split,
                    args.resolution, args.dpml, dc["cell_x"], dc["cell_y"],
                    dc["crop_x_px"], dc["crop_y_px"], args.decay_tol, str(tmp_dir)
                )
                all_tasks.append(task)

    # Resume: filter out already-completed tasks
    if args.resume:
        index_path = shards_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                existing_index = json.load(f)
            # Build set of (geometry_id, wavelength_um_rounded) already done
            done = set()
            for entry in existing_index:
                gid = entry.get("geometry_id", "")
                wl = round(float(entry.get("wavelength_um", 0)), 4)
                done.add((gid, wl))
            before = len(all_tasks)
            all_tasks = [
                t for t in all_tasks
                if (t[1], round(float(t[4]), 4)) not in done  # t[1]=geom_id, t[4]=wavelength_um
            ]
            print(f"Resume: {before - len(all_tasks)} tasks already done, {len(all_tasks)} remaining")
        else:
            print("Resume: no existing index found, starting fresh")

    # Shuffle tasks for better load balancing
    rng.shuffle(all_tasks)

    # Write geometries.jsonl (always overwrite — deterministic from seeds)
    geom_path = out_dir / "geometries.jsonl"
    with open(geom_path, "w") as f:
        for row in geom_meta:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(geom_meta)} geometry records to {geom_path}")

    # Start shard writer process
    q = mp.Queue(maxsize=args.queue_max)
    writer_proc = mp.Process(
        target=shard_writer_process,
        args=(q, str(shards_dir), args.shard_size, args.compress, args.index_every_shard,
              augment, args.resume),
        daemon=True,
    )
    writer_proc.start()

    # Run simulations with multiprocessing
    successes, failures = 0, 0
    error_log = []

    with mp.Pool(processes=args.n_procs) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, all_tasks),
                                     total=len(all_tasks), desc="Running FDTD"):
            if status == "OK":
                q.put(("OK", payload))
                successes += 1
            else:
                failures += 1
                error_log.append(payload)
                q.put(("ERR", payload))

    # Signal writer to finish
    q.put(None)
    writer_proc.join()

    # Write error log if any
    if error_log:
        err_path = out_dir / "errors.txt"
        with open(err_path, "w") as f:
            for err in error_log:
                f.write(err + "\n")
        print(f"Wrote {len(error_log)} errors to {err_path}")

    # Cleanup tmp dir
    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    print()
    print(f"=" * 60)
    print(f"COMPLETE")
    print(f"=" * 60)
    print(f"Simulations: {successes} succeeded, {failures} failed")
    aug_str = "D4/D2 augmentation per domain" if augment else "no augmentation"
    print(f"Augmentation: {aug_str}")
    print(f"Shards:    {shards_dir / 'shard_*.npz'}")
    print(f"Index:     {shards_dir / 'index.json'}")
    print(f"Metadata:  {geom_path}")


if __name__ == "__main__":
    main()
