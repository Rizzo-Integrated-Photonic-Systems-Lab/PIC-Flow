"""Parameter bounds for each device family — no Meep / FDTD imports.

Shared by `unified_sweep.py` (dataset generation) and inference tools that only need
ranges for validation (`tools/predict_parametric_device.py`) and port lists (`INPUT_PORTS`)
for notebooks / UI.
"""

from __future__ import annotations

# Parameter ranges per device (9 device types).
# All values should be multiples of 1/(2*resolution) for pixel-grid alignment.
# At resolution=20: quantum = 1/40 = 0.025 µm.
# Rectangular domain: 480×160 interior at res=20 (24.0 × 8.0 µm).
PARAM_RANGES = {
    "straight": {
        "wg_width_um": (0.40, 0.575),
        "dev_length_um": (6.0, 18.0),
    },
    "taper": {
        "wg_width_in": (0.40, 0.575),
        "wg_width_out": (0.60, 2.0),
        "taper_length_um": (3.0, 15.0),
    },
    "mmi": {
        "wg_width_um": (0.40, 0.575),
        "mmi_width_um": (4.5, 5.5),
        "mmi_length_um": (8.0, 15.0),
        "taper_width_um": (0.575, 1.5),
        "taper_length_um": (1.0, 3.0),
    },
    "sbend": {
        "wg_width_um": (0.40, 0.575),
        "lateral_offset_um": (2.0, 5.5),
        "R_min_um": (3.0, 7.0),
    },
    "ybranch": {
        "wg_width_um": (0.40, 0.575),
        "l_junction_um": (1.0, 3.0),
        "l_bend_um": (4.0, 7.0),
        "h_bend_um": (0.575, 2.5),
        "l_out_um": (1.0, 4.0),
    },
    "directional_coupler": {
        "wg_width_um": (0.40, 0.575),
        "gap_um": (0.10, 0.35),
        "wg_length_um": (5.0, 8.0),
        "bend_length_um": (4.0, 6.0),
        "lead_extra_gap_um": (0.825, 2.0),
    },
    "euler_bend": {
        "wg_width": (0.40, 0.575),
        "R_min_um": (2.0, 3.4),
    },
    "circular_bend": {
        "wg_width": (0.40, 0.575),
        "bend_radius_um": (2.0, 3.5),
    },
    "crossing": {
        "wg_width_h": (0.40, 0.575),
        "wg_width_v": (0.40, 0.575),
    },
}

# Valid source / input port indices per device (also used by `unified_sweep.py`).
INPUT_PORTS = {
    "straight": [1, 2],
    "taper": [1, 2],
    "mmi": [1, 2],
    "sbend": [1, 2],
    "ybranch": [1, 2, 3],
    "directional_coupler": [1, 2],
    "euler_bend": [1, 2],
    "circular_bend": [1, 2],
    "crossing": [1, 2, 3, 4],
}
