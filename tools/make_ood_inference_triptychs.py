#!/usr/bin/env python3
"""
Run FDTD + flow-matching inference on out-of-distribution devices and emit
paper-style triptychs (geometry + FDTD |Ez| + model |Ez|).

OOD families:
  * sbend            -- various radii / lateral offsets (Euler S-bends).
  * taper            -- various input/output widths and lengths.
  * cascaded_y1x3    -- a custom 1x3 splitter built from two Y-branches: the
                        first Y-branch splits the input into two arms; the
                        upper arm continues straight to a single output port,
                        while the lower arm feeds a second Y-branch that
                        splits into two more outputs (3 outputs total).

The script always runs FDTD on the chosen geometry, anchors the reference
field, runs the trained flow-matching model, and writes per-case PNG/PDF/NPZ
plus a combined grid figure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Model"
FDTD_DIR = REPO_ROOT / "FDTD"
TOOLS_DIR = REPO_ROOT / "tools"
for _p in (str(MODEL_DIR), str(FDTD_DIR), str(TOOLS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import phase_anchor_mask, phase_anchor_roi  # noqa: E402
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
    _draw_thick_line_mask,
    _meep_coord_to_cropped_px,
    _volume_to_line_px,
)
from devices_base import Device2DBase  # noqa: E402
from utils import (  # noqa: E402
    neff_siwire_from_tables,
    get_mode_alpha_2dir,
    pick_in_out_from_alpha,
)
from _triptych_metrics import residual_metrics, format_metric_label  # noqa: E402


DEFAULT_CKPT = (
    Path("/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm")
    / "REAL_VS_COMPLEX_real_h56_phase_residual_v100x12"
    / "checkpoints"
    / "0000300.pt"
)
DEFAULT_RESOLUTION = 20
DEFAULT_DPML = 1.0
DEFAULT_CROP_X_PX = 480
DEFAULT_CROP_Y_PX = 160
DEFAULT_WAVELENGTH_UM = 1.55

DEVICE_LABELS = {
    "sbend": "S-Bend",
    "taper": "Taper",
    "cascaded_y1x3": "Cascaded Y-Branch",
}


# ----------------------------------------------------------------------------
# OOD case definitions
# ----------------------------------------------------------------------------
@dataclass
class OODCase:
    name: str
    device_type: str
    params: dict[str, float]
    source_port: int = 1
    note: str = ""


def default_sbend_cases() -> list[OODCase]:
    # In-dataset sweep ranges (for reference): R_min in [3.0, 7.0],
    # lateral_offset in [2.0, 5.5], wg_width in [0.40, 0.575].
    # We deliberately push outside or near the edge of these ranges.
    return [
        OODCase(
            name="sbend_tightR_largeOffset",
            device_type="sbend",
            params={"wg_width_um": 0.45, "lateral_offset_um": 6.0, "R_min_um": 2.5},
            source_port=1,
            note="R_min below trained min (2.5 um), offset above trained max (6.0 um)",
        ),
        OODCase(
            name="sbend_smallOffset_largeR",
            device_type="sbend",
            params={"wg_width_um": 0.45, "lateral_offset_um": 1.5, "R_min_um": 8.5},
            source_port=1,
            note="offset below trained min (1.5 um), R above trained max (8.5 um)",
        ),
        OODCase(
            name="sbend_thinWG_midR",
            device_type="sbend",
            params={"wg_width_um": 0.38, "lateral_offset_um": 4.0, "R_min_um": 5.0},
            source_port=1,
            note="wg_width below trained min (0.38 um)",
        ),
    ]


def default_taper_cases() -> list[OODCase]:
    # Trained ranges: width_in [0.40, 0.575], width_out [0.60, 2.0],
    # taper_length [3.0, 15.0].
    return [
        OODCase(
            name="taper_short_steep",
            device_type="taper",
            params={"wg_width_in": 0.45, "wg_width_out": 2.4, "taper_length_um": 2.0},
            source_port=1,
            note="length below trained min (2 um), width_out above trained max (2.4 um)",
        ),
        OODCase(
            name="taper_long_wide",
            device_type="taper",
            params={"wg_width_in": 0.50, "wg_width_out": 2.5, "taper_length_um": 18.0},
            source_port=1,
            note="length above trained max (18 um), width_out above trained max",
        ),
        OODCase(
            name="taper_thinIn_thinOut",
            device_type="taper",
            params={"wg_width_in": 0.38, "wg_width_out": 0.55, "taper_length_um": 8.0},
            source_port=1,
            note="width_out below trained min (0.55 um), width_in below trained min",
        ),
    ]


def default_cascaded_cases() -> list[OODCase]:
    return [
        OODCase(
            name="cascaded_y1x3_default",
            device_type="cascaded_y1x3",
            params={
                "wg_width_um": 0.45,
                "l_in_um": 1.0,
                "l_junction1_um": 2.0,
                "l_bend1_um": 4.0,
                "h_bend1_um": 1.5,
                "l_mid_um": 1.5,
                "l_junction2_um": 2.0,
                "l_bend2_um": 3.5,
                "h_bend2_um": 1.0,
            },
            source_port=1,
            note="Cascaded 1x3 Y-branch: first split + upper-straight + second split",
        ),
    ]


# ----------------------------------------------------------------------------
# Cascaded 1x3 Y-branch: custom device built from primitives
# ----------------------------------------------------------------------------
class CascadedY1x3(Device2DBase):
    """
    1x3 splitter:
        input -> Y-branch (split into upper/lower)
        upper arm -> straight waveguide -> output port 2
        lower arm -> Y-branch (split into upper/lower)
                  -> output ports 3 (upper) and 4 (lower)

    Geometry primitives (Meep blocks + sine-bend block stacks) only; no Prism.
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wavelength_um: float = DEFAULT_WAVELENGTH_UM,
        l_in_um: float = 1.0,
        l_junction1_um: float = 2.0,
        l_bend1_um: float = 4.0,
        h_bend1_um: float = 1.5,
        l_mid_um: float = 1.5,
        l_junction2_um: float = 2.0,
        l_bend2_um: float = 3.5,
        h_bend2_um: float = 1.0,
        junction_widths_um: list[float] | None = None,
        use_prism_junction: bool = True,
        bend_n_segments: int = 80,
        n_core: float | None = None,
        n_clad: float = 1.444,
        port_y_pad_um: float = 0.4,
        source_shift_um: float = 0.5,
        cell_x: float | None = None,
        cell_y: float | None = None,
        dpml: float | None = None,
        resolution: int | None = None,
    ):
        super().__init__(cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)
        import meep as mp

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.l_in_um = float(l_in_um)
        self.l_junction1_um = float(l_junction1_um)
        self.l_bend1_um = float(l_bend1_um)
        self.h_bend1_um = float(h_bend1_um)
        self.l_mid_um = float(l_mid_um)
        self.l_junction2_um = float(l_junction2_um)
        self.l_bend2_um = float(l_bend2_um)
        self.h_bend2_um = float(h_bend2_um)
        self.bend_n_segments = int(bend_n_segments)
        self.use_prism_junction = bool(use_prism_junction)
        self.n_clad = float(n_clad)
        self.port_y_pad_um = float(port_y_pad_um)
        self.source_shift_um = float(source_shift_um)

        # Tidy3D-style 13-width junction profile, scaled by wg_width / 0.5.
        # Matches FDTD/ybranch/ybranch.py exactly so the multimode region
        # of each cascaded Y matches the trained Y-branch.
        if junction_widths_um is None:
            base = np.array(
                [0.5, 0.5, 0.6, 0.7, 0.9, 1.26, 1.4, 1.4, 1.4, 1.4, 1.31, 1.2, 1.2],
                dtype=float,
            )
            scale = self.wg_width_um / 0.5
            widths = (base * scale).tolist()
        else:
            widths = [float(w) for w in junction_widths_um]
        if len(widths) >= 1:
            widths[0] = self.wg_width_um  # continuity with input guide
        self.junction_widths_um = widths

        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
            if n_core is None
            else float(n_core)
        )
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

        # Will be set in build_geometry().
        self.geometry = None
        self.port_1 = None
        self.port_2 = None
        self.port_3 = None
        self.port_4 = None
        self.src_1 = None

        self.build_geometry()

    # --- helpers ----------------------------------------------------------
    def _ybranch_junction(self, x0: float, x1: float, y_center: float):
        """
        Tidy3D-style Y-branch junction (multimode taper) drawn as a Prism with
        the 13-width profile scaled by wg_width/0.5. Falls back to overlapping
        blocks if Prism construction fails. Matches FDTD/ybranch/ybranch.py:370-423.
        Returns (geometry_list, w_end) where w_end is the width at the wide
        end (used to align the S-bend start edges with the junction edge).
        """
        import meep as mp

        widths = self.junction_widths_um
        Nw = len(widths)
        w_end = float(widths[-1]) if Nw else float(self.wg_width_um)
        if Nw < 2:
            return [], w_end

        x_top = np.linspace(x0, x1, Nw)
        y_top = 0.5 * np.array(widths)

        verts = []
        for i in range(Nw):
            verts.append(mp.Vector3(float(x_top[i]), float(y_center + y_top[i]), 0))
        for i in range(Nw - 1, -1, -1):
            verts.append(mp.Vector3(float(x_top[i]), float(y_center - y_top[i]), 0))

        if self.use_prism_junction:
            try:
                return [mp.Prism(
                    vertices=verts, height=mp.inf, axis=mp.Z, material=self.core_medium,
                )], w_end
            except Exception:
                pass

        # Fallback: overlapping blocks (segmented taper).
        dx = (x1 - x0) / (Nw - 1)
        blocks = []
        for i in range(Nw - 1):
            xc = 0.5 * (x_top[i] + x_top[i + 1])
            wseg = 0.5 * (widths[i] + widths[i + 1])
            blocks.append(
                mp.Block(
                    size=mp.Vector3(dx * 1.05, wseg, mp.inf),
                    center=mp.Vector3(float(xc), float(y_center), 0),
                    material=self.core_medium,
                )
            )
        return blocks, w_end

    def _sine_sbend_blocks(self, x0, x1, y_center, h, w, n_seg=80):
        """Sine-profile S-bend (zero slope at both ends)."""
        import meep as mp
        L = x1 - x0
        if L <= 0:
            return []
        seg_len = L / n_seg
        blocks = []
        for i in range(n_seg):
            u = (i + 0.5) / n_seg
            x = x0 + u * L
            s = u * L
            y = y_center + (s * h / L) - (h * np.sin(2 * np.pi * s / L) / (2 * np.pi))
            blocks.append(
                mp.Block(
                    size=mp.Vector3(seg_len * 1.05, w, mp.inf),
                    center=mp.Vector3(float(x), float(y), 0),
                    material=self.core_medium,
                )
            )
        return blocks

    def _straight(self, x0, x1, y_center, w):
        import meep as mp
        if x1 <= x0:
            return []
        L = x1 - x0
        return [
            mp.Block(
                size=mp.Vector3(L, w, mp.inf),
                center=mp.Vector3(0.5 * (x0 + x1), y_center, 0),
                material=self.core_medium,
            )
        ]

    # --- geometry ---------------------------------------------------------
    def build_geometry(self):
        import meep as mp

        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml

        # X-axis layout, left-to-right:
        #   stem  | junc1 | bend1 | upper-straight . . . . . . . . . . .|
        #         |       |       | mid-lower | junc2 | bend2 | out-fan |
        x_in_start = -half_x  # extend stem into PML
        x_stem_end = nonpml_left + self.l_in_um
        x_j1_end = x_stem_end + self.l_junction1_um
        x_b1_end = x_j1_end + self.l_bend1_um
        x_mid_end = x_b1_end + self.l_mid_um
        x_j2_end = x_mid_end + self.l_junction2_um
        x_b2_end = x_j2_end + self.l_bend2_um

        if x_b2_end > nonpml_right:
            raise ValueError(
                f"CascadedY1x3 does not fit in non-PML x extent: "
                f"required end {x_b2_end:.3g} um vs avail {nonpml_right:.3g} um"
            )

        # Y fit check using the actual junction-aligned bend offsets.
        w_end_pre = float(self.junction_widths_um[-1]) if self.junction_widths_um else self.wg_width_um
        y_base_pre = 0.5 * (w_end_pre - self.wg_width_um)
        y_top_pre = +y_base_pre + self.h_bend1_um
        y_lower_pre_pre = -y_base_pre - self.h_bend1_um
        y_lower_lower_pre = y_lower_pre_pre - y_base_pre - self.h_bend2_um
        max_y = max(abs(y_top_pre), abs(y_lower_lower_pre)) + 0.5 * self.wg_width_um + 0.2
        if max_y > (half_y - self.dpml):
            raise ValueError(
                f"CascadedY1x3 does not fit in non-PML y extent: "
                f"required |y| {max_y:.3g} um vs avail {half_y - self.dpml:.3g} um"
            )

        geometry: list[Any] = []

        # 1) Input stem from left PML edge to junction-1 start.
        geometry += self._straight(x_in_start, x_stem_end, 0.0, self.wg_width_um)

        # 2) Junction 1: Tidy3D-style multimode taper (Prism polygon).
        j1_geom, w_end1 = self._ybranch_junction(x_stem_end, x_j1_end, y_center=0.0)
        geometry += j1_geom

        # 3) Sine S-bend split. Bend edges align with junction wide-end edges:
        #    y_base = 0.5 * (w_end - w_in), so each arm starts at y = ±y_base.
        y_base1 = 0.5 * (w_end1 - self.wg_width_um)
        geometry += self._sine_sbend_blocks(
            x_j1_end, x_b1_end, +y_base1, +self.h_bend1_um, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )
        geometry += self._sine_sbend_blocks(
            x_j1_end, x_b1_end, -y_base1, -self.h_bend1_um, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )
        # End-of-bend y positions (matches original ybranch.py:460-461).
        y_top = +y_base1 + self.h_bend1_um
        y_lower_pre = -y_base1 - self.h_bend1_um

        # 4) Upper arm straight all the way to the right PML edge.
        geometry += self._straight(x_b1_end, half_x, y_top, self.wg_width_um)

        # 5) Lower arm pre-junction-2 straight section.
        geometry += self._straight(x_b1_end, x_mid_end, y_lower_pre, self.wg_width_um)

        # 6) Junction 2 (lower arm): same multimode profile.
        j2_geom, w_end2 = self._ybranch_junction(x_mid_end, x_j2_end, y_center=y_lower_pre)
        geometry += j2_geom

        # 7) Second sine S-bend split with edge-aligned starting offsets.
        y_base2 = 0.5 * (w_end2 - self.wg_width_um)
        geometry += self._sine_sbend_blocks(
            x_j2_end, x_b2_end, y_lower_pre + y_base2, +self.h_bend2_um, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )
        geometry += self._sine_sbend_blocks(
            x_j2_end, x_b2_end, y_lower_pre - y_base2, -self.h_bend2_um, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )
        y_lower_upper = y_lower_pre + y_base2 + self.h_bend2_um
        y_lower_lower = y_lower_pre - y_base2 - self.h_bend2_um

        # 8) Bottom-arm straights from end of bend-2 to right PML edge.
        geometry += self._straight(x_b2_end, half_x, y_lower_upper, self.wg_width_um)
        geometry += self._straight(x_b2_end, half_x, y_lower_lower, self.wg_width_um)

        self.geometry = geometry

        # Device window for plotting helpers (full interior).
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = self.inner_x
        self.dev_wy = self.inner_y

        # --- Ports (transverse line volumes; size_x=0 so they're vertical lines)
        port_span = self.wg_width_um + self.port_y_pad_um
        port_size = mp.Vector3(0, port_span, 0)

        port_x_left = nonpml_left + 0.5
        port_x_right = nonpml_right - 0.5

        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, 0.0, 0), size=port_size)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_right, y_top, 0), size=port_size)
        self.port_3 = mp.Volume(center=mp.Vector3(port_x_right, y_lower_upper, 0), size=port_size)
        self.port_4 = mp.Volume(center=mp.Vector3(port_x_right, y_lower_lower, 0), size=port_size)

        # Source upstream of port 1 (still inside non-PML region).
        src_x = max(port_x_left - self.source_shift_um, nonpml_left + 0.1)
        self.src_1 = mp.Volume(center=mp.Vector3(src_x, 0.0, 0), size=port_size)

        # Cache port y-positions for masks/diagnostics.
        self.y_top = float(y_top)
        self.y_lower_upper = float(y_lower_upper)
        self.y_lower_lower = float(y_lower_lower)

    # --- FDTD run --------------------------------------------------------
    def simulate(self, decay_tol: float = 1e-5, df_frac: float = 0.1):
        """Run FDTD with port-1 excitation and return (eps_full, Ez_full, S_dict)."""
        import meep as mp

        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=self.src_1,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=mp.Vector3(+1, 0, 0),
            )
        ]

        sim = mp.Simulation(
            cell_size=self.cell,
            resolution=int(self.resolution),
            boundary_layers=[mp.PML(float(self.dpml))],
            geometry=self.geometry,
            default_material=self.clad_medium,
            sources=sources,
        )

        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3))
        m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_4))

        dft = sim.add_dft_fields(
            [mp.Ez], fcen, 0, 1,
            center=self.full_plane.center, size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_full = sim.get_epsilon().T.astype(np.float32)
        Ez_full = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        # S-parameters (toward-device convention).
        a1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        a2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        a3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        a4 = get_mode_alpha_2dir(sim, m4, band=1, eig_parity=mp.NO_PARITY)
        a1_in, b1_out = pick_in_out_from_alpha(a1, +1, dir_plus=0, dir_minus=1)
        _, b2_out = pick_in_out_from_alpha(a2, -1, dir_plus=0, dir_minus=1)
        _, b3_out = pick_in_out_from_alpha(a3, -1, dir_plus=0, dir_minus=1)
        _, b4_out = pick_in_out_from_alpha(a4, -1, dir_plus=0, dir_minus=1)
        S = {
            "S11": (b1_out / a1_in).item(),
            "S21": (b2_out / a1_in).item(),
            "S31": (b3_out / a1_in).item(),
            "S41": (b4_out / a1_in).item(),
        }

        sim.reset_meep()
        return eps_full, Ez_full, S


def _cascaded_masks(
    dev: CascadedY1x3,
    *,
    cell_x: float,
    cell_y: float,
    dpml: float,
    resolution: int,
    ny: int,
    nx: int,
    thickness_px: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source + 4 port masks for the cascaded device, in cropped pixel coords."""
    src_line = _volume_to_line_px(dev.src_1, cell_x, cell_y, dpml, resolution)
    src_mask = _draw_thick_line_mask(
        ny, nx,
        src_line["line_start_px"][0], src_line["line_start_px"][1],
        src_line["line_end_px"][0], src_line["line_end_px"][1],
        thickness_px=thickness_px,
    )

    port_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    port_vols = {1: dev.port_1, 2: dev.port_2, 3: dev.port_3, 4: dev.port_4}
    port_masks = []
    for pid in port_ids.tolist():
        line = _volume_to_line_px(port_vols[pid], cell_x, cell_y, dpml, resolution)
        port_masks.append(
            _draw_thick_line_mask(
                ny, nx,
                line["line_start_px"][0], line["line_start_px"][1],
                line["line_end_px"][0], line["line_end_px"][1],
                thickness_px=thickness_px,
            )
        )
    port_masks = np.stack(port_masks, axis=0).astype(np.float32)
    return src_mask.astype(np.float32), port_ids, port_masks


# ----------------------------------------------------------------------------
# Sample container & simulation helpers
# ----------------------------------------------------------------------------
def _build_sample_for_existing(
    case: OODCase,
    *,
    wavelength_um: float,
    resolution: int,
    dpml: float,
    crop_x_px: int,
    crop_y_px: int,
    decay_tol: float,
) -> dict[str, Any]:
    """Build sample dict for sbend/taper via the existing FDTD entry point."""
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * float(dpml)
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * float(dpml)

    dev = build_device(
        case.device_type, case.params, wavelength_um, resolution,
        dpml, cell_x, cell_y, crop_x_px, crop_y_px,
    )
    eps_full, Ez_full, S, cell_size = run_sim_and_extract(
        dev, case.device_type, int(case.source_port), float(decay_tol)
    )

    pml_px = int(round(float(dpml) * float(resolution)))
    if pml_px > 0:
        eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px]
        interior_x = cell_size[0] - 2.0 * dpml
        interior_y = cell_size[1] - 2.0 * dpml
    else:
        eps = eps_full
        Ez = Ez_full
        interior_x = cell_size[0]
        interior_y = cell_size[1]

    ny, nx = eps.shape
    src_mask, port_ids, port_masks = get_device_masks(
        dev, case.device_type, int(case.source_port),
        cell_size[0], cell_size[1], dpml, resolution, ny, nx, thickness_px=3,
    )

    return {
        "case": case,
        "device_type": case.device_type,
        "input_port": int(case.source_port),
        "wavelength_um": float(wavelength_um),
        "eps": eps.astype(np.float32),
        "ez_real": Ez.real.astype(np.float32),
        "ez_imag": Ez.imag.astype(np.float32),
        "src_mask": src_mask.astype(np.float32),
        "port_ids": np.asarray(port_ids, dtype=np.int32),
        "port_masks": np.asarray(port_masks, dtype=np.float32),
        "dx_um": 1.0 / float(resolution),
        "dy_um": 1.0 / float(resolution),
        "Lx_um": float(interior_x),
        "Ly_um": float(interior_y),
        "params": {k: float(v) for k, v in case.params.items()},
        "sparams": {k: complex(v) for k, v in S.items()},
    }


def _build_sample_for_cascaded(
    case: OODCase,
    *,
    wavelength_um: float,
    resolution: int,
    dpml: float,
    crop_x_px: int,
    crop_y_px: int,
    decay_tol: float,
) -> dict[str, Any]:
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * float(dpml)
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * float(dpml)

    dev = CascadedY1x3(
        wavelength_um=wavelength_um,
        resolution=int(resolution),
        cell_x=cell_x,
        cell_y=cell_y,
        dpml=dpml,
        **{k: float(v) for k, v in case.params.items()},
    )
    eps_full, Ez_full, S = dev.simulate(decay_tol=float(decay_tol))

    pml_px = int(round(float(dpml) * float(resolution)))
    if pml_px > 0:
        eps = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez = Ez_full[pml_px:-pml_px, pml_px:-pml_px]
    else:
        eps = eps_full
        Ez = Ez_full
    ny, nx = eps.shape

    src_mask, port_ids, port_masks = _cascaded_masks(
        dev,
        cell_x=dev.cell_x, cell_y=dev.cell_y, dpml=dev.dpml,
        resolution=dev.resolution, ny=ny, nx=nx, thickness_px=3,
    )

    return {
        "case": case,
        "device_type": case.device_type,
        "input_port": int(case.source_port),
        "wavelength_um": float(wavelength_um),
        "eps": eps.astype(np.float32),
        "ez_real": Ez.real.astype(np.float32),
        "ez_imag": Ez.imag.astype(np.float32),
        "src_mask": src_mask.astype(np.float32),
        "port_ids": np.asarray(port_ids, dtype=np.int32),
        "port_masks": np.asarray(port_masks, dtype=np.float32),
        "dx_um": 1.0 / float(resolution),
        "dy_um": 1.0 / float(resolution),
        "Lx_um": float(dev.inner_x),
        "Ly_um": float(dev.inner_y),
        "params": {k: float(v) for k, v in case.params.items()},
        "sparams": {k: complex(v) for k, v in S.items()},
    }


# ----------------------------------------------------------------------------
# Anchoring + inference (mirrors make_paper_inference_triptychs.py)
# ----------------------------------------------------------------------------
def _anchor_reference(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ezr = sample["ez_real"]
    ezi = sample["ez_imag"]
    eps = sample["eps"]
    src = sample["src_mask"]
    ezr2, ezi2, _ = phase_anchor_mask(ezr, ezi, src > 0.5, eps_r=eps, thr_eps=3.0)
    if float((src > 0.5).sum()) < 4.0:
        ezr2, ezi2, _ = phase_anchor_roi(
            ezr, ezi, eps_r=eps, pml_cells=0, margin=2,
            roi_x=(40, 140), thr_eps=3.0,
        )
    return ezr2.astype(np.float32, copy=False), ezi2.astype(np.float32, copy=False)


def _select_runtime_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _predict_sample(
    sample: dict[str, Any],
    *,
    ckpt: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    num_steps: int,
    time_grid: str,
    seed: int,
    amp: bool,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")

    eps = sample["eps"]
    src = sample["src_mask"]
    wavelength_um = float(sample["wavelength_um"])

    cond_maps = _build_cond_maps(
        eps, src,
        stats=stats, ckpt_args=ckpt_args, device=device,
        dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
    )
    cond = _build_cond_vector(wavelength_um, stats, device=device)
    lambda_um_t = torch.tensor([[wavelength_um]], device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x0 = torch.randn(
        (1, 2, eps.shape[0], eps.shape[1]),
        device=device, dtype=torch.float32, generator=gen,
    )

    if time_grid == "checkpoint":
        time_grid = str(_ckpt_get(ckpt_args, "time_grid", "linear"))
    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            fields_norm = fm_sample(
                model, x0,
                num_steps=int(num_steps),
                use_stoc_samp=False,
                cond_maps=cond_maps, cond=cond, lambda_um=lambda_um_t,
                phys_gate=1.0, phase_gate=1.0,
                time_grid=time_grid, progress=progress,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    fields_norm = fields_norm.float()
    ezr = fields_norm[0, 0].detach().cpu().numpy() * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
    ezi = fields_norm[0, 1].detach().cpu().numpy() * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])
    return ezr.astype(np.float32, copy=False), ezi.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _setup_matplotlib(show: bool, usetex: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "text.usetex": bool(usetex),
    })


def _extent(sample: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -0.5 * float(sample["Lx_um"]),
        0.5 * float(sample["Lx_um"]),
        -0.5 * float(sample["Ly_um"]),
        0.5 * float(sample["Ly_um"]),
    )


def _overlay_source_mask(ax: Any, sample: dict[str, Any]) -> None:
    ext = _extent(sample)
    src = sample["src_mask"]
    if np.any(src > 0.5):
        overlay = np.ma.masked_where(src <= 0.5, np.ones_like(src, dtype=np.float32))
        ax.imshow(
            overlay, origin="lower", extent=ext, cmap="Greens",
            vmin=0, vmax=1, alpha=0.55, aspect="equal", interpolation="nearest",
        )


def _add_colorbar(fig: Any, im: Any, ax: Any, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, shrink=0.74)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _compute_residual_metrics(
    sample: dict[str, Any],
    fdtd_ezr: np.ndarray, fdtd_ezi: np.ndarray,
    pred_ezr: np.ndarray, pred_ezi: np.ndarray,
) -> dict[str, float]:
    return residual_metrics(
        sample["eps"], sample["src_mask"],
        fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi,
        dx_um=float(sample["dx_um"]), dy_um=float(sample["dy_um"]),
        wavelength_um=float(sample["wavelength_um"]),
    )


def _annotate_metric(ax: Any, metric: dict[str, float]) -> None:
    ax.text(
        0.97, 0.95, format_metric_label(metric),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0),
    )


def _plot_triptych(
    sample: dict[str, Any],
    fdtd_ezr: np.ndarray, fdtd_ezi: np.ndarray,
    pred_ezr: np.ndarray, pred_ezi: np.ndarray,
    *,
    out_path: Path, show: bool,
) -> None:
    import matplotlib.pyplot as plt

    ext = _extent(sample)
    eps = sample["eps"]
    fdtd_mag = np.abs(fdtd_ezr + 1j * fdtd_ezi)
    pred_mag = np.abs(pred_ezr + 1j * pred_ezi)
    vmax = float(np.percentile(np.concatenate([fdtd_mag.ravel(), pred_mag.ravel()]), 99.5))
    vmax = vmax if vmax > 0 else None

    metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    panels = [
        (eps, r"Geometry $\varepsilon_r$", "viridis", None, r"$\varepsilon_r$"),
        (fdtd_mag, r"FDTD $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
        (pred_mag, r"Model $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
    ]
    for col, (ax, (arr, title, cmap, vmax_i, cbar_label)) in enumerate(zip(axes, panels)):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal",
                       vmin=0 if vmax_i else None, vmax=vmax_i)
        if title != r"Geometry $\varepsilon_r$":
            level = 0.5 * (float(eps.min()) + float(eps.max()))
            ax.contour(eps, levels=[level], colors=["white"], linewidths=0.45,
                       origin="lower", extent=ext, alpha=0.75)
        _overlay_source_mask(ax, sample)
        ax.set_title(title)
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        _add_colorbar(fig, im, ax, cbar_label)
        if col == 2:
            _annotate_metric(ax, metric)

    case: OODCase = sample["case"]
    dev_label = DEVICE_LABELS.get(sample["device_type"], sample["device_type"])
    params_text = ", ".join(f"{k}={v:g}" for k, v in sorted(case.params.items()))
    fig.suptitle(
        rf"OOD {dev_label}: source $P_{{{sample['input_port']}}}$, "
        rf"$\lambda={sample['wavelength_um']:.2f}\,\mu$m"
        + (f" | {params_text}" if params_text else ""),
        y=1.08, fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_combined(
    rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    out_path: Path, show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(11.2, 2.65 * len(rows)),
                             constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])

    for r, (sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi) in enumerate(rows):
        ext = _extent(sample)
        eps = sample["eps"]
        fdtd_mag = np.abs(fdtd_ezr + 1j * fdtd_ezi)
        pred_mag = np.abs(pred_ezr + 1j * pred_ezi)
        vmax = float(np.percentile(np.concatenate([fdtd_mag.ravel(), pred_mag.ravel()]), 99.5))
        vmax = vmax if vmax > 0 else None
        metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)
        panels = [
            (eps, r"Geometry $\varepsilon_r$", "viridis", None, r"$\varepsilon_r$"),
            (fdtd_mag, r"FDTD $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
            (pred_mag, r"Model $|E_z|$", "magma", vmax, r"$|E_z|$ (arb. units)"),
        ]
        for c, (arr, title, cmap, vmax_i, cbar_label) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal",
                           vmin=0 if vmax_i else None, vmax=vmax_i)
            if c > 0:
                level = 0.5 * (float(eps.min()) + float(eps.max()))
                ax.contour(eps, levels=[level], colors=["white"], linewidths=0.4,
                           origin="lower", extent=ext, alpha=0.75)
            _overlay_source_mask(ax, sample)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                row_label = DEVICE_LABELS.get(sample["device_type"], sample["device_type"])
                ax.set_ylabel(row_label + "\n" + r"$y$ ($\mu$m)")
            else:
                ax.set_ylabel(r"$y$ ($\mu$m)")
            ax.set_xlabel(r"$x$ ($\mu$m)")
            _add_colorbar(fig, im, ax, cbar_label)
            if c == 2:
                _annotate_metric(ax, metric)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _gather_cases(
    families: tuple[str, ...],
    cases_json: str | None,
) -> list[OODCase]:
    if cases_json:
        with open(cases_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--cases-json must contain a list of {name, device, params, source_port?}")
        out: list[OODCase] = []
        for entry in data:
            out.append(
                OODCase(
                    name=str(entry["name"]),
                    device_type=str(entry["device"]),
                    params={str(k): float(v) for k, v in entry["params"].items()},
                    source_port=int(entry.get("source_port", 1)),
                    note=str(entry.get("note", "")),
                )
            )
        return out

    out = []
    if "sbend" in families:
        out += default_sbend_cases()
    if "taper" in families:
        out += default_taper_cases()
    if "cascaded_y1x3" in families:
        out += default_cascaded_cases()
    return out


def _save_npz(sample: dict[str, Any], fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi, out_path: Path) -> None:
    case: OODCase = sample["case"]
    metric = _compute_residual_metrics(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi)
    np.savez_compressed(
        out_path,
        eps=sample["eps"].astype(np.float32),
        src_mask=sample["src_mask"].astype(np.float32),
        port_ids=sample["port_ids"].astype(np.int32),
        port_masks=sample["port_masks"].astype(np.float32),
        fdtd_Ez_real=fdtd_ezr.astype(np.float32),
        fdtd_Ez_imag=fdtd_ezi.astype(np.float32),
        pred_Ez_real=pred_ezr.astype(np.float32),
        pred_Ez_imag=pred_ezi.astype(np.float32),
        device=np.array(sample["device_type"]),
        case_name=np.array(case.name),
        input_port=np.int32(sample["input_port"]),
        wavelength_um=np.float32(sample["wavelength_um"]),
        params_json=np.array(json.dumps(sample["params"], sort_keys=True)),
        sparams_json=np.array(json.dumps(
            {k: [float(v.real), float(v.imag)] for k, v in sample["sparams"].items()},
            sort_keys=True,
        )),
        residual_metrics_json=np.array(json.dumps(metric, sort_keys=True)),
        Lx_um=np.float32(sample["Lx_um"]),
        Ly_um=np.float32(sample["Ly_um"]),
        dx_um=np.float32(sample["dx_um"]),
        dy_um=np.float32(sample["dy_um"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD-device FDTD + flow-matching triptychs.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="FM+phase+residual checkpoint.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "ood_inference_triptychs"))
    parser.add_argument(
        "--families",
        default="sbend,taper,cascaded_y1x3",
        help="Comma-separated subset of {sbend,taper,cascaded_y1x3}.",
    )
    parser.add_argument(
        "--cases-json", default="",
        help="Optional JSON file with explicit case list overriding --families defaults.",
    )
    parser.add_argument(
        "--cases", default="",
        help="Comma-separated subset of case names to run (filters defaults/--cases-json).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any case whose <case_name>_triptych.png already exists in --out-dir.",
    )
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--dpml", type=float, default=DEFAULT_DPML)
    parser.add_argument("--crop-x-px", type=int, default=DEFAULT_CROP_X_PX)
    parser.add_argument("--crop-y-px", type=int, default=DEFAULT_CROP_Y_PX)
    parser.add_argument("--decay-tol", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="checkpoint")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device-runtime", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--usetex", action="store_true")
    args = parser.parse_args()

    _setup_matplotlib(show=bool(args.show), usetex=bool(args.usetex))

    families = tuple(s.strip() for s in args.families.split(",") if s.strip())
    cases = _gather_cases(families, args.cases_json or None)
    if args.cases:
        wanted = {s.strip() for s in args.cases.split(",") if s.strip()}
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            raise RuntimeError(f"--cases contains unknown names: {sorted(missing)}")
    if not cases:
        raise RuntimeError("No OOD cases selected. Pass --families, --cases, or --cases-json.")

    runtime_device = _select_runtime_device(args.device_runtime)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        skipped = [c.name for c in cases if (out_dir / f"{c.name}_triptych.png").is_file()]
        cases = [c for c in cases if not (out_dir / f"{c.name}_triptych.png").is_file()]
        if skipped:
            print(f"[ood-triptychs] skip-existing: {skipped}")
        if not cases:
            print("[ood-triptychs] all selected cases already present; nothing to do.")
            return

    print(f"[ood-triptychs] checkpoint: {ckpt_path}")
    print(f"[ood-triptychs] runtime_device: {runtime_device}")
    print(f"[ood-triptychs] cases: {[c.name for c in cases]}")

    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    print(f"[ood-triptychs] loaded weights: {state_key}")

    combined_rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, case in enumerate(cases):
        print(f"[ood-triptychs] {case.name} ({case.device_type}) :: {case.params} :: {case.note}")

        if case.device_type in ("sbend", "taper"):
            sample = _build_sample_for_existing(
                case,
                wavelength_um=float(args.wavelength_um),
                resolution=int(args.resolution),
                dpml=float(args.dpml),
                crop_x_px=int(args.crop_x_px),
                crop_y_px=int(args.crop_y_px),
                decay_tol=float(args.decay_tol),
            )
        elif case.device_type == "cascaded_y1x3":
            sample = _build_sample_for_cascaded(
                case,
                wavelength_um=float(args.wavelength_um),
                resolution=int(args.resolution),
                dpml=float(args.dpml),
                crop_x_px=int(args.crop_x_px),
                crop_y_px=int(args.crop_y_px),
                decay_tol=float(args.decay_tol),
            )
        else:
            raise ValueError(f"Unsupported device_type '{case.device_type}'")

        fdtd_ezr, fdtd_ezi = _anchor_reference(sample)
        pred_ezr, pred_ezi = _predict_sample(
            sample,
            ckpt=ckpt, model=model, device=runtime_device,
            num_steps=int(args.num_steps),
            time_grid=str(args.time_grid),
            seed=int(args.seed) + i,
            amp=bool(args.amp),
            progress=bool(args.progress),
        )
        combined_rows.append((sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi))

        out_path = out_dir / f"{case.name}_triptych.png"
        _plot_triptych(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi,
                       out_path=out_path, show=bool(args.show))
        _save_npz(sample, fdtd_ezr, fdtd_ezi, pred_ezr, pred_ezi, out_path.with_suffix(".npz"))
        print(f"[ood-triptychs] saved: {out_path}")
        print(f"[ood-triptychs] saved: {out_path.with_suffix('.pdf')}")
        print(f"[ood-triptychs] saved: {out_path.with_suffix('.npz')}")

    combined_path = out_dir / "ood_combined_triptychs.png"
    _plot_combined(combined_rows, out_path=combined_path, show=bool(args.show))
    print(f"[ood-triptychs] saved: {combined_path}")
    print(f"[ood-triptychs] saved: {combined_path.with_suffix('.pdf')}")
    print(f"[ood-triptychs] elapsed_s={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
