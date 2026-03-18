# mmi.py
"""
2×2 Multimode Interferometer (MMI) device for photonic dataset generation.

Geometry:
    Port 1 ══[wg]══▷[taper]▷┌──────────────┐◁[taper]◁══[wg]══ Port 3
                              │   MMI body    │
    Port 2 ══[wg]══▷[taper]▷│  W × L_mmi   │◁[taper]◁══[wg]══ Port 4
                              └──────────────┘

Access waveguides at ±W_mmi/3 from center (general interference condition).
Tapers widen from wg_width to taper_width at the MMI junction.
"""
import meep as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import neff_siwire_from_tables
from devices_base import Device2DBase


def _find_neff_tables_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "neff_tables"),
        os.path.join(os.path.dirname(here), "neff_tables"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return "neff_tables"


class MMI2x2(Device2DBase):
    """
    Symmetric 2×2 multimode interferometer with tapered access waveguides.

    Port numbering (looking down +x, y up):
        1: left,  top arm
        2: left,  bottom arm
        3: right, top arm
        4: right, bottom arm

    Access waveguides are positioned at ±W_mmi/3 from center (general
    interference condition for 2×2 MMI). Tapers transition from wg_width
    to taper_width at the MMI junction.

    Supports deterministic pixel-exact crops:
      - Provide crop_x_px/crop_y_px, resolution, dpml and leave cell_x_um/cell_y_um None
      - With quantize_grid=True (default), dpml is snapped to integer pixels and
        cell is chosen so interior (non-PML) is exactly crop_x_px × crop_y_px.
      - Rectangular grids (crop_x_px ≠ crop_y_px) are supported for elongated devices.
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        mmi_width_um: float = 4.0,
        mmi_length_um: float = 15.0,
        taper_width_um: float = 1.0,
        taper_length_um: float = 2.0,
        wavelength_um: float = 1.55,
        resolution: int = 20,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 5.0 / 7.0,
        crop_x_px: int = 336,
        crop_y_px: int = 112,
        quantize_grid: bool = True,
        source_shift_um: float = 0.5,
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        fit_margin_um: float = 0.5,
        orientation: str = "horizontal",
    ):
        crop_x_px = int(crop_x_px)
        crop_y_px = int(crop_y_px)
        if crop_x_px <= 0 or crop_y_px <= 0:
            raise ValueError("crop_x_px and crop_y_px must be > 0")
        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        self.orientation = str(orientation)
        self.crop_x_px = crop_x_px
        self.crop_y_px = crop_y_px
        self.fit_margin_um = float(fit_margin_um)

        # Quantize dpml + cell so (crop_*_px + 2*pml_px) is integer pixels.
        if bool(quantize_grid) and (cell_x_um is None or cell_y_um is None):
            pml_px = int(np.round(float(dpml) * float(resolution)))
            dpml_q = float(pml_px) / float(resolution)
            full_x_px = int(crop_x_px + 2 * pml_px)
            full_y_px = int(crop_y_px + 2 * pml_px)
            cell_x_um_q = float(full_x_px) / float(resolution)
            cell_y_um_q = float(full_y_px) / float(resolution)
            cx = cell_x_um_q if cell_x_um is None else float(cell_x_um)
            cy = cell_y_um_q if cell_y_um is None else float(cell_y_um)
            dpml_use = dpml_q
        else:
            cx = float(cell_x_um) if cell_x_um is not None else (float(crop_x_px) / float(resolution) + 2.0 * float(dpml))
            cy = float(cell_y_um) if cell_y_um is not None else (float(crop_y_px) / float(resolution) + 2.0 * float(dpml))
            dpml_use = float(dpml)

        super().__init__(cell_x=float(cx), cell_y=float(cy), dpml=float(dpml_use), resolution=int(resolution))

        self.wg_width_um = float(wg_width_um)
        self.mmi_width_um = float(mmi_width_um)
        self.mmi_length_um = float(mmi_length_um)
        self.taper_width_um = float(taper_width_um)
        self.taper_length_um = float(taper_length_um)
        self.wavelength_um = float(wavelength_um)

        if n_core is None:
            self.n_core = float(
                neff_siwire_from_tables(
                    self.wg_width_um,
                    self.wavelength_um,
                    tables_dir=_find_neff_tables_dir(),
                )
            )
        else:
            self.n_core = float(n_core)

        self.n_clad = float(n_clad)
        self.source_shift_um = float(source_shift_um)

        # Access waveguide y-offset (general interference: ±W/3)
        self.y_access_um = self.mmi_width_um / 3.0

        # To be filled by build_geometry
        self.geometry = None
        self.port_1 = self.port_2 = self.port_3 = self.port_4 = None
        self.src_1 = self.src_2 = self.src_3 = self.src_4 = None
        self.full_plane = None
        self.clad_medium = None

        self.x_port_left_um = None
        self.x_port_right_um = None
        self.port_y_offset_um = None

        self.src_x_left_um = None
        self.src_x_right_um = None
        self.src_y_top_um = None
        self.src_y_bot_um = None

        self.build_geometry()

    # ── Orientation helpers (same pattern as DirectionalCoupler2D) ──

    def _orient_coords(self, x_local: float, y_local: float) -> tuple[float, float]:
        """Transform local (propagation, transverse) coords to Meep (x, y)."""
        if self.orientation == "horizontal":
            return float(x_local), float(y_local)
        else:
            return float(y_local), float(x_local)

    def _orient_vec3(self, x_local: float, y_local: float, z: float = 0.0) -> mp.Vector3:
        """Create a Meep Vector3 with orientation-aware coordinates."""
        mx, my = self._orient_coords(x_local, y_local)
        return mp.Vector3(mx, my, z)

    def _orient_size(self, size_along: float, size_across: float) -> mp.Vector3:
        """Create size Vector3: (size along propagation, size across, z)."""
        if self.orientation == "horizontal":
            return mp.Vector3(size_along, size_across, mp.inf)
        else:
            return mp.Vector3(size_across, size_along, mp.inf)

    def _propagation_direction(self):
        """Return the Meep direction constant for wave propagation."""
        return mp.X if self.orientation == "horizontal" else mp.Y

    def nonpml_shape(self) -> tuple[int, int]:
        p = self.pml_px
        nx = int(np.round(self.cell_x * self.resolution)) - 2 * p
        ny = int(np.round(self.cell_y * self.resolution)) - 2 * p
        return ny, nx

    def _um_to_px(self, x_um: float, y_um: float, crop_pml: bool) -> tuple[float, float]:
        """Convert um coordinates to pixel coordinates in DISPLAY space."""
        p = self.pml_px if crop_pml else 0
        px_x_sim = (float(x_um) + 0.5 * self.cell_x) * float(self.resolution) - float(p)
        px_y_sim = (float(y_um) + 0.5 * self.cell_y) * float(self.resolution) - float(p)
        if self.orientation == "vertical":
            return px_y_sim, px_x_sim
        return px_x_sim, px_y_sim

    # ── Port accessors ──

    def get_port_centers_um(self):
        yoff = float(self.port_y_offset_um)
        return {
            1: (float(self.x_port_left_um), +yoff),
            2: (float(self.x_port_left_um), -yoff),
            3: (float(self.x_port_right_um), +yoff),
            4: (float(self.x_port_right_um), -yoff),
        }

    def get_port_y_span_um(self):
        """Port span captures one access waveguide mode without overlapping neighbor."""
        gap_between_access = 2.0 * self.y_access_um - self.wg_width_um
        return float(self.wg_width_um + 0.5 * gap_between_access)

    def get_port_region_px(self, port: int, crop_pml: bool = True) -> dict:
        if port not in (1, 2, 3, 4):
            raise ValueError("port must be 1..4")
        centers = self.get_port_centers_um()
        x0, y0 = centers[int(port)]
        span = self.get_port_y_span_um()
        half = 0.5 * span
        x_px, y_px = self._um_to_px(x0, y0, crop_pml)
        x_s, y_s = self._um_to_px(x0, y0 - half, crop_pml)
        x_e, y_e = self._um_to_px(x0, y0 + half, crop_pml)
        dir_px = (0.0, 1.0) if self.orientation == "vertical" else (1.0, 0.0)
        return {
            "center_px": (x_px, y_px),
            "line_start_px": (x_s, y_s),
            "line_end_px": (x_e, y_e),
            "direction_px": dir_px,
        }

    def get_source_region_px(self, input_port: int = 1, crop_pml: bool = True) -> dict:
        if input_port not in (1, 2, 3, 4):
            raise ValueError("input_port must be 1..4")
        if input_port in (1, 2):
            x0 = float(self.src_x_left_um)
            y0 = float(self.src_y_top_um if input_port == 1 else self.src_y_bot_um)
        else:
            x0 = float(self.src_x_right_um)
            y0 = float(self.src_y_top_um if input_port == 3 else self.src_y_bot_um)
        span = self.get_port_y_span_um()
        half = 0.5 * span
        x_px, y_px = self._um_to_px(x0, y0, crop_pml)
        x_s, y_s = self._um_to_px(x0, y0 - half, crop_pml)
        x_e, y_e = self._um_to_px(x0, y0 + half, crop_pml)
        dir_px = (0.0, 1.0) if self.orientation == "vertical" else (1.0, 0.0)
        return {
            "center_px": (x_px, y_px),
            "line_start_px": (x_s, y_s),
            "line_end_px": (x_e, y_e),
            "direction_px": dir_px,
        }

    def get_eps_and_cell(self, crop_pml: bool = False):
        sim = mp.Simulation(
            cell_size=self.cell,
            resolution=self.resolution,
            boundary_layers=[mp.PML(self.dpml)],
            geometry=self.geometry,
            default_material=self.clad_medium,
            sources=[],
        )
        sim.init_sim()
        eps_2d = sim.get_epsilon()
        eps_mid = eps_2d.T
        sim.reset_meep()

        if self.orientation == "vertical":
            eps_mid = eps_mid.T
            cell_x_display, cell_y_display = self.cell_y, self.cell_x
        else:
            cell_x_display, cell_y_display = self.cell_x, self.cell_y

        if not crop_pml:
            return eps_mid, (cell_x_display, cell_y_display)

        p = self.pml_px
        if p <= 0:
            return eps_mid, (cell_x_display, cell_y_display)

        return eps_mid[p:-p, p:-p], (cell_x_display - 2 * self.dpml, cell_y_display - 2 * self.dpml)

    # ── Geometry ──

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        Lm = self.mmi_length_um
        Wm = self.mmi_width_um
        Lt = self.taper_length_um
        Wt = self.taper_width_um
        Wg = self.wg_width_um
        y_acc = self.y_access_um  # = Wm / 3

        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_top = +half_y - self.dpml

        # ── Fit checks ──
        # X: MMI body + 2 tapers must fit inside non-PML with margin
        x_extent = 0.5 * Lm + Lt + self.fit_margin_um
        if x_extent > nonpml_right:
            raise ValueError(
                f"MMI2x2: device X extent ({x_extent:.2f} µm from center) exceeds "
                f"non-PML boundary ({nonpml_right:.2f} µm). "
                f"Reduce mmi_length_um or taper_length_um, or increase crop/cell."
            )

        # Y: MMI body must fit inside non-PML with margin
        y_extent = 0.5 * Wm + self.fit_margin_um
        if y_extent > nonpml_top:
            raise ValueError(
                f"MMI2x2: MMI body Y extent ({y_extent:.2f} µm from center) exceeds "
                f"non-PML boundary ({nonpml_top:.2f} µm). "
                f"Reduce mmi_width_um or increase crop/cell."
            )

        # Tapers must not overlap each other (on same side)
        if Wt > 2.0 * y_acc:
            raise ValueError(
                f"MMI2x2: taper_width_um ({Wt:.3f}) exceeds 2*y_access ({2*y_acc:.3f}), "
                f"tapers would overlap. Reduce taper_width_um or increase mmi_width_um."
            )

        geometry = []

        # 1. Central MMI box
        geometry.append(
            mp.Block(
                size=self._orient_size(Lm, Wm),
                center=self._orient_vec3(0.0, 0.0),
                material=core,
            )
        )

        # 2. Four tapers (trapezoids via mp.Prism)
        x_mmi_left = -0.5 * Lm
        x_mmi_right = +0.5 * Lm
        x_taper_left_start = x_mmi_left - Lt
        x_taper_right_end = x_mmi_right + Lt

        def make_taper_prism(x_left, x_right, y_center, w_left, w_right):
            """Create a trapezoidal taper prism from x_left to x_right."""
            vertices = [
                self._orient_vec3(x_left, y_center - 0.5 * w_left),
                self._orient_vec3(x_left, y_center + 0.5 * w_left),
                self._orient_vec3(x_right, y_center + 0.5 * w_right),
                self._orient_vec3(x_right, y_center - 0.5 * w_right),
            ]
            return mp.Prism(vertices=vertices, height=mp.inf, material=core)

        # Left tapers: narrow (wg_width) at lead side, wide (taper_width) at MMI
        geometry.append(make_taper_prism(x_taper_left_start, x_mmi_left, +y_acc, Wg, Wt))
        geometry.append(make_taper_prism(x_taper_left_start, x_mmi_left, -y_acc, Wg, Wt))

        # Right tapers: wide (taper_width) at MMI, narrow (wg_width) at lead side
        geometry.append(make_taper_prism(x_mmi_right, x_taper_right_end, +y_acc, Wt, Wg))
        geometry.append(make_taper_prism(x_mmi_right, x_taper_right_end, -y_acc, Wt, Wg))

        # 3. Four straight leads: from taper ends to cell edges (through PML)
        x_pml_left = -half_x
        x_pml_right = +half_x

        # Left leads
        if x_taper_left_start > x_pml_left:
            lead_len_left = x_taper_left_start - x_pml_left
            x_center_left = 0.5 * (x_pml_left + x_taper_left_start)
            geometry.append(mp.Block(
                size=self._orient_size(lead_len_left, Wg),
                center=self._orient_vec3(x_center_left, +y_acc),
                material=core,
            ))
            geometry.append(mp.Block(
                size=self._orient_size(lead_len_left, Wg),
                center=self._orient_vec3(x_center_left, -y_acc),
                material=core,
            ))

        # Right leads
        if x_taper_right_end < x_pml_right:
            lead_len_right = x_pml_right - x_taper_right_end
            x_center_right = 0.5 * (x_taper_right_end + x_pml_right)
            geometry.append(mp.Block(
                size=self._orient_size(lead_len_right, Wg),
                center=self._orient_vec3(x_center_right, +y_acc),
                material=core,
            ))
            geometry.append(mp.Block(
                size=self._orient_size(lead_len_right, Wg),
                center=self._orient_vec3(x_center_right, -y_acc),
                material=core,
            ))

        self.geometry = geometry

        # ── Port and source positions ──
        # Ports near non-PML boundary on straight leads
        port_x_left = nonpml_left + self.fit_margin_um
        port_x_right = nonpml_right - self.fit_margin_um
        port_y_offset = y_acc

        self.x_port_left_um = float(port_x_left)
        self.x_port_right_um = float(port_x_right)
        self.port_y_offset_um = float(port_y_offset)

        port_y_span = self.get_port_y_span_um()
        if self.orientation == "horizontal":
            port_size = mp.Vector3(0, port_y_span, 0)
        else:
            port_size = mp.Vector3(port_y_span, 0, 0)

        # Source positions (shifted upstream of ports, clamped inside non-PML)
        src_x_left = float(port_x_left) - float(self.source_shift_um)
        src_x_left = max(src_x_left, float(nonpml_left) + 0.1)
        src_x_right = float(port_x_right) + float(self.source_shift_um)
        src_x_right = min(src_x_right, float(nonpml_right) - 0.1)

        self.src_x_left_um = float(src_x_left)
        self.src_x_right_um = float(src_x_right)
        self.src_y_top_um = float(+port_y_offset)
        self.src_y_bot_um = float(-port_y_offset)

        # Port volumes
        self.port_1 = mp.Volume(center=self._orient_vec3(port_x_left, +port_y_offset), size=port_size)
        self.port_2 = mp.Volume(center=self._orient_vec3(port_x_left, -port_y_offset), size=port_size)
        self.port_3 = mp.Volume(center=self._orient_vec3(port_x_right, +port_y_offset), size=port_size)
        self.port_4 = mp.Volume(center=self._orient_vec3(port_x_right, -port_y_offset), size=port_size)

        # Source volumes
        self.src_1 = mp.Volume(center=self._orient_vec3(src_x_left, +port_y_offset), size=port_size)
        self.src_2 = mp.Volume(center=self._orient_vec3(src_x_left, -port_y_offset), size=port_size)
        self.src_3 = mp.Volume(center=self._orient_vec3(src_x_right, +port_y_offset), size=port_size)
        self.src_4 = mp.Volume(center=self._orient_vec3(src_x_right, -port_y_offset), size=port_size)

        self.ports = {
            "port_1": self.port_1,
            "port_2": self.port_2,
            "port_3": self.port_3,
            "port_4": self.port_4,
        }
        self.sources = {
            "src_1": self.src_1,
            "src_2": self.src_2,
            "src_3": self.src_3,
            "src_4": self.src_4,
        }

        self.full_plane = mp.Volume(
            center=mp.Vector3(0.0, 0.0, 0.0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0.0),
        )

        # Device window (active region: MMI + tapers)
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = (Lm + 2 * Lt) + 2.0 * self.fit_margin_um
        self.dev_wy = Wm + 2.0 * self.fit_margin_um

    # ── Simulation ──

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        """Run FDTD simulation and extract 4-port S-parameters."""
        if input_port not in (1, 2, 3, 4):
            raise ValueError("input_port must be 1..4.")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        # Set k-vector direction based on orientation and input port
        if self.orientation == "horizontal":
            if input_port in (1, 2):
                src_vol = self.src_1 if input_port == 1 else self.src_2
                k = mp.Vector3(+1, 0, 0)
            else:
                src_vol = self.src_3 if input_port == 3 else self.src_4
                k = mp.Vector3(-1, 0, 0)
        else:  # vertical
            if input_port in (1, 2):
                src_vol = self.src_1 if input_port == 1 else self.src_2
                k = mp.Vector3(0, +1, 0)
            else:
                src_vol = self.src_3 if input_port == 3 else self.src_4
                k = mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k,
            )
        ]

        sim = mp.Simulation(
            cell_size=self.cell,
            resolution=self.resolution,
            boundary_layers=[mp.PML(self.dpml)],
            geometry=self.geometry,
            default_material=self.clad_medium,
            sources=sources,
        )

        prop_dir = self._propagation_direction()
        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=prop_dir))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=prop_dir))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3, direction=prop_dir))
        m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_4, direction=prop_dir))

        dft_fields = sim.add_dft_fields(
            [mp.Ez, mp.Hx, mp.Hy],
            fcen, 0, 1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        res1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.NO_PARITY)
        res2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.NO_PARITY)
        res3 = sim.get_eigenmode_coefficients(m3, [1], eig_parity=mp.NO_PARITY)
        res4 = sim.get_eigenmode_coefficients(m4, [1], eig_parity=mp.NO_PARITY)

        a1_fwd, a1_bwd = res1.alpha[0, 0, 0], res1.alpha[0, 0, 1]
        a2_fwd, a2_bwd = res2.alpha[0, 0, 0], res2.alpha[0, 0, 1]
        a3_fwd, a3_bwd = res3.alpha[0, 0, 0], res3.alpha[0, 0, 1]
        a4_fwd, a4_bwd = res4.alpha[0, 0, 0], res4.alpha[0, 0, 1]

        # Forward = +x direction; for left ports incoming = fwd, for right ports incoming = bwd
        incoming = {1: a1_fwd, 2: a2_fwd, 3: a3_bwd, 4: a4_bwd}
        outgoing = {1: a1_bwd, 2: a2_bwd, 3: a3_fwd, 4: a4_fwd}

        a_in = incoming[int(input_port)]
        if abs(a_in) < 1e-12:
            sim.reset_meep()
            raise ValueError("Input mode amplitude is ~0; check geometry/ports/source placement.")

        S = {}
        for port in (1, 2, 3, 4):
            S[(port, input_port)] = outgoing[port] / a_in

        eps_mid = sim.get_epsilon().T
        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0).T
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0).T
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0).T

        if self.orientation == "vertical":
            eps_mid = eps_mid.T
            Ez_mid = Ez_mid.T
            Hx_mid = Hx_mid.T
            Hy_mid = Hy_mid.T
            cell_x_display, cell_y_display = self.cell_y, self.cell_x
        else:
            cell_x_display, cell_y_display = self.cell_x, self.cell_y

        sim.reset_meep()
        return eps_mid, Ez_mid, Hx_mid, Hy_mid, S, (cell_x_display, cell_y_display)
