# directional_coupler.py
import meep as mp
import numpy as np

from utils import neff_siwire_from_tables
from devices_base import Device2DBase


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
    t = ((xx - x0f) * vx + (yy - y0f) * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    px = x0f + t * vx
    py = y0f + t * vy
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    thr2 = float(thickness_px * thickness_px)
    return (d2 <= thr2).astype(np.float32)


class DirectionalCoupler2D(Device2DBase):
    """
    Symmetric 2×2 directional coupler with two parallel effective-index waveguides.

    Port numbering (looking down +x, y up):
        1: left,  top arm
        2: left,  bottom arm
        3: right, top arm
        4: right, bottom arm

    This version supports deterministic pixel-exact crops like EulerZigZag2D:
      - Provide crop_px, resolution, dpml and leave cell_x_um/cell_y_um None
      - With quantize_grid=True (default), dpml is snapped to integer pixels and
        cell is chosen so interior (non-PML) is exactly crop_px×crop_px.
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        gap_um: float = 0.2,           # edge–to–edge gap between the two guides [µm]
        wg_length_um: float = 10.0,    # coupling region length (centered at x=0) [µm]
        wavelength_um: float = 1.55,
        resolution: int = 20,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 2.0 / 3.0,
        crop_px: int = 384,            # target NON-PML crop in pixels
        quantize_grid: bool = True,
        pad_y_um: float = 1.0,         # conceptual padding (not used directly in geometry)
        source_shift_um: float = 0.5,  # distance source is upstream of input port [µm]
        lead_extra_gap_um: float = 1.0,# extra separation between IO leads
        bend_length_um: float = 2.0,   # S-bend length (will be clamped to fit)
        bend_n_segments: int = 32,     # number of segments to approximate bend with
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        fit_margin_um: float = 0.5,    # safety margin from inner PML faces
    ):
        crop_px = int(crop_px)
        if crop_px <= 0:
            raise ValueError("crop_px must be > 0")
        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        self.crop_px = crop_px
        self.fit_margin_um = float(fit_margin_um)

        # Quantize dpml + cell so (crop_px + 2*pml_px) is integer pixels.
        if bool(quantize_grid) and (cell_x_um is None or cell_y_um is None):
            pml_px = int(np.round(float(dpml) * float(resolution)))
            dpml_q = float(pml_px) / float(resolution)
            full_px = int(crop_px + 2 * pml_px)
            cell_um_q = float(full_px) / float(resolution)
            cx = cell_um_q if cell_x_um is None else float(cell_x_um)
            cy = cell_um_q if cell_y_um is None else float(cell_y_um)
            dpml_use = dpml_q
        else:
            cx = float(cell_x_um) if cell_x_um is not None else (float(crop_px) / float(resolution) + 2.0 * float(dpml))
            cy = float(cell_y_um) if cell_y_um is not None else (float(crop_px) / float(resolution) + 2.0 * float(dpml))
            dpml_use = float(dpml)

        super().__init__(cell_x_um=float(cx), cell_y_um=float(cy), dpml=float(dpml_use), resolution=int(resolution))

        self.wg_width_um = float(wg_width_um)
        self.gap_um = float(gap_um)
        self.domain_length_um = float(wg_length_um)
        self.wavelength_um = float(wavelength_um)

        if n_core is None:
            self.n_core = float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
        else:
            self.n_core = float(n_core)

        self.n_clad = float(n_clad)
        self.pad_y_um = float(pad_y_um)
        self.source_shift_um = float(source_shift_um)

        self.coupling_length_um = float(wg_length_um)
        self.lead_extra_gap_um = float(lead_extra_gap_um)
        self.bend_length_um = float(bend_length_um)
        self.bend_n_segments = int(bend_n_segments)

        # to be filled
        self.geometry = None
        self.port_1 = self.port_2 = self.port_3 = self.port_4 = None
        self.src_1 = self.src_2 = None
        self.full_plane = None
        self.clad_medium = None

        self.x_port_left_um = None
        self.x_port_right_um = None
        self.port_y_offset_um = None

        self.src_x_left_um = None
        self.src_y_top_um = None
        self.src_y_bot_um = None

        self.build_geometry()

    def pml_px(self) -> int:
        return int(np.round(float(self.dpml) * float(self.resolution)))

    def nonpml_shape(self) -> tuple[int, int]:
        p = self.pml_px()
        nx = int(np.round(self.cell_x * self.resolution)) - 2 * p
        ny = int(np.round(self.cell_y * self.resolution)) - 2 * p
        return ny, nx

    def _um_to_px(self, x_um: float, y_um: float, crop_pml: bool) -> tuple[float, float]:
        p = self.pml_px() if crop_pml else 0
        px_x = (float(x_um) + 0.5 * self.cell_x) * float(self.resolution) - float(p)
        px_y = (float(y_um) + 0.5 * self.cell_y) * float(self.resolution) - float(p)
        return px_x, px_y

    def get_port_centers_um(self):
        yoff = float(self.port_y_offset_um)
        return {
            1: (float(self.x_port_left_um), +yoff),
            2: (float(self.x_port_left_um), -yoff),
            3: (float(self.x_port_right_um), +yoff),
            4: (float(self.x_port_right_um), -yoff),
        }

    def get_port_y_span_um(self):
        # just big enough to capture one arm, not both
        return float(self.wg_width_um + 0.5 * self.gap_um)

    def get_port_region_px(self, port: int, crop_pml: bool = True) -> dict:
        if port not in (1, 2, 3, 4):
            raise ValueError("port must be 1..4")
        centers = self.get_port_centers_um()
        x0, y0 = centers[int(port)]
        span = self.get_port_y_span_um()
        half = 0.5 * span
        # Port plane is a vertical line segment at x=x0
        x_px, y_px = self._um_to_px(x0, y0, crop_pml)
        x_s, y_s = self._um_to_px(x0, y0 - half, crop_pml)
        x_e, y_e = self._um_to_px(x0, y0 + half, crop_pml)
        return {
            "center_px": (x_px, y_px),
            "line_start_px": (x_s, y_s),
            "line_end_px": (x_e, y_e),
            "direction_px": (1.0, 0.0),  # propagation axis is ±x; monitor plane normal is x
        }

    def get_source_region_px(self, input_port: int = 1, crop_pml: bool = True) -> dict:
        if input_port not in (1, 2):
            raise ValueError("input_port must be 1 or 2")
        x0 = float(self.src_x_left_um)
        y0 = float(self.src_y_top_um if input_port == 1 else self.src_y_bot_um)
        span = self.get_port_y_span_um()
        half = 0.5 * span
        x_px, y_px = self._um_to_px(x0, y0, crop_pml)
        x_s, y_s = self._um_to_px(x0, y0 - half, crop_pml)
        x_e, y_e = self._um_to_px(x0, y0 + half, crop_pml)
        return {
            "center_px": (x_px, y_px),
            "line_start_px": (x_s, y_s),
            "line_end_px": (x_e, y_e),
            "direction_px": (1.0, 0.0),
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
        eps_2d = sim.get_epsilon()  # [nx, ny]
        eps_mid = eps_2d.T          # [ny, nx]
        sim.reset_meep()

        if not crop_pml:
            return eps_mid, (self.cell_x, self.cell_y)

        p = self.pml_px()
        if p <= 0:
            return eps_mid, (self.cell_x, self.cell_y)
        return eps_mid[p:-p, p:-p], (self.cell_x - 2 * self.dpml, self.cell_y - 2 * self.dpml)

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        core_half = 0.5 * self.wg_width_um
        center_offset_y = 0.5 * (self.wg_width_um + self.gap_um)
        self.center_offset_y = center_offset_y

        lead_center_offset_y = center_offset_y + 0.5 * self.lead_extra_gap_um
        self.lead_center_offset_y = lead_center_offset_y

        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = +half_y - self.dpml

        # Quick fit sanity (y)
        y_max_core = max(abs(lead_center_offset_y), abs(center_offset_y)) + core_half + self.fit_margin_um
        if y_max_core > min(abs(nonpml_top), abs(nonpml_bot)):
            raise ValueError("DirectionalCoupler2D does not fit in non-PML region (y). Reduce lead_extra_gap/gap/wg_width or increase crop/cell.")

        # Coupling region centered at x=0
        Lc = float(self.coupling_length_um)
        x_in = -0.5 * Lc
        x_out = +0.5 * Lc
        self.x_in_um = x_in
        self.x_out_um = x_out

        # Must fit coupling region inside non-PML with margins
        if (x_in < nonpml_left + self.fit_margin_um) or (x_out > nonpml_right - self.fit_margin_um):
            raise ValueError("DirectionalCoupler2D coupling_length_um does not fit inside non-PML region. Reduce wg_length_um or increase crop/cell.")

        geometry = []

        # central parallel guides
        geometry.append(
            mp.Block(
                size=mp.Vector3(Lc, self.wg_width_um, mp.inf),
                center=mp.Vector3(0.0, +center_offset_y, 0.0),
                material=core,
            )
        )
        geometry.append(
            mp.Block(
                size=mp.Vector3(Lc, self.wg_width_um, mp.inf),
                center=mp.Vector3(0.0, -center_offset_y, 0.0),
                material=core,
            )
        )

        def s_bend_blocks(y_start: float, y_end: float, x_start: float, x_end: float, n_seg: int):
            dx = (x_end - x_start) / float(n_seg)
            dy = (y_end - y_start)
            L = (x_end - x_start)
            blocks = []
            for i in range(int(n_seg)):
                u = (i + 0.5) / float(n_seg)
                x0 = x_start + u * L
                y0 = y_start + 0.5 * dy * (1.0 - np.cos(np.pi * u))
                blocks.append(
                    mp.Block(
                        size=mp.Vector3(dx * 1.05, self.wg_width_um, mp.inf),
                        center=mp.Vector3(x0, y0, 0.0),
                        material=core,
                    )
                )
            return blocks

        # Keep bends strictly inside non-PML with a safety margin
        x_left_lim = nonpml_left + self.fit_margin_um
        x_right_lim = nonpml_right - self.fit_margin_um

        # Clamp bend length to available space
        max_left = x_in - x_left_lim
        max_right = x_right_lim - x_out
        L_bend = float(min(self.bend_length_um, max_left, max_right))

        # Default ports at coupling edges (if no bends)
        port_x_left = x_in
        port_x_right = x_out
        port_y_offset = center_offset_y
        src_y_offset = center_offset_y

        if L_bend <= 1e-9:
            # Extend straight leads through PML
            x_pml_left = -half_x
            x_pml_right = +half_x

            if x_in > x_pml_left:
                lead_len_left = x_in - x_pml_left
                x_center_left = 0.5 * (x_pml_left + x_in)
                geometry.append(mp.Block(size=mp.Vector3(lead_len_left, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_left, +center_offset_y, 0.0), material=core))
                geometry.append(mp.Block(size=mp.Vector3(lead_len_left, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_left, -center_offset_y, 0.0), material=core))

            if x_out < x_pml_right:
                lead_len_right = x_pml_right - x_out
                x_center_right = 0.5 * (x_out + x_pml_right)
                geometry.append(mp.Block(size=mp.Vector3(lead_len_right, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_right, +center_offset_y, 0.0), material=core))
                geometry.append(mp.Block(size=mp.Vector3(lead_len_right, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_right, -center_offset_y, 0.0), material=core))

            self.geometry = geometry
        else:
            x_left_beg = x_in - L_bend
            x_left_end = x_in
            x_right_beg = x_out
            x_right_end = x_out + L_bend

            # left: wide lead -> close coupling
            geometry += s_bend_blocks(+lead_center_offset_y, +center_offset_y, x_left_beg, x_left_end, self.bend_n_segments)
            geometry += s_bend_blocks(-lead_center_offset_y, -center_offset_y, x_left_beg, x_left_end, self.bend_n_segments)

            # right: close coupling -> wide lead
            geometry += s_bend_blocks(+center_offset_y, +lead_center_offset_y, x_right_beg, x_right_end, self.bend_n_segments)
            geometry += s_bend_blocks(-center_offset_y, -lead_center_offset_y, x_right_beg, x_right_end, self.bend_n_segments)

            # straight leads to PML boundary
            x_pml_left = -half_x
            x_pml_right = +half_x

            if x_left_beg > x_pml_left:
                lead_len_left = x_left_beg - x_pml_left
                x_center_left = 0.5 * (x_pml_left + x_left_beg)
                geometry.append(mp.Block(size=mp.Vector3(lead_len_left, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_left, +lead_center_offset_y, 0.0), material=core))
                geometry.append(mp.Block(size=mp.Vector3(lead_len_left, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_left, -lead_center_offset_y, 0.0), material=core))

            if x_right_end < x_pml_right:
                lead_len_right = x_pml_right - x_right_end
                x_center_right = 0.5 * (x_right_end + x_pml_right)
                geometry.append(mp.Block(size=mp.Vector3(lead_len_right, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_right, +lead_center_offset_y, 0.0), material=core))
                geometry.append(mp.Block(size=mp.Vector3(lead_len_right, self.wg_width_um, mp.inf),
                                         center=mp.Vector3(x_center_right, -lead_center_offset_y, 0.0), material=core))

            self.geometry = geometry

            # With bends, put ports on straight leads near non-PML boundary
            desired_margin = self.fit_margin_um
            candidate_left = nonpml_left + desired_margin
            max_left_before_bend = x_left_beg - 0.2
            port_x_left = min(candidate_left, max_left_before_bend)
            port_x_right = -port_x_left

            port_y_offset = lead_center_offset_y
            src_y_offset = lead_center_offset_y

        self.x_port_left_um = float(port_x_left)
        self.x_port_right_um = float(port_x_right)
        self.port_y_offset_um = float(port_y_offset)

        port_y_span = self.get_port_y_span_um()
        port_size = mp.Vector3(0, port_y_span, 0)

        # Source plane (kept inside non-PML)
        src_x_left = float(port_x_left) - float(self.source_shift_um)
        src_x_left = max(src_x_left, float(nonpml_left) + 0.1)

        self.src_x_left_um = float(src_x_left)
        self.src_y_top_um = float(+src_y_offset)
        self.src_y_bot_um = float(-src_y_offset)

        # ports
        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, +port_y_offset, 0.0), size=port_size)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_left, -port_y_offset, 0.0), size=port_size)
        self.port_3 = mp.Volume(center=mp.Vector3(port_x_right, +port_y_offset, 0.0), size=port_size)
        self.port_4 = mp.Volume(center=mp.Vector3(port_x_right, -port_y_offset, 0.0), size=port_size)

        # sources
        self.src_1 = mp.Volume(center=mp.Vector3(src_x_left, +src_y_offset, 0.0), size=port_size)
        self.src_2 = mp.Volume(center=mp.Vector3(src_x_left, -src_y_offset, 0.0), size=port_size)

        self.full_plane = mp.Volume(center=mp.Vector3(0.0, 0.0, 0.0), size=mp.Vector3(self.cell_x, self.cell_y, 0.0))

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        if input_port not in (1, 2):
            raise ValueError("input_port must be 1 (top left) or 2 (bottom left).")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        src_vol = self.src_1 if input_port == 1 else self.src_2
        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
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

        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=mp.X))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=mp.X))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3, direction=mp.X))
        m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_4, direction=mp.X))

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

        if input_port == 1:
            a_in = a1_fwd
        else:
            a_in = a2_fwd
        if abs(a_in) < 1e-12:
            sim.reset_meep()
            raise ValueError("Input mode amplitude is ~0; check geometry/ports/source placement.")

        S = {}
        S[(1, input_port)] = a1_bwd / a_in
        S[(2, input_port)] = a2_bwd / a_in
        S[(3, input_port)] = a3_fwd / a_in
        S[(4, input_port)] = a4_fwd / a_in

        eps_2d = sim.get_epsilon()
        eps_mid = eps_2d.T

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0).T
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0).T
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0).T

        sim.reset_meep()
        return eps_mid, Ez_mid, Hx_mid, Hy_mid, S, (self.cell_x, self.cell_y)
