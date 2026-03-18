# directional_coupler.py
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
      - Provide crop_x_px/crop_y_px, resolution, dpml and leave cell_x_um/cell_y_um None
      - With quantize_grid=True (default), dpml is snapped to integer pixels and
        cell is chosen so interior (non-PML) is exactly crop_x_px × crop_y_px.
      - Rectangular grids (crop_x_px ≠ crop_y_px) are supported for elongated devices.
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
        dpml: float = 5.0 / 7.0,
        crop_x_px: int = 336,          # target NON-PML crop in X (propagation) pixels
        crop_y_px: int = 112,          # target NON-PML crop in Y (transverse) pixels
        crop_px: int | None = None,    # deprecated: if set, overrides both crop_x_px and crop_y_px
        quantize_grid: bool = True,
        pad_y_um: float = 1.0,         # conceptual padding (not used directly in geometry)
        source_shift_um: float = 0.5,  # distance source is upstream of input port [µm]
        lead_extra_gap_um: float = 1.0,# extra separation between IO leads
        bend_length_um: float = 2.0,   # S-bend length (will be clamped to fit)
        bend_n_segments: int = 32,     # number of segments to approximate bend with
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        fit_margin_um: float = 0.5,    # safety margin from inner PML faces
        orientation: str = "horizontal",  # "horizontal" (left→right) or "vertical" (top→bottom)
    ):
        # Handle deprecated square crop_px parameter
        if crop_px is not None:
            crop_x_px = int(crop_px)
            crop_y_px = int(crop_px)
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
        self.gap_um = float(gap_um)
        self.domain_length_um = float(wg_length_um)
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
        self.src_x_right_um = None
        self.src_y_top_um = None
        self.src_y_bot_um = None

        self.build_geometry()

    def _orient_coords(self, x_local: float, y_local: float) -> tuple[float, float]:
        """Transform local (propagation, transverse) coords to Meep (x, y)."""
        if self.orientation == "horizontal":
            return float(x_local), float(y_local)
        else:  # vertical
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
        p = self.pml_px  # attribute from Device2DBase
        nx = int(np.round(self.cell_x * self.resolution)) - 2 * p
        ny = int(np.round(self.cell_y * self.resolution)) - 2 * p
        return ny, nx

    def _um_to_px(self, x_um: float, y_um: float, crop_pml: bool) -> tuple[float, float]:
        """Convert um coordinates to pixel coordinates in DISPLAY space."""
        p = self.pml_px if crop_pml else 0  # attribute from Device2DBase
        # Convert to simulation pixel coordinates
        px_x_sim = (float(x_um) + 0.5 * self.cell_x) * float(self.resolution) - float(p)
        px_y_sim = (float(y_um) + 0.5 * self.cell_y) * float(self.resolution) - float(p)

        # For vertical orientation, swap to match transposed display
        if self.orientation == "vertical":
            return px_y_sim, px_x_sim
        return px_x_sim, px_y_sim

    def get_device_window_um(self):
        """Return device window in display coordinates: (cx, cy, wx, wy)."""
        if self.orientation == "vertical":
            # Swap x and y for vertical display
            return (self.dev_cy, self.dev_cx, self.dev_wy, self.dev_wx)
        return (self.dev_cx, self.dev_cy, self.dev_wx, self.dev_wy)

    def get_display_grid_size(self):
        """Return (nx, ny) in display coordinates."""
        if self.orientation == "vertical":
            return (self.ny, self.nx)
        return (self.nx, self.ny)

    def get_display_cell_size(self):
        """Return (cell_x, cell_y) in display coordinates."""
        if self.orientation == "vertical":
            return (self.cell_y, self.cell_x)
        return (self.cell_x, self.cell_y)

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
        # Direction in display space
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
        # Direction in display space
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
        eps_2d = sim.get_epsilon()  # [nx, ny]
        eps_mid = eps_2d.T          # [ny, nx]
        sim.reset_meep()

        # For vertical orientation, transpose to display device vertically
        if self.orientation == "vertical":
            eps_mid = eps_mid.T
            cell_x_display, cell_y_display = self.cell_y, self.cell_x
        else:
            cell_x_display, cell_y_display = self.cell_x, self.cell_y

        if not crop_pml:
            return eps_mid, (cell_x_display, cell_y_display)

        p = self.pml_px  # attribute from Device2DBase
        if p <= 0:
            return eps_mid, (cell_x_display, cell_y_display)

        dpml_x = 2 * self.dpml if self.orientation == "horizontal" else 2 * self.dpml
        dpml_y = 2 * self.dpml if self.orientation == "horizontal" else 2 * self.dpml
        return eps_mid[p:-p, p:-p], (cell_x_display - dpml_x, cell_y_display - dpml_y)

    def core_mask(self, nx, ny, dx, dy):
        # For vertical orientation, swap dimensions to build mask in logical coordinates
        if self.orientation == "vertical":
            nx, ny = ny, nx
            dx, dy = dy, dx

        x = (np.arange(nx) - (nx - 1) / 2.0) * float(dx)
        y = (np.arange(ny) - (ny - 1) / 2.0) * float(dy)
        core_half = 0.5 * float(self.wg_width_um)

        center_offset = float(self.center_offset_y)
        lead_offset = float(self.lead_center_offset_y)
        x_in = float(self.x_in_um)
        x_out = float(self.x_out_um)
        L_bend = float(self.L_bend_um)

        if L_bend <= 1e-9:
            y_top = np.full_like(x, center_offset, dtype=np.float32)
        else:
            x_left_beg = float(self.x_left_beg_um)
            x_right_end = float(self.x_right_end_um)

            y_top = np.full_like(x, lead_offset, dtype=np.float32)

            # Center coupling region
            mid_mask = (x >= x_in) & (x <= x_out)
            y_top[mid_mask] = center_offset

            # Left S-bend: lead -> center
            left_mask = (x >= x_left_beg) & (x <= x_in)
            if np.any(left_mask):
                u = (x[left_mask] - x_left_beg) / L_bend
                dy_b = center_offset - lead_offset
                y_top[left_mask] = lead_offset + 0.5 * dy_b * (1.0 - np.cos(np.pi * u))

            # Right S-bend: center -> lead
            right_mask = (x >= x_out) & (x <= x_right_end)
            if np.any(right_mask):
                u = (x[right_mask] - x_out) / L_bend
                dy_b = lead_offset - center_offset
                y_top[right_mask] = center_offset + 0.5 * dy_b * (1.0 - np.cos(np.pi * u))

        yy = y[:, None]
        top = (np.abs(yy - y_top[None, :]) <= core_half)
        bot = (np.abs(yy + y_top[None, :]) <= core_half)
        mask = np.clip(top | bot, 0, 1).astype(np.uint8)

        # Transpose back for vertical orientation display
        if self.orientation == "vertical":
            mask = mask.T

        return mask

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
                size=self._orient_size(Lc, self.wg_width_um),
                center=self._orient_vec3(0.0, +center_offset_y),
                material=core,
            )
        )
        geometry.append(
            mp.Block(
                size=self._orient_size(Lc, self.wg_width_um),
                center=self._orient_vec3(0.0, -center_offset_y),
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
                        size=self._orient_size(dx * 1.05, self.wg_width_um),
                        center=self._orient_vec3(x0, y0),
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
        self.L_bend_um = float(L_bend)
        self.x_left_beg_um = float(x_in - L_bend)
        self.x_right_end_um = float(x_out + L_bend)

        # Device window: coupling + bends (exclude long straight leads)
        dev_x0 = float(self.x_left_beg_um)
        dev_x1 = float(self.x_right_end_um)
        self.dev_cx = 0.5 * (dev_x0 + dev_x1)
        self.dev_cy = 0.0
        self.dev_wx = (dev_x1 - dev_x0) + 2.0 * float(self.fit_margin_um)
        dev_y_max = max(abs(lead_center_offset_y), abs(center_offset_y)) + core_half + float(self.fit_margin_um)
        self.dev_wy = 2.0 * dev_y_max

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
                geometry.append(mp.Block(size=self._orient_size(lead_len_left, self.wg_width_um),
                                         center=self._orient_vec3(x_center_left, +center_offset_y), material=core))
                geometry.append(mp.Block(size=self._orient_size(lead_len_left, self.wg_width_um),
                                         center=self._orient_vec3(x_center_left, -center_offset_y), material=core))

            if x_out < x_pml_right:
                lead_len_right = x_pml_right - x_out
                x_center_right = 0.5 * (x_out + x_pml_right)
                geometry.append(mp.Block(size=self._orient_size(lead_len_right, self.wg_width_um),
                                         center=self._orient_vec3(x_center_right, +center_offset_y), material=core))
                geometry.append(mp.Block(size=self._orient_size(lead_len_right, self.wg_width_um),
                                         center=self._orient_vec3(x_center_right, -center_offset_y), material=core))

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
                geometry.append(mp.Block(size=self._orient_size(lead_len_left, self.wg_width_um),
                                         center=self._orient_vec3(x_center_left, +lead_center_offset_y), material=core))
                geometry.append(mp.Block(size=self._orient_size(lead_len_left, self.wg_width_um),
                                         center=self._orient_vec3(x_center_left, -lead_center_offset_y), material=core))

            if x_right_end < x_pml_right:
                lead_len_right = x_pml_right - x_right_end
                x_center_right = 0.5 * (x_right_end + x_pml_right)
                geometry.append(mp.Block(size=self._orient_size(lead_len_right, self.wg_width_um),
                                         center=self._orient_vec3(x_center_right, +lead_center_offset_y), material=core))
                geometry.append(mp.Block(size=self._orient_size(lead_len_right, self.wg_width_um),
                                         center=self._orient_vec3(x_center_right, -lead_center_offset_y), material=core))

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
        # Port size: zero along propagation direction, span across
        if self.orientation == "horizontal":
            port_size = mp.Vector3(0, port_y_span, 0)
        else:
            port_size = mp.Vector3(port_y_span, 0, 0)

        # Source plane (kept inside non-PML)
        src_x_left = float(port_x_left) - float(self.source_shift_um)
        src_x_left = max(src_x_left, float(nonpml_left) + 0.1)

        src_x_right = float(port_x_right) + float(self.source_shift_um)
        src_x_right = min(src_x_right, float(nonpml_right) - 0.1)

        self.src_x_left_um = float(src_x_left)
        self.src_x_right_um = float(src_x_right)
        self.src_y_top_um = float(+src_y_offset)
        self.src_y_bot_um = float(-src_y_offset)

        # ports
        self.port_1 = mp.Volume(center=self._orient_vec3(port_x_left, +port_y_offset), size=port_size)
        self.port_2 = mp.Volume(center=self._orient_vec3(port_x_left, -port_y_offset), size=port_size)
        self.port_3 = mp.Volume(center=self._orient_vec3(port_x_right, +port_y_offset), size=port_size)
        self.port_4 = mp.Volume(center=self._orient_vec3(port_x_right, -port_y_offset), size=port_size)

        # sources
        self.src_1 = mp.Volume(center=self._orient_vec3(src_x_left, +src_y_offset), size=port_size)
        self.src_2 = mp.Volume(center=self._orient_vec3(src_x_left, -src_y_offset), size=port_size)
        self.src_3 = mp.Volume(center=self._orient_vec3(src_x_right, +src_y_offset), size=port_size)
        self.src_4 = mp.Volume(center=self._orient_vec3(src_x_right, -src_y_offset), size=port_size)

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

        self.full_plane = mp.Volume(center=mp.Vector3(0.0, 0.0, 0.0), size=mp.Vector3(self.cell_x, self.cell_y, 0.0))

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        if input_port not in (1, 2, 3, 4):
            raise ValueError("input_port must be 1..4.")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        # Set k-vector direction based on orientation and input port
        if self.orientation == "horizontal":
            if input_port == 1:
                src_vol = self.src_1
                k = mp.Vector3(+1, 0, 0)
            elif input_port == 2:
                src_vol = self.src_2
                k = mp.Vector3(+1, 0, 0)
            elif input_port == 3:
                src_vol = self.src_3
                k = mp.Vector3(-1, 0, 0)
            else:
                src_vol = self.src_4
                k = mp.Vector3(-1, 0, 0)
        else:  # vertical
            if input_port == 1:
                src_vol = self.src_1
                k = mp.Vector3(0, +1, 0)
            elif input_port == 2:
                src_vol = self.src_2
                k = mp.Vector3(0, +1, 0)
            elif input_port == 3:
                src_vol = self.src_3
                k = mp.Vector3(0, -1, 0)
            else:
                src_vol = self.src_4
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

        incoming = {
            1: a1_fwd,
            2: a2_fwd,
            3: a3_bwd,
            4: a4_bwd,
        }
        outgoing = {
            1: a1_bwd,
            2: a2_bwd,
            3: a3_fwd,
            4: a4_fwd,
        }
        a_in = incoming[int(input_port)]
        if abs(a_in) < 1e-12:
            sim.reset_meep()
            raise ValueError("Input mode amplitude is ~0; check geometry/ports/source placement.")

        S = {}
        for port in (1, 2, 3, 4):
            S[(port, input_port)] = outgoing[port] / a_in

        eps_2d = sim.get_epsilon()
        eps_mid = eps_2d.T

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0).T
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0).T
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0).T

        # For vertical orientation, transpose to display device vertically
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
