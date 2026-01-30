# ybranch.py
from __future__ import annotations

import meep as mp
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import neff_siwire_from_tables, rect_mask
from devices_base import Device2DBase


def _draw_thick_line_mask(
    ny: int,
    nx: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    thickness_px: int = 3,
) -> np.ndarray:
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


class YBranch2D(Device2DBase):
    """
    Symmetric 1×2 Y-junction / power splitter (effective-index 2D).

    Port numbering (looking down +x, y up):

        1: left  input (single stem, centered at y=0)
        2: right top output arm
        3: right bottom output arm

    This follows the geometry idea used in the Tidy3D example:
      - A *junction* region modeled as a tapered polygon whose width varies along x
      - Then two output S-bends that separate vertically (sine-profile)
    :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wavelength_um: float = 1.55,
        resolution: int | None = None,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 1.0,
        pad_y_um: float = 2.0,
        source_shift_um: float = 0.5,

        # --- Geometry lengths (roughly like the Tidy3D notebook) ---
        l_in_um: float = 1.0,            # straight stem length before junction
        l_junction_um: float = 2.0,      # junction (taper polygon) length
        l_bend_um: float = 6.0,          # horizontal length of the output S-bends
        h_bend_um: float = 2.0,          # vertical offset added by the output bends
        l_out_um: float = 1.0,           # straight output length after bends

        # Junction width profile (13-segment optimized example, scaled)
        junction_widths_um: list[float] | None = None,
        junction_n_segments: int = 13,
        use_prism_junction: bool = True,

        bend_n_segments: int = 80,

        # Euler-aligned sizing (optional):
        crop_px: int | None = None,        # target non-PML crop in pixels (square)
        quantize_grid: bool = True,        # snap dpml to integer pixels when crop_px is used
        fit_margin_um: float = 0.5,        # safety margin from inner PML faces
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,

        port_y_span_um: float | None = None,
        device_center_x_um: float | None = None,
        orientation: str = "horizontal",  # "horizontal" (left→right) or "vertical" (top→bottom)
    ):
        crop_px_i = None if crop_px is None else int(crop_px)
        if crop_px_i is not None and crop_px_i <= 0:
            raise ValueError("crop_px must be > 0")
        resolution_i = int(32 if resolution is None else resolution)
        if resolution_i <= 0:
            raise ValueError("resolution must be > 0")

        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        self.orientation = str(orientation)

        # Quantize dpml + cell so (crop_px + 2*pml_px) is integer pixels (Euler-style).
        if crop_px_i is not None and bool(quantize_grid) and (cell_x_um is None or cell_y_um is None):
            pml_px = int(np.round(float(dpml) * float(resolution_i)))
            dpml_q = float(pml_px) / float(resolution_i)
            full_px = int(crop_px_i + 2 * pml_px)
            cell_um_q = float(full_px) / float(resolution_i)
            cx = cell_um_q if cell_x_um is None else float(cell_x_um)
            cy = cell_um_q if cell_y_um is None else float(cell_y_um)
            dpml_use = dpml_q
        else:
            cx = float(cell_x_um) if cell_x_um is not None else None
            cy = float(cell_y_um) if cell_y_um is not None else None
            dpml_use = float(dpml)

        super().__init__(cell_x=cx, cell_y=cy, dpml=float(dpml_use), resolution=int(resolution_i))

        self.crop_px = crop_px_i
        self.fit_margin_um = float(fit_margin_um)

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.resolution = int(self.resolution)

        if n_core is None:
            self.n_core = neff_siwire_from_tables(self.wg_width_um, self.wavelength_um)
        else:
            self.n_core = float(n_core)

        self.n_clad = float(n_clad)
        self.pad_y_um = float(pad_y_um)
        self.source_shift_um = float(source_shift_um)

        self.l_in_um = float(l_in_um)
        self.l_junction_um = float(l_junction_um)
        self.l_bend_um = float(l_bend_um)
        self.h_bend_um = float(h_bend_um)
        self.l_out_um = float(l_out_um)

        self.use_prism_junction = bool(use_prism_junction)
        self.bend_n_segments = int(bend_n_segments)
        self.junction_n_segments = int(junction_n_segments)
        self.port_y_span_um = port_y_span_um
        self._port_y_span_um = None
        self.device_center_x_um = device_center_x_um

        # cached port y-locations (set in build_geometry)
        self.y_port_1_um = 0.0
        self.y_port_2_um = None
        self.y_port_3_um = None

        # default junction widths: the Tidy3D notebook’s 13 widths, scaled by wg_width/0.5
        # (their w1=0.5 um in the example) :contentReference[oaicite:2]{index=2}
        if junction_widths_um is None:
            base = np.array([0.5, 0.5, 0.6, 0.7, 0.9, 1.26, 1.4, 1.4, 1.4, 1.4, 1.31, 1.2, 1.2], dtype=float)
            scale = self.wg_width_um / 0.5
            widths = (base * scale).tolist()
        else:
            widths = [float(w) for w in junction_widths_um]

        # force junction start width to match input guide width (keeps continuity)
        if len(widths) >= 1:
            widths[0] = self.wg_width_um
        self.junction_widths_um = widths

        # filled by build_geometry()
        self.geometry = None
        self.clad_medium = None

        self.port_1 = None
        self.port_2 = None
        self.port_3 = None

        self.src_1 = None
        self.src_2 = None
        self.src_3 = None
        self.full_plane = None

        self.x_port_left_um = None
        self.x_port_right_um = None

        # results
        self.eps_mid = None
        self.Ez_mid = None
        self.Hx_mid = None
        self.Hy_mid = None
        self.S = None

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

    def _device_x0(self, x_npml_left: float, x_npml_right: float, margin: float, L_req: float) -> float:
        if self.device_center_x_um is None:
            return (x_npml_right - margin) - L_req
        x0 = float(self.device_center_x_um) - 0.5 * L_req
        if (x0 < x_npml_left + margin) or (x0 + L_req > x_npml_right - margin):
            raise ValueError(
                "YBranch2D device_center_x_um places device outside non-PML region. "
                "Reduce the shift, increase cell_x, or reduce l_in/l_junction/l_bend/l_out."
            )
        return x0

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

    def get_port_region_px(self, port: int, crop_pml: bool = True) -> dict:
        if port not in (1, 2, 3):
            raise ValueError("port must be 1..3")
        centers = self.get_port_centers_um()
        x0, y0 = centers[int(port)]
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

    def get_source_region_px(self, input_port: int = 1, crop_pml: bool = True) -> dict:
        """
        Get source region in pixel coordinates.

        input_port=1: splitter mode (excite from stem)
        input_port=2: combiner mode (excite from top arm)
        input_port=3: combiner mode (excite from bottom arm)
        """
        if input_port not in (1, 2, 3):
            raise ValueError("YBranch2D input_port must be 1, 2, or 3.")

        half_x = 0.5 * self.cell_x
        nonpml_left = -half_x + float(self.dpml)
        nonpml_right = half_x - float(self.dpml)
        span = self.get_port_y_span_um()
        half = 0.5 * span

        if input_port == 1:
            # Splitter mode: source to the left of port 1
            x0 = float(self.x_port_left_um) - float(self.source_shift_um)
            x0 = max(x0, nonpml_left + 0.1)
            y0 = 0.0
        elif input_port == 2:
            # Combiner mode: source to the right of port 2 (top arm)
            x0 = float(self.x_port_right_um) + float(self.source_shift_um)
            x0 = min(x0, nonpml_right - 0.1)
            y0 = float(self.y_port_2_um)
        else:  # input_port == 3
            # Combiner mode: source to the right of port 3 (bottom arm)
            x0 = float(self.x_port_right_um) + float(self.source_shift_um)
            x0 = min(x0, nonpml_right - 0.1)
            y0 = float(self.y_port_3_um)

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

    def make_line_mask_from_region_px(self, ny: int, nx: int, region_px: dict, thickness_px: int = 3) -> np.ndarray:
        (x0, y0) = region_px["line_start_px"]
        (x1, y1) = region_px["line_end_px"]
        return _draw_thick_line_mask(ny, nx, x0, y0, x1, y1, thickness_px=int(thickness_px)).astype(np.float32)

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        geometry = []

        # non-PML x-range
        half_cell = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        x_pml_left = -half_cell
        x_pml_right = half_cell
        x_npml_left = x_pml_left + self.dpml
        x_npml_right = x_pml_right - self.dpml
        y_npml_bot = -half_y + self.dpml
        y_npml_top = +half_y - self.dpml

        margin = float(self.fit_margin_um)
        L_req = self.l_in_um + self.l_junction_um + self.l_bend_um + self.l_out_um
        avail = (x_npml_right - margin) - (x_npml_left + margin)
        if L_req > avail:
            raise ValueError(
                f"YBranch2D does not fit in cell_x={self.cell_x:.3g}um (needs {L_req:.3g}um, has {avail:.3g}um). "
                "Increase cell_x_um or reduce l_in/l_junction/l_bend/l_out."
            )

        # place device horizontally (default: right-aligned with margin)
        x0 = self._device_x0(x_npml_left, x_npml_right, margin, L_req)  # device left reference
        x_in_end = x0 + self.l_in_um
        x_j0 = x_in_end
        x_j1 = x_j0 + self.l_junction_um
        x_b0 = x_j1
        x_b1 = x_b0 + self.l_bend_um
        x_out0 = x_b1

        # straight input stem from left PML edge to junction start
        if x_j0 > x_pml_left:
            stem_len = x_j0 - x_pml_left
            stem_center = 0.5 * (x_pml_left + x_j0)
            geometry.append(
                mp.Block(
                    size=self._orient_size(stem_len, self.wg_width_um),
                    center=self._orient_vec3(stem_center, 0.0),
                    material=core,
                )
            )

        # --- junction polygon (tapered width along x) ---
        widths = self.junction_widths_um
        Nw = len(widths)

        if Nw < 2:
            # degenerate: just treat as straight
            pass
        else:
            x_top = np.linspace(x_j0, x_j1, Nw)
            y_top = 0.5 * np.array(widths)

            # polygon vertices: top edge left->right, then bottom edge right->left
            verts = []
            for i in range(Nw):
                verts.append(self._orient_vec3(float(x_top[i]), float(y_top[i])))
            for i in range(Nw - 1, -1, -1):
                verts.append(self._orient_vec3(float(x_top[i]), float(-y_top[i])))

            if self.use_prism_junction:
                try:
                    geometry.append(
                        mp.Prism(
                            vertices=verts,
                            height=mp.inf,
                            axis=mp.Z,
                            material=core,
                        )
                    )
                except Exception:
                    # fallback: approximate junction with overlapping blocks (segmented taper)
                    dx = (x_j1 - x_j0) / (Nw - 1)
                    for i in range(Nw - 1):
                        xc = 0.5 * (x_top[i] + x_top[i + 1])
                        wseg = 0.5 * (widths[i] + widths[i + 1])
                        geometry.append(
                            mp.Block(
                                size=self._orient_size(dx * 1.05, wseg),
                                center=self._orient_vec3(float(xc), 0.0),
                                material=core,
                            )
                        )
            else:
                # explicit segmented taper
                dx = (x_j1 - x_j0) / (Nw - 1)
                for i in range(Nw - 1):
                    xc = 0.5 * (x_top[i] + x_top[i + 1])
                    wseg = 0.5 * (widths[i] + widths[i + 1])
                    geometry.append(
                        mp.Block(
                            size=self._orient_size(dx * 1.05, wseg),
                            center=self._orient_vec3(float(xc), 0.0),
                            material=core,
                        )
                    )

        # --- output S-bends (sine profile like the notebook) ---
        # The Tidy3D example uses a sine-based S-bend to separate arms. :contentReference[oaicite:3]{index=3}
        w_in = self.wg_width_um
        w_end = float(widths[-1]) if len(widths) else w_in
        y_base = 0.5 * (w_end - w_in)  # aligns bend start edge with junction edge

        def sine_s_bend_blocks(y0: float, h: float, x_start: float, x_end: float, n_seg: int):
            """
            Centerline (their form, written in our coords):
                s = x - x_start in [0, L]
                y(s) = y0 + s*h/L - h*sin(2π s/L)/(2π)
            This has zero slope at both ends.
            :contentReference[oaicite:4]{index=4}
            """
            L = x_end - x_start
            dx = L / n_seg
            blocks = []
            for i in range(n_seg):
                u = (i + 0.5) / n_seg
                x = x_start + u * L
                s = u * L
                y = y0 + (s * h / L) - (h * np.sin(2 * np.pi * s / L) / (2 * np.pi))
                blocks.append(
                    mp.Block(
                        size=self._orient_size(dx * 1.05, w_in),
                        center=self._orient_vec3(float(x), float(y)),
                        material=core,
                    )
                )
            return blocks

        geometry += sine_s_bend_blocks(+y_base, +self.h_bend_um, x_b0, x_b1, self.bend_n_segments)
        geometry += sine_s_bend_blocks(-y_base, -self.h_bend_um, x_b0, x_b1, self.bend_n_segments)

        # straight outputs from end of bends into right PML
        y_out_top = +y_base + self.h_bend_um
        y_out_bot = -y_base - self.h_bend_um
        self.y_port_2_um = float(y_out_top)
        self.y_port_3_um = float(y_out_bot)

        # Device window: junction + bends (exclude long straight leads)
        dev_x0 = float(x_j0)
        dev_x1 = float(x_b1)
        max_width = max(widths) if len(widths) else w_in
        dev_y_max = max(0.5 * float(max_width), abs(y_out_top) + 0.5 * float(w_in)) + margin
        self.dev_cx = 0.5 * (dev_x0 + dev_x1)
        self.dev_cy = 0.0
        self.dev_wx = (dev_x1 - dev_x0) + 2.0 * margin
        self.dev_wy = 2.0 * dev_y_max

        # Y fit sanity (non-PML)
        y_max_core = max(abs(y_out_top), abs(y_out_bot)) + 0.5 * float(self.wg_width_um) + margin
        if y_max_core > min(abs(y_npml_top), abs(y_npml_bot)):
            raise ValueError(
                "YBranch2D does not fit in non-PML region (y). Reduce h_bend/pad_y or increase crop/cell."
            )

        if x_out0 < x_pml_right:
            out_len = x_pml_right - x_out0
            out_center = 0.5 * (x_out0 + x_pml_right)
            geometry.append(
                mp.Block(
                    size=self._orient_size(out_len, w_in),
                    center=self._orient_vec3(float(out_center), float(y_out_top)),
                    material=core,
                )
            )
            geometry.append(
                mp.Block(
                    size=self._orient_size(out_len, w_in),
                    center=self._orient_vec3(float(out_center), float(y_out_bot)),
                    material=core,
                )
            )

        self.geometry = geometry

        # --- Ports ---
        port_margin = 0.8
        port_x_left = x_npml_left + port_margin
        port_x_left = min(port_x_left, x_j0 - 0.25)  # keep before junction
        port_x_right = x_npml_right - port_margin
        port_x_right = max(port_x_right, x_out0 + 0.25)  # keep after bends

        self.x_port_left_um = float(port_x_left)
        self.x_port_right_um = float(port_x_right)

        if self.port_y_span_um is None:
            port_y_span = w_in + 0.4
        else:
            port_y_span = float(self.port_y_span_um)
        self._port_y_span_um = float(port_y_span)

        # Port size: zero along propagation direction, span across
        if self.orientation == "horizontal":
            port_size = mp.Vector3(0, port_y_span, 0)
        else:
            port_size = mp.Vector3(port_y_span, 0, 0)

        self.port_1 = mp.Volume(center=self._orient_vec3(port_x_left, 0.0), size=port_size)
        self.port_2 = mp.Volume(center=self._orient_vec3(port_x_right, float(y_out_top)), size=port_size)
        self.port_3 = mp.Volume(center=self._orient_vec3(port_x_right, float(y_out_bot)), size=port_size)

        # source upstream of port 1, but inside non-PML
        src_x = port_x_left - self.source_shift_um
        min_src_x = x_npml_left + 0.1
        if src_x < min_src_x:
            src_x = min_src_x
        self.src_1 = mp.Volume(center=self._orient_vec3(float(src_x), 0.0), size=port_size)

        # sources for ports 2 and 3 (combiner mode: input from output arms)
        # Source is downstream (to the right) of port, inside non-PML
        src_x_right = port_x_right + self.source_shift_um
        max_src_x = x_npml_right - 0.1
        if src_x_right > max_src_x:
            src_x_right = max_src_x
        self.src_2 = mp.Volume(center=self._orient_vec3(float(src_x_right), float(y_out_top)), size=port_size)
        self.src_3 = mp.Volume(center=self._orient_vec3(float(src_x_right), float(y_out_bot)), size=port_size)

        # full-plane DFT
        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

    def core_mask(self, nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
        """
        Return a uint8 mask of the waveguide core regions for overlay plots.
        Mirrors the build_geometry centerline logic with segmented rectangles.
        """
        # For vertical orientation, swap dimensions to build mask in logical coordinates
        nx_orig, ny_orig = nx, ny
        dx_orig, dy_orig = dx, dy
        if self.orientation == "vertical":
            nx, ny = ny_orig, nx_orig
            dx, dy = dy_orig, dx_orig

        mask = np.zeros((ny, nx), dtype=np.uint8)

        def add_rect(cx: float, cy: float, wx: float, wy: float):
            if wx <= 0 or wy <= 0:
                return
            mask[:] = np.maximum(mask, rect_mask(nx, ny, dx, dy, cx, cy, wx, wy))

        w_in = float(self.wg_width_um)

        # replicate placement from build_geometry
        half_cell = 0.5 * self.cell_x
        x_pml_left = -half_cell
        x_pml_right = half_cell
        x_npml_left = x_pml_left + self.dpml
        x_npml_right = x_pml_right - self.dpml

        margin = float(self.fit_margin_um)
        L_req = self.l_in_um + self.l_junction_um + self.l_bend_um + self.l_out_um
        x0 = self._device_x0(x_npml_left, x_npml_right, margin, L_req)
        x_in_end = x0 + self.l_in_um
        x_j0 = x_in_end
        x_j1 = x_j0 + self.l_junction_um
        x_b0 = x_j1
        x_b1 = x_b0 + self.l_bend_um
        x_out0 = x_b1

        # input stem
        if x_j0 > x_pml_left:
            stem_len = x_j0 - x_pml_left
            stem_center = 0.5 * (x_pml_left + x_j0)
            add_rect(stem_center, 0.0, stem_len, w_in)

        # junction taper as segmented blocks
        widths = self.junction_widths_um
        Nw = len(widths)
        if Nw >= 2:
            dx_seg = (x_j1 - x_j0) / (Nw - 1)
            for i in range(Nw - 1):
                xc = 0.5 * (x_j0 + i * dx_seg + x_j0 + (i + 1) * dx_seg)
                wseg = 0.5 * (widths[i] + widths[i + 1])
                add_rect(float(xc), 0.0, float(dx_seg) * 1.05, float(wseg))

        # s-bends as centerlines
        w_end = float(widths[-1]) if len(widths) else w_in
        y_base = 0.5 * (w_end - w_in)

        def sine_y(u: float, y0: float, h: float):
            L = (x_b1 - x_b0)
            s = u * L
            return y0 + (s * h / L) - (h * np.sin(2 * np.pi * s / L) / (2 * np.pi))

        for sign in (+1.0, -1.0):
            y0 = sign * y_base
            h = sign * self.h_bend_um
            for i in range(self.bend_n_segments):
                u0 = i / self.bend_n_segments
                u1 = (i + 1) / self.bend_n_segments
                x0s = x_b0 + u0 * (x_b1 - x_b0)
                x1s = x_b0 + u1 * (x_b1 - x_b0)
                y0s = sine_y(u0, y0, h)
                y1s = sine_y(u1, y0, h)
                xmid = 0.5 * (x0s + x1s)
                ymid = 0.5 * (y0s + y1s)
                add_rect(float(xmid), float(ymid), float(x1s - x0s) * 1.05, w_in)

        # output straights
        y_out_top = +y_base + self.h_bend_um
        y_out_bot = -y_base - self.h_bend_um
        if x_out0 < x_pml_right:
            out_len = x_pml_right - x_out0
            add_rect(0.5 * (x_out0 + x_pml_right), float(y_out_top), out_len, w_in)
            add_rect(0.5 * (x_out0 + x_pml_right), float(y_out_bot), out_len, w_in)

        mask = mask.astype(np.uint8)

        # Transpose back for vertical orientation display
        if self.orientation == "vertical":
            mask = mask.T

        return mask

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
        """
        Return port centers as {port_id: (x_um, y_um)}.
        Ports:
          1: input (left)
          2: top output (right)
          3: bottom output (right)
        """
        if self.x_port_left_um is None or self.x_port_right_um is None:
            raise RuntimeError("Ports have not been initialized; build_geometry() has not run.")
        if self.y_port_2_um is None or self.y_port_3_um is None:
            raise RuntimeError("Port y-locations have not been initialized; build_geometry() has not run.")

        return {
            1: (float(self.x_port_left_um), float(self.y_port_1_um)),
            2: (float(self.x_port_right_um), float(self.y_port_2_um)),
            3: (float(self.x_port_right_um), float(self.y_port_3_um)),
        }

    def get_port_y_span_um(self):
        """
        Cross-section height used for port monitors / source masks.
        """
        if self._port_y_span_um is None:
            # conservative fallback
            return float(self.wg_width_um + 0.4)
        return float(self._port_y_span_um)

    def get_eps_and_cell(self, crop_pml: bool = False):
        """
        Return the permittivity grid (ny, nx) and (cell_x, cell_y) without running a source.

        This builds a Simulation with no sources, initializes it, and extracts epsilon.
        It's fast compared to a full time-domain run.
        """
        sim = mp.Simulation(
            cell_size=self.cell,
            resolution=self.resolution,
            boundary_layers=[mp.PML(self.dpml)],
            geometry=self.geometry,
            default_material=self.clad_medium,
            sources=[],
        )

        # Initialize fields / geometry discretization without advancing time
        sim.init_sim()

        # Meep returns epsilon as [nx, ny]; transpose to match run_sim outputs [ny, nx]
        eps_2d = sim.get_epsilon()
        eps_mid = eps_2d.T

        # Free internal Meep state (helps avoid memory creep)
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

        dpml_x = 2 * self.dpml
        dpml_y = 2 * self.dpml
        return eps_mid[p:-p, p:-p], (cell_x_display - dpml_x, cell_y_display - dpml_y)

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        """
        Run Meep simulation with excitation from specified port.

        input_port=1: splitter mode (excite from stem, measure outputs at ports 2 & 3)
        input_port=2: combiner mode (excite from top arm, measure at ports 1 & 3)
        input_port=3: combiner mode (excite from bottom arm, measure at ports 1 & 2)

        Returns: eps_mid, Ez_mid, Hx_mid, Hy_mid, S_dict, (cell_x, cell_y)
        """
        if input_port not in (1, 2, 3):
            raise ValueError("YBranch2D input_port must be 1, 2, or 3.")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        # Select source volume and propagation direction based on input port
        if input_port == 1:
            src_vol = self.src_1
            # Propagating in +x direction (left to right)
            eig_kpoint = mp.Vector3(+1, 0, 0) if self.orientation == "horizontal" else mp.Vector3(0, +1, 0)
        elif input_port == 2:
            src_vol = self.src_2
            # Propagating in -x direction (right to left, into the combiner)
            eig_kpoint = mp.Vector3(-1, 0, 0) if self.orientation == "horizontal" else mp.Vector3(0, -1, 0)
        else:  # input_port == 3
            src_vol = self.src_3
            # Propagating in -x direction (right to left, into the combiner)
            eig_kpoint = mp.Vector3(-1, 0, 0) if self.orientation == "horizontal" else mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=eig_kpoint,
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

        dft_fields = sim.add_dft_fields(
            [mp.Ez, mp.Hx, mp.Hy],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        res1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.NO_PARITY)
        res2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.NO_PARITY)
        res3 = sim.get_eigenmode_coefficients(m3, [1], eig_parity=mp.NO_PARITY)

        # alpha[freq_idx, band_idx, direction_idx]
        # direction 0 = +x (or +y for vertical), direction 1 = -x (or -y)
        a1_fwd = res1.alpha[0, 0, 0]  # +x at port 1
        a1_bwd = res1.alpha[0, 0, 1]  # -x at port 1
        a2_fwd = res2.alpha[0, 0, 0]  # +x at port 2
        a2_bwd = res2.alpha[0, 0, 1]  # -x at port 2
        a3_fwd = res3.alpha[0, 0, 0]  # +x at port 3
        a3_bwd = res3.alpha[0, 0, 1]  # -x at port 3

        S = {}

        if input_port == 1:
            # Splitter mode: input from left (port 1), outputs to right (ports 2, 3)
            # "toward device" at port 1 is +x direction (fwd), so incoming = fwd
            # "toward device" at port 2,3 is -x direction (bwd), so outgoing = fwd
            a_in = a1_fwd
            S[(1, 1)] = a1_bwd / a_in  # reflection
            S[(2, 1)] = a2_fwd / a_in  # transmission to port 2
            S[(3, 1)] = a3_fwd / a_in  # transmission to port 3

        elif input_port == 2:
            # Combiner mode: input from top arm (port 2), output at stem (port 1)
            # "toward device" at port 2 is -x direction (bwd), so incoming = bwd
            # "toward device" at port 1 is +x direction (fwd), so outgoing = bwd (leaving to left)
            a_in = a2_bwd
            S[(1, 2)] = a1_bwd / a_in  # transmission to port 1 (combiner output)
            S[(2, 2)] = a2_fwd / a_in  # reflection at port 2
            S[(3, 2)] = a3_fwd / a_in  # crosstalk to port 3

        else:  # input_port == 3
            # Combiner mode: input from bottom arm (port 3), output at stem (port 1)
            # "toward device" at port 3 is -x direction (bwd), so incoming = bwd
            a_in = a3_bwd
            S[(1, 3)] = a1_bwd / a_in  # transmission to port 1 (combiner output)
            S[(2, 3)] = a2_fwd / a_in  # crosstalk to port 2
            S[(3, 3)] = a3_fwd / a_in  # reflection at port 3

        self.S = S

        mode_name = "Y-splitter" if input_port == 1 else "Y-combiner"
        print(f"=== 2D {mode_name}, excitation at port {input_port} ===")
        power_sum = 0.0
        for p in (1, 2, 3):
            s_val = S[(p, input_port)]
            p_val = abs(s_val) ** 2
            power_sum += p_val
            print(f"S{p}{input_port} = {s_val:.4g}, |S{p}{input_port}|^2 = {p_val:.4g}")
        print(f"Sum of guided powers Σ|S·{input_port}|^2 = {power_sum:.4g}")

        eps_2d = sim.get_epsilon()
        self.eps_mid = eps_2d.T

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0)
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0)
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0)

        self.Ez_mid = Ez_mid.T
        self.Hx_mid = Hx_mid.T
        self.Hy_mid = Hy_mid.T

        # For vertical orientation, transpose to display device vertically
        if self.orientation == "vertical":
            self.eps_mid = self.eps_mid.T
            self.Ez_mid = self.Ez_mid.T
            self.Hx_mid = self.Hx_mid.T
            self.Hy_mid = self.Hy_mid.T
            cell_x_display, cell_y_display = self.cell_y, self.cell_x
        else:
            cell_x_display, cell_y_display = self.cell_x, self.cell_y

        return self.eps_mid, self.Ez_mid, self.Hx_mid, self.Hy_mid, S, (cell_x_display, cell_y_display)
