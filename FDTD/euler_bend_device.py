# euler_bend_device.py
"""
EulerZigZag2D (Meep) with:
- Deterministic pixel-exact crops (no "different size boxes" artifacts) via dpml/cell quantization
- Edge-aligned I/O ports: input/output always axis-aligned and perpendicular to the boundary
- Turn-devices (end_with_turn=True) end with an axis-aligned direction so output port is on an edge

Key changes (vs your pasted version):
1) quantize_grid=True (default): dpml is snapped so pml_px is an integer, and cell size is chosen so
   (crop_px + 2*pml_px) is an integer number of pixels. This makes eps crops exactly crop_px×crop_px.
2) path_rotation_deg is snapped to multiples of 90° (0/90/180/270) so ports land on edges.
3) If end_with_turn=True and force_turn_ports_axis_aligned=True (default), the *final* turn is snapped
   to 90° (or nearest multiple of 90°) so the output lead is axis-aligned.
4) Simplified lead extension to boundaries with explicit ray intersection (removes sign confusion).
"""

import meep as mp
import numpy as np

from utils import neff_siwire_from_tables
from devices_base import Device2DBase


def _unit(vx: float, vy: float, eps: float = 1e-12):
    n = float(np.sqrt(vx * vx + vy * vy))
    if n < eps:
        return 1.0, 0.0
    return vx / n, vy / n


def _euler_bend_points(
    *,
    theta_total_rad: float,
    R_min_um: float,
    n_pts: int = 256,
    sign: float = 1.0,
):
    """
    Simple "Euler" bend approximation:
    curvature ramps 0 -> kmax -> 0 (triangular profile), so the path starts/ends straight.

    Peak curvature: kmax = 1/R_min.
    Total turning angle theta = ∫ k(s) ds = 0.5 * kmax * L  =>  L = 2 * theta / kmax = 2 * theta * R_min.

    Returns x, y coordinates with x[0]=y[0]=0, and path length L.
    """
    theta = float(theta_total_rad)
    if theta <= 0:
        raise ValueError("theta_total_rad must be > 0")
    R = float(R_min_um)
    if R <= 0:
        raise ValueError("R_min_um must be > 0")

    kmax = 1.0 / R
    L = 2.0 * theta / kmax  # = 2 * theta * R

    n = max(32, int(n_pts))
    s = np.linspace(0.0, L, n, dtype=np.float64)
    ds = float(s[1] - s[0])

    # triangular curvature profile
    k = np.empty_like(s)
    mid = 0.5 * L
    for i, si in enumerate(s):
        if si <= mid:
            k[i] = kmax * (si / mid)
        else:
            k[i] = kmax * ((L - si) / mid)
    k *= float(sign)

    th = np.cumsum(k) * ds
    x = np.cumsum(np.cos(th)) * ds
    y = np.cumsum(np.sin(th)) * ds

    x -= x[0]
    y -= y[0]
    return x, y, float(L)


def _polyline_to_blocks(points_xy: np.ndarray, wg_width_um: float, material: mp.Medium):
    """
    Convert a centerline polyline into a list of rotated mp.Blocks aligned to each segment.
    """
    blocks = []
    pts = np.asarray(points_xy, dtype=np.float64)
    for a, b in zip(pts[:-1], pts[1:]):
        x0, y0 = float(a[0]), float(a[1])
        x1, y1 = float(b[0]), float(b[1])
        dx = x1 - x0
        dy = y1 - y0
        seg_len = float(np.sqrt(dx * dx + dy * dy))
        if seg_len <= 1e-9:
            continue

        tx, ty = _unit(dx, dy)
        nx, ny = -ty, tx

        center = mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0)
        e1 = mp.Vector3(tx, ty, 0.0)  # tangent
        e2 = mp.Vector3(nx, ny, 0.0)  # normal

        size = mp.Vector3(seg_len * 1.05, float(wg_width_um), mp.inf)
        blocks.append(mp.Block(size=size, center=center, material=material, e1=e1, e2=e2))
    return blocks


def _rotate_points(pts: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 2D points around origin by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T


def _snap_deg_90(x: float) -> float:
    return float(int(np.round(float(x) / 90.0)) * 90.0) % 360.0


def _snap_deg_90_nonsigned(x: float) -> float:
    # For angles like bend magnitudes (0..180), snap to {0,90,180,...}
    return float(int(np.round(float(x) / 90.0)) * 90.0)


class EulerZigZag2D(Device2DBase):
    """
    2D effective-index waveguide with alternating Euler bends (zig-zag).

    This class supports arbitrary crop sizes by setting crop_px and leaving cell_x_um/cell_y_um None.
    When quantize_grid=True (default), dpml and cell size are quantized so the non-PML interior is
    exactly crop_px×crop_px pixels (deterministic shapes).
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wavelength_um: float = 1.55,
        resolution: int = 24,
        dpml: float = 2.0 / 3.0,
        crop_px: int = 512,
        n_core: float | None = None,
        n_clad: float = 1.444,
        # geometry controls
        y0_um: float = 0.0,
        n_zigs: int = 1,
        start_up: bool = True,
        end_with_turn: bool = False,
        end_turn_deg: float | None = None,
        R_min_um: float = 1.5,
        straight_x_um: float = 1.0,
        straight_y_um: float = 0.8,
        lead_x_um: float = 2.0,
        min_straight_input_um: float = 2.0,
        bend_angle_deg: float = 90.0,
        lead_extend_through_pml: bool = True,
        lead_boundary_clearance_um: float = 0.0,
        fit_margin_um: float = 0.2,
        path_rotation_deg: float = 0.0,
        # ports/sources
        port_margin_um: float = 0.5,
        port_span_pad_um: float = 0.25,
        source_shift_um: float = 0.5,
        # grid overrides
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        # NEW: stability + port alignment
        quantize_grid: bool = True,
        force_turn_ports_axis_aligned: bool = True,
        turn_axis_deg: float = 90.0,
    ):
        crop_px = int(crop_px)
        if crop_px <= 0:
            raise ValueError("crop_px must be > 0")
        self.crop_px = crop_px

        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        # Deterministic dpml/cell quantization => exact crop_px interior.
        if bool(quantize_grid):
            pml_px = int(np.round(float(dpml) * float(resolution)))
            dpml_q = float(pml_px) / float(resolution)
            full_px = int(crop_px + 2 * pml_px)
            cell_um_q = float(full_px) / float(resolution)
        else:
            dpml_q = float(dpml)
            cell_um_q = (float(crop_px) / float(resolution)) + 2.0 * float(dpml_q)

        cx = float(cell_um_q) if cell_x_um is None else float(cell_x_um)
        cy = float(cell_um_q) if cell_y_um is None else float(cell_y_um)

        super().__init__(cell_x_um=cx, cell_y_um=cy, dpml=float(dpml_q), resolution=int(resolution))

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)

        self.y0_um = float(y0_um)
        self.n_zigs = int(n_zigs)
        self.start_up = bool(start_up)
        self.end_with_turn = bool(end_with_turn)
        self.end_turn_deg = None if end_turn_deg is None else float(end_turn_deg)
        self.R_min_um = float(R_min_um)
        self.straight_x_um = float(straight_x_um)
        self.straight_y_um = float(straight_y_um)
        self.lead_x_um = float(lead_x_um)
        self.min_straight_input_um = float(min_straight_input_um)
        self.bend_angle_deg = float(bend_angle_deg)
        self.lead_extend_through_pml = bool(lead_extend_through_pml)
        self.lead_boundary_clearance_um = float(lead_boundary_clearance_um)
        self.fit_margin_um = float(fit_margin_um)

        # Enforce rotation to {0,90,180,270} so ports land on edges.
        self.path_rotation_deg = _snap_deg_90(path_rotation_deg)

        self.port_margin_um = float(port_margin_um)
        self.port_span_pad_um = float(port_span_pad_um)
        self.source_shift_um = float(source_shift_um)

        self.force_turn_ports_axis_aligned = bool(force_turn_ports_axis_aligned)
        self.turn_axis_deg = float(turn_axis_deg)

        if n_core is None:
            self.n_core = neff_siwire_from_tables(self.wg_width_um, self.wavelength_um)
        else:
            self.n_core = float(n_core)

        # to be filled by build_geometry()
        self.geometry = None
        self.clad_medium = None
        self.port_1 = None
        self.port_2 = None
        self.src_1 = None
        self.full_plane = None

        # Source region info (um, physical coords)
        self.src_center_um = None
        self.src_size_um = None
        self.src_direction_rad = None

        # scalar meta for masks
        self.x_port_1_um = None
        self.y_port_1_um = None
        self.x_port_2_um = None
        self.y_port_2_um = None
        self.x_src_um = None
        self.y_src_um = None

        self.build_geometry()

    def pml_px(self) -> int:
        return int(np.round(float(self.dpml) * float(self.resolution)))

    def nonpml_shape(self) -> tuple[int, int]:
        p = self.pml_px()
        nx = int(np.round(self.cell_x * self.resolution)) - 2 * p
        ny = int(np.round(self.cell_y * self.resolution)) - 2 * p
        return ny, nx

    def get_port_centers_um(self):
        return {
            1: (float(self.x_port_1_um), float(self.y_port_1_um)),
            2: (float(self.x_port_2_um), float(self.y_port_2_um)),
        }

    def get_port_y_span_um(self):
        return float(self.wg_width_um + 2.0 * self.port_span_pad_um)

    def _ray_to_cell_boundary(self, pt: np.ndarray, d: np.ndarray, clearance: float) -> np.ndarray | None:
        """
        Ray from pt along direction d (must be nonzero) to cell boundary with clearance.
        Returns the nearest positive intersection point, or None if no valid hit.
        """
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        px, py = float(pt[0]), float(pt[1])
        dx, dy = float(d[0]), float(d[1])

        eps = 1e-12
        if abs(dx) < eps and abs(dy) < eps:
            return None

        t_vals = []
        # Vertical boundaries x = ±half_x
        if abs(dx) >= eps:
            t_left = ((-half_x + clearance) - px) / dx
            t_right = ((+half_x - clearance) - px) / dx
            if t_left > 0:
                t_vals.append(t_left)
            if t_right > 0:
                t_vals.append(t_right)

        # Horizontal boundaries y = ±half_y
        if abs(dy) >= eps:
            t_bot = ((-half_y + clearance) - py) / dy
            t_top = ((+half_y - clearance) - py) / dy
            if t_bot > 0:
                t_vals.append(t_bot)
            if t_top > 0:
                t_vals.append(t_top)

        if not t_vals:
            return None
        t = float(min(t_vals))
        return np.array([px + dx * t, py + dy * t], dtype=np.float64)

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = +half_y - self.dpml

        if self.lead_boundary_clearance_um < 0:
            raise ValueError("lead_boundary_clearance_um must be >= 0")
        if self.fit_margin_um < 0:
            raise ValueError("fit_margin_um must be >= 0")

        # Build the zigzag centerline in local coordinates (starts along +x), then rotate it.
        pts = []
        x = 0.0
        y = 0.0
        pts.append((x, y))

        def add_straight(dx_um: float, dy_um: float, n: int):
            nonlocal x, y
            n = max(2, int(n))
            for t in np.linspace(0.0, 1.0, n, dtype=np.float64)[1:]:
                pts.append((x + dx_um * float(t), y + dy_um * float(t)))
            x += float(dx_um)
            y += float(dy_um)

        # Ensure minimum straight input section
        input_lead = max(self.lead_x_um, self.min_straight_input_um)
        add_straight(input_lead, 0.0, n=max(8, int(input_lead * self.resolution)))

        up = bool(self.start_up)
        dir_x, dir_y = 1.0, 0.0  # local forward direction (starts +x)

        n_zigs = max(0, int(self.n_zigs))
        for zi in range(n_zigs):
            sgn = +1.0 if up else -1.0

            theta_deg = float(self.bend_angle_deg)

            # If this is a turn-device, enforce an axis-aligned *final* direction by making the final turn 90°.
            if bool(self.end_with_turn) and (zi == n_zigs - 1):
                if self.end_turn_deg is not None:
                    if self.force_turn_ports_axis_aligned:
                        theta_deg = _snap_deg_90_nonsigned(self.end_turn_deg)
                    else:
                        theta_deg = float(self.end_turn_deg)
                elif self.force_turn_ports_axis_aligned:
                    theta_deg = _snap_deg_90_nonsigned(self.turn_axis_deg)

            bend_rad = np.deg2rad(theta_deg)

            # First bend
            xb, yb, _ = _euler_bend_points(theta_total_rad=bend_rad, R_min_um=self.R_min_um, n_pts=128, sign=sgn)
            for i in range(1, len(xb)):
                pts.append((x + float(xb[i]), y + float(yb[i])))
            x += float(xb[-1])
            y += float(yb[-1])

            # Straight along bent direction
            bent_dir_x = float(np.cos(sgn * bend_rad))
            bent_dir_y = float(np.sin(sgn * bend_rad))
            add_straight(
                bent_dir_x * self.straight_y_um,
                bent_dir_y * self.straight_y_um,
                n=max(8, int(abs(self.straight_y_um) * self.resolution)),
            )

            # If we end after the first bend, forward dir is bent_dir.
            if bool(self.end_with_turn) and (zi == n_zigs - 1):
                dir_x, dir_y = bent_dir_x, bent_dir_y
                break

            # Second bend back to +x
            xb2, yb2, _ = _euler_bend_points(theta_total_rad=bend_rad, R_min_um=self.R_min_um, n_pts=128, sign=-sgn)
            rot = sgn * bend_rad
            cr, sr = float(np.cos(rot)), float(np.sin(rot))
            for i in range(1, len(xb2)):
                xx = float(xb2[i])
                yy = float(yb2[i])
                xr = cr * xx - sr * yy
                yr = sr * xx + cr * yy
                pts.append((x + xr, y + yr))
            xx = float(xb2[-1])
            yy = float(yb2[-1])
            xr_end = cr * xx - sr * yy
            yr_end = sr * xx + cr * yy
            x += xr_end
            y += yr_end

            dir_x, dir_y = 1.0, 0.0

            # Horizontal straight between zigs
            add_straight(self.straight_x_um, 0.0, n=max(8, int(self.straight_x_um * self.resolution)))
            up = not up

        # Output lead in current forward direction
        add_straight(dir_x * self.lead_x_um, dir_y * self.lead_x_um, n=max(8, int(self.lead_x_um * self.resolution)))

        pts = np.asarray(pts, dtype=np.float64)

        # Center and apply vertical offset (before rotation)
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        pts[:, 0] -= 0.5 * (x_min + x_max)
        pts[:, 1] -= 0.5 * (y_min + y_max)
        pts[:, 1] += self.y0_um

        # Rotate path (snapped to multiples of 90 earlier)
        rot_rad = np.deg2rad(self.path_rotation_deg)
        pts = _rotate_points(pts, rot_rad)

        # Fit check in non-PML region
        core_half_w = 0.5 * self.wg_width_um
        x_min_core = float(pts[:, 0].min()) - core_half_w
        x_max_core = float(pts[:, 0].max()) + core_half_w
        y_min_core = float(pts[:, 1].min()) - core_half_w
        y_max_core = float(pts[:, 1].max()) + core_half_w

        if (
            x_min_core < (nonpml_left + self.fit_margin_um)
            or x_max_core > (nonpml_right - self.fit_margin_um)
            or y_min_core < (nonpml_bot + self.fit_margin_um)
            or y_max_core > (nonpml_top - self.fit_margin_um)
        ):
            raise ValueError(
                "Waveguide does not fit inside the non-PML region. "
                f"core_x=[{x_min_core:.3f},{x_max_core:.3f}] um, "
                f"core_y=[{y_min_core:.3f},{y_max_core:.3f}] um, "
                f"allowed_x=[{(nonpml_left + self.fit_margin_um):.3f},{(nonpml_right - self.fit_margin_um):.3f}] um, "
                f"allowed_y=[{(nonpml_bot + self.fit_margin_um):.3f},{(nonpml_top - self.fit_margin_um):.3f}] um."
            )

        # Centerline -> blocks
        geometry = _polyline_to_blocks(pts, wg_width_um=self.wg_width_um, material=core)

        # Endpoint tangents
        input_pt = pts[0]
        input_tangent = pts[1] - pts[0]
        input_tangent = input_tangent / np.linalg.norm(input_tangent)

        output_pt = pts[-1]
        output_tangent = pts[-1] - pts[-2]
        output_tangent = output_tangent / np.linalg.norm(output_tangent)

        # Extend leads to cell boundary (into PML) so fields settle / ports are stable.
        clearance = self.lead_boundary_clearance_um if self.lead_extend_through_pml else self.dpml

        # input extends backwards
        input_end = self._ray_to_cell_boundary(input_pt, -input_tangent, clearance)
        if input_end is not None:
            lead_len = float(np.linalg.norm(input_end - input_pt))
            if lead_len > 0.01:
                lead_center = 0.5 * (input_pt + input_end)
                tx, ty = _unit(float(-input_tangent[0]), float(-input_tangent[1]))
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len * 1.05, self.wg_width_um, mp.inf),
                        center=mp.Vector3(float(lead_center[0]), float(lead_center[1]), 0.0),
                        material=core,
                        e1=mp.Vector3(tx, ty, 0.0),
                        e2=mp.Vector3(-ty, tx, 0.0),
                    )
                )

        # output extends forwards
        output_end = self._ray_to_cell_boundary(output_pt, output_tangent, clearance)
        if output_end is not None:
            lead_len = float(np.linalg.norm(output_end - output_pt))
            if lead_len > 0.01:
                lead_center = 0.5 * (output_pt + output_end)
                tx, ty = _unit(float(output_tangent[0]), float(output_tangent[1]))
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len * 1.05, self.wg_width_um, mp.inf),
                        center=mp.Vector3(float(lead_center[0]), float(lead_center[1]), 0.0),
                        material=core,
                        e1=mp.Vector3(tx, ty, 0.0),
                        e2=mp.Vector3(-ty, tx, 0.0),
                    )
                )

        self.geometry = geometry

        # Ports/source: placed relative to non-PML boundary intersection along the axis-aligned tangents.
        span = self.get_port_y_span_um()
        half_span = 0.5 * span

        def find_nonpml_boundary_intersection(pt, d_toward_boundary):
            tx, ty = float(d_toward_boundary[0]), float(d_toward_boundary[1])
            px, py = float(pt[0]), float(pt[1])
            eps = 1e-12
            t_vals = []

            if abs(tx) > eps:
                t_left = (nonpml_left - px) / tx
                t_right = (nonpml_right - px) / tx
                if t_left > 0:
                    t_vals.append(t_left)
                if t_right > 0:
                    t_vals.append(t_right)

            if abs(ty) > eps:
                t_bot = (nonpml_bot - py) / ty
                t_top = (nonpml_top - py) / ty
                if t_bot > 0:
                    t_vals.append(t_bot)
                if t_top > 0:
                    t_vals.append(t_top)

            if not t_vals:
                return np.array([px, py], dtype=np.float64)
            t = float(min(t_vals))
            return np.array([px + tx * t, py + ty * t], dtype=np.float64)

        # --- Input boundary hit (toward boundary is -input_tangent) ---
        input_boundary_pt = find_nonpml_boundary_intersection(input_pt, -input_tangent)

        # Source upstream, monitor downstream
        src_pt = input_boundary_pt + input_tangent * self.port_margin_um
        port_1_pt = input_boundary_pt + input_tangent * (self.port_margin_um + self.source_shift_um)

        # Perpendicular for visualization
        src_perp = np.array([-input_tangent[1], input_tangent[0]], dtype=np.float64)

        self.x_port_1_um = float(port_1_pt[0])
        self.y_port_1_um = float(port_1_pt[1])
        self.x_src_um = float(src_pt[0])
        self.y_src_um = float(src_pt[1])

        self.src_center_um = (float(src_pt[0]), float(src_pt[1]))
        self.src_size_um = (0.0, span)
        self.src_direction_rad = float(np.arctan2(float(input_tangent[1]), float(input_tangent[0])))

        # Port 1 volume (axis-aligned by construction)
        if abs(float(input_tangent[0])) >= abs(float(input_tangent[1])):
            port_1_size = mp.Vector3(0, span, 0)
            src_size_vec = mp.Vector3(0, span, 0)
        else:
            port_1_size = mp.Vector3(span, 0, 0)
            src_size_vec = mp.Vector3(span, 0, 0)

        self.port_1 = mp.Volume(center=mp.Vector3(float(port_1_pt[0]), float(port_1_pt[1]), 0.0), size=port_1_size)
        self.src_1 = mp.Volume(center=mp.Vector3(float(src_pt[0]), float(src_pt[1]), 0.0), size=src_size_vec)

        self.src_line_start_um = (float(src_pt[0] - src_perp[0] * half_span), float(src_pt[1] - src_perp[1] * half_span))
        self.src_line_end_um = (float(src_pt[0] + src_perp[0] * half_span), float(src_pt[1] + src_perp[1] * half_span))

        # --- Output boundary hit (toward boundary is +output_tangent) ---
        output_boundary_pt = find_nonpml_boundary_intersection(output_pt, output_tangent)

        # Place output monitor inward from boundary
        port_2_pt = output_boundary_pt - output_tangent * self.port_margin_um

        self.x_port_2_um = float(port_2_pt[0])
        self.y_port_2_um = float(port_2_pt[1])

        output_perp = np.array([-output_tangent[1], output_tangent[0]], dtype=np.float64)

        self.output_center_um = (float(port_2_pt[0]), float(port_2_pt[1]))
        self.output_line_start_um = (float(port_2_pt[0] - output_perp[0] * half_span), float(port_2_pt[1] - output_perp[1] * half_span))
        self.output_line_end_um = (float(port_2_pt[0] + output_perp[0] * half_span), float(port_2_pt[1] + output_perp[1] * half_span))

        if abs(float(output_tangent[0])) >= abs(float(output_tangent[1])):
            port_2_size = mp.Vector3(0, span, 0)
        else:
            port_2_size = mp.Vector3(span, 0, 0)

        self.port_2 = mp.Volume(center=mp.Vector3(float(port_2_pt[0]), float(port_2_pt[1]), 0.0), size=port_2_size)

        self.full_plane = mp.Volume(center=mp.Vector3(0.0, 0.0, 0.0), size=mp.Vector3(self.cell_x, self.cell_y, 0.0))

        # Store tangents for eigenmode source/monitor k-points
        self.input_tangent = input_tangent
        self.output_tangent = output_tangent

        # Hard guarantee: if we forced axis-aligned turn ports, both tangents must be axis-aligned
        if self.force_turn_ports_axis_aligned:
            def _is_axis_aligned(t):
                return (abs(t[0]) > 0.999 and abs(t[1]) < 1e-3) or (abs(t[1]) > 0.999 and abs(t[0]) < 1e-3)
            if not (_is_axis_aligned(self.input_tangent) and _is_axis_aligned(self.output_tangent)):
                raise ValueError("Ports are not axis-aligned; check turn enforcement / rotation snapping.")

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

    def _um_to_px(self, x_um, y_um, crop_pml: bool):
        p = self.pml_px() if crop_pml else 0
        px_x = (x_um + 0.5 * self.cell_x) * self.resolution - p
        px_y = (y_um + 0.5 * self.cell_y) * self.resolution - p
        return px_x, px_y

    def get_source_region_px(self, crop_pml: bool = True):
        cx, cy = self._um_to_px(self.src_center_um[0], self.src_center_um[1], crop_pml)
        sx, sy = self._um_to_px(self.src_line_start_um[0], self.src_line_start_um[1], crop_pml)
        ex, ey = self._um_to_px(self.src_line_end_um[0], self.src_line_end_um[1], crop_pml)
        dx = float(self.input_tangent[0])
        dy = float(self.input_tangent[1])
        return {
            "center_px": (cx, cy),
            "line_start_px": (sx, sy),
            "line_end_px": (ex, ey),
            "direction_px": (dx, dy),
        }

    def get_output_region_px(self, crop_pml: bool = True):
        cx, cy = self._um_to_px(self.output_center_um[0], self.output_center_um[1], crop_pml)
        sx, sy = self._um_to_px(self.output_line_start_um[0], self.output_line_start_um[1], crop_pml)
        ex, ey = self._um_to_px(self.output_line_end_um[0], self.output_line_end_um[1], crop_pml)
        dx = float(self.output_tangent[0])
        dy = float(self.output_tangent[1])
        return {
            "center_px": (cx, cy),
            "line_start_px": (sx, sy),
            "line_end_px": (ex, ey),
            "direction_px": (dx, dy),
        }

    def run_sim(self, decay_tol: float = 1e-6):
        """
        Run FDTD simulation, compute S-parameters (S11, S21), and return fields.

        Returns:
            eps_mid: [ny, nx]
            Ez_mid:  [ny, nx] complex Ez at center frequency
            S: {(1,1): S11, (2,1): S21}
            cell: (cell_x, cell_y) in µm
        """
        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        kdir_in = mp.Vector3(float(self.input_tangent[0]), float(self.input_tangent[1]), 0.0)
        kdir_out = mp.Vector3(float(self.output_tangent[0]), float(self.output_tangent[1]), 0.0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=self.src_1,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=kdir_in,
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

        # Axis choice (should be axis-aligned now)
        dir_in = mp.X if abs(float(self.input_tangent[0])) >= abs(float(self.input_tangent[1])) else mp.Y
        dir_out = mp.X if abs(float(self.output_tangent[0])) >= abs(float(self.output_tangent[1])) else mp.Y

        mon_1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=dir_in))
        mon_2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=dir_out))

        dft_fields = sim.add_dft_fields(
            [mp.Ez],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        res_1 = sim.get_eigenmode_coefficients(mon_1, [1], eig_parity=mp.NO_PARITY)
        res_2 = sim.get_eigenmode_coefficients(mon_2, [1], eig_parity=mp.NO_PARITY)

        a1_fwd = res_1.alpha[0, 0, 0]
        a1_bwd = res_1.alpha[0, 0, 1]

        a2_fwd = res_2.alpha[0, 0, 0]
        a2_bwd = res_2.alpha[0, 0, 1]

        def _incident_and_reflected(a_fwd, a_bwd, direction_axis, tangent):
            if direction_axis == mp.X:
                sgn = float(np.sign(float(tangent[0])))
            else:
                sgn = float(np.sign(float(tangent[1])))
            if sgn == 0.0:
                sgn = 1.0
            # If propagation is along +axis, incident is forward; if along -axis, incident is backward.
            if sgn > 0:
                return a_fwd, a_bwd
            return a_bwd, a_fwd

        a_in, a_ref = _incident_and_reflected(a1_fwd, a1_bwd, dir_in, self.input_tangent)
        a_out, _ = _incident_and_reflected(a2_fwd, a2_bwd, dir_out, self.output_tangent)

        if abs(a_in) < 1e-12:
            sim.reset_meep()
            raise ValueError("Input mode amplitude is ~0; check port direction/sign convention.")

        S = {
            (1, 1): a_ref / a_in,
            (2, 1): a_out / a_in,
        }

        eps_2d = sim.get_epsilon()
        eps_mid = eps_2d.T
        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0).T
        sim.reset_meep()

        return eps_mid, Ez_mid, S, (self.cell_x, self.cell_y)
