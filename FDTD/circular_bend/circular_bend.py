# circular_bend.py
"""
Constant-radius 90-degree circular bend waveguide (2D effective-index)Compatible with Device2DBase, supports:
- Single-frequency run_sim(): eps + Ez + S11/S21
- Broadband run_spectrum(): S11(λ), S21(λ)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meep as mp
import numpy as np

from devices_base import Device2DBase
from utils import (
    neff_siwire_from_tables,
    get_mode_alpha_2dir,
    pick_in_out_from_alpha,
)


# -------------------------
# Geometry helpers
# -------------------------
def _unit(v, eps=1e-12):
    """Normalize a 2D vector."""
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _rotate_2d(pts, angle_rad):
    """Rotate 2D points by angle_rad."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T


def circular_arc_points(R_um, theta_rad, n_pts=128, center=(0, 0)):
    """
    Generate points along a circular arc.

    Args:
        R_um: radius in microns
        theta_rad: total angle to sweep (e.g., pi/2 for 90 degrees)
        n_pts: number of points along arc
        center: (cx, cy) center of circular arc

    Returns:
        np.ndarray of shape (n_pts, 2) with (x, y) coordinates
    """
    theta = float(theta_rad)
    R = float(R_um)
    n = max(16, int(n_pts))

    angles = np.linspace(0.0, theta, n, dtype=np.float64)
    x = R * np.cos(angles) + float(center[0])
    y = R * np.sin(angles) + float(center[1])

    return np.column_stack([x, y])


def polyline_to_blocks(points_xy, wg_width_um, material):
    """
    Convert a centerline polyline into a list of rotated mp.Blocks.

    Args:
        points_xy: np.ndarray of shape (N, 2) with (x, y) coordinates
        wg_width_um: waveguide width
        material: mp.Medium for the blocks

    Returns:
        list of mp.Block objects
    """
    blocks = []
    pts = np.asarray(points_xy, dtype=np.float64)

    for a, b in zip(pts[:-1], pts[1:]):
        a = np.asarray(a, dtype=np.float64).reshape(2)
        b = np.asarray(b, dtype=np.float64).reshape(2)
        d = b - a
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len <= 1e-9:
            continue

        t = _unit(d)  # tangent
        n = np.array([-t[1], t[0]], dtype=np.float64)  # normal

        center = mp.Vector3(float(0.5 * (a[0] + b[0])), float(0.5 * (a[1] + b[1])), 0.0)
        e1 = mp.Vector3(float(t[0]), float(t[1]), 0.0)
        e2 = mp.Vector3(float(n[0]), float(n[1]), 0.0)

        size = mp.Vector3(seg_len * 1.05, float(wg_width_um), mp.inf)
        blocks.append(mp.Block(size=size, center=center, material=material, e1=e1, e2=e2))

    return blocks


# -------------------------
# Device class
# -------------------------
class CircularBend2D(Device2DBase):
    """
    2D constant-radius circular bend waveguide with two mode ports.

    Geometry:
      - Input lead (straight) along +x
      - Circular bend section (constant radius)
      - Output lead (straight) rotated by bend_angle_deg

    Supports:
      - run_sim(): single-frequency eps + Ez snapshot + S11/S21
      - run_spectrum(): broadband wavelength grid + S11(λ), S21(λ)
    """

    def __init__(
        self,
        wg_width=0.45,
        wavelength_um=1.55,
        bend_radius_um=5.0,
        bend_angle_deg=90.0,
        lead_in_um=3.0,
        lead_out_um=3.0,
        n_core=None,
        n_clad=1.444,
        port_y_pad=1.0,
        source_shift=0.5,
        fit_margin_um=0.5,
        # sim window overrides
        cell_x=None,
        cell_y=None,
        dpml=None,
        resolution=None,
    ):
        super().__init__(cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)

        self.wg_width = float(wg_width)
        self.wavelength_um = float(wavelength_um)
        self.bend_radius_um = float(bend_radius_um)
        self.bend_angle_deg = float(bend_angle_deg)
        self.bend_angle_rad = np.deg2rad(self.bend_angle_deg)
        self.lead_in_um = float(lead_in_um)
        self.lead_out_um = float(lead_out_um)

        self.n_clad = float(n_clad)
        self.port_y_pad = float(port_y_pad)
        self.source_shift = float(source_shift)
        self.fit_margin_um = float(fit_margin_um)

        # Effective-index core
        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width, self.wavelength_um))
            if n_core is None
            else float(n_core)
        )
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        # Full plane for field visualization
        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

        # Will be set in build_geometry
        self.geometry = None
        self.port_in = None
        self.port_out = None
        self.src_vol = None

        self.build_geometry()

    def _clamp_point_inside_nonpml(self, point_xy, pad=0.1):
        """Clamp a point to remain safely inside the non-PML region."""
        x_min, x_max, y_min, y_max = self._nonpml_bounds
        px, py = float(point_xy[0]), float(point_xy[1])
        return np.array([
            min(max(px, x_min + float(pad)), x_max - float(pad)),
            min(max(py, y_min + float(pad)), y_max - float(pad)),
        ], dtype=np.float64)

    def build_geometry(self):
        """Build the circular bend geometry with straight input/output leads."""
        core = self.core_medium

        def ray_to_bounds(point, direction, x_min, x_max, y_min, y_max, eps=1e-12):
            """Return positive distance to the first axis-aligned boundary along direction."""
            dx, dy = float(direction[0]), float(direction[1])
            t_candidates = []
            if dx > eps:
                t_candidates.append((x_max - point[0]) / dx)
            elif dx < -eps:
                t_candidates.append((x_min - point[0]) / dx)
            if dy > eps:
                t_candidates.append((y_max - point[1]) / dy)
            elif dy < -eps:
                t_candidates.append((y_min - point[1]) / dy)
            t_candidates = [t for t in t_candidates if t > 0]
            if not t_candidates:
                return 0.0
            return float(min(t_candidates))

        # Build circular arc centerline
        arc_center = (0.0, self.bend_radius_um)
        arc_start_angle = -np.pi / 2  # Start at bottom of circle
        arc_end_angle = arc_start_angle + self.bend_angle_rad

        n_arc = max(32, int(np.ceil(self.bend_radius_um * abs(self.bend_angle_rad) * self.resolution)))
        angles = np.linspace(arc_start_angle, arc_end_angle, n_arc)

        bend_pts = []
        for angle in angles:
            x = arc_center[0] + self.bend_radius_um * np.cos(angle)
            y = arc_center[1] + self.bend_radius_um * np.sin(angle)
            bend_pts.append((x, y))
        bend_pts = np.asarray(bend_pts, dtype=np.float64)

        # Center the bend
        bend_center_x = 0.5 * (bend_pts[:, 0].min() + bend_pts[:, 0].max())
        bend_center_y = 0.5 * (bend_pts[:, 1].min() + bend_pts[:, 1].max())
        bend_pts[:, 0] -= bend_center_x
        bend_pts[:, 1] -= bend_center_y

        # Store for mask generation
        self._polyline_pts = bend_pts.copy()

        # Tangent at start (input) and end (output)
        in_tangent = _unit(bend_pts[1] - bend_pts[0])
        out_tangent = _unit(bend_pts[-1] - bend_pts[-2])

        # Cell boundaries
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = +half_y - self.dpml
        self._nonpml_bounds = (nonpml_left, nonpml_right, nonpml_bot, nonpml_top)

        half_w = 0.5 * float(self.wg_width)
        x_extent = max(abs(float(bend_pts[:, 0].min())), abs(float(bend_pts[:, 0].max()))) + half_w + self.fit_margin_um
        y_extent = max(abs(float(bend_pts[:, 1].min())), abs(float(bend_pts[:, 1].max()))) + half_w + self.fit_margin_um
        if x_extent > min(abs(nonpml_left), abs(nonpml_right)):
            raise ValueError(
                f"CircularBend2D exceeds non-PML x extent ({x_extent:.2f} um > "
                f"{min(abs(nonpml_left), abs(nonpml_right)):.2f} um). Reduce bend_radius_um/angle or increase crop."
            )
        if y_extent > min(abs(nonpml_bot), abs(nonpml_top)):
            raise ValueError(
                f"CircularBend2D exceeds non-PML y extent ({y_extent:.2f} um > "
                f"{min(abs(nonpml_bot), abs(nonpml_top)):.2f} um). Reduce bend_radius_um/angle or increase crop."
            )

        # Build geometry as polyline blocks for the bend
        bend_blocks = polyline_to_blocks(bend_pts, wg_width_um=self.wg_width, material=core)

        # Add straight input/output leads along tangents until beyond PML boundaries.
        bend_start = bend_pts[0]
        bend_end = bend_pts[-1]
        dir_in = -_unit(in_tangent)
        dir_out = _unit(out_tangent)

        x_min = -half_x - self.dpml
        x_max = +half_x + self.dpml
        y_min = -half_y - self.dpml
        y_max = +half_y + self.dpml

        lead_in_length = ray_to_bounds(bend_start, dir_in, x_min, x_max, y_min, y_max) + 1.0
        lead_out_length = ray_to_bounds(bend_end, dir_out, x_min, x_max, y_min, y_max) + 1.0

        lead_in_end = bend_start + dir_in * lead_in_length
        lead_out_end = bend_end + dir_out * lead_out_length
        lead_in_blocks = polyline_to_blocks(
            np.vstack([lead_in_end, bend_start]),
            wg_width_um=self.wg_width,
            material=core,
        )
        lead_out_blocks = polyline_to_blocks(
            np.vstack([bend_end, lead_out_end]),
            wg_width_um=self.wg_width,
            material=core,
        )

        self._lead_in_end = lead_in_end
        self._lead_out_end = lead_out_end
        self.geometry = lead_in_blocks + bend_blocks + lead_out_blocks

        # Device window for conditioning: bounding box of bend only
        x0, x1 = float(bend_pts[:, 0].min()), float(bend_pts[:, 0].max())
        y0, y1 = float(bend_pts[:, 1].min()), float(bend_pts[:, 1].max())
        self.dev_cx = 0.5 * (x0 + x1)
        self.dev_cy = 0.5 * (y0 + y1)
        self.dev_wx = (x1 - x0) + 2.0 * (self.fit_margin_um + half_w)
        self.dev_wy = (y1 - y0) + 2.0 * (self.fit_margin_um + half_w)

        # Ports: placed at non-PML boundary (inside by small margin)
        port_span = self.wg_width + self.port_y_pad
        port_margin = self.dpml + 0.5  # 0.5 µm inside the non-PML region

        nx_min = -half_x + port_margin
        nx_max = +half_x - port_margin
        ny_min = -half_y + port_margin
        ny_max = +half_y - port_margin

        def port_size_from_dir(direction):
            if abs(direction[0]) >= abs(direction[1]):
                return mp.Vector3(0, port_span, 0)
            return mp.Vector3(port_span, 0, 0)

        t_in = ray_to_bounds(bend_start, dir_in, nx_min, nx_max, ny_min, ny_max)
        if t_in <= 0:
            raise ValueError("CircularBend2D has no room to place the input port inside the non-PML region.")
        port_in_center = bend_start + dir_in * t_in
        self.port_in = mp.Volume(
            center=mp.Vector3(port_in_center[0], port_in_center[1], 0),
            size=port_size_from_dir(dir_in),
        )

        src_center = self._clamp_point_inside_nonpml(port_in_center + dir_in * self.source_shift)
        self.src_vol = mp.Volume(center=mp.Vector3(src_center[0], src_center[1], 0), size=self.port_in.size)

        t_out = ray_to_bounds(bend_end, dir_out, nx_min, nx_max, ny_min, ny_max)
        if t_out <= 0:
            raise ValueError("CircularBend2D has no room to place the output port inside the non-PML region.")
        port_out_center = bend_end + dir_out * t_out
        self.port_out = mp.Volume(
            center=mp.Vector3(port_out_center[0], port_out_center[1], 0),
            size=port_size_from_dir(dir_out),
        )

        # Store tangents for k-vector directions
        self._in_tangent = in_tangent
        self._out_tangent = out_tangent

    def core_mask(self, nx, ny, dx, dy):
        """Generate a mask for the waveguide core including straight leads."""
        if not hasattr(self, '_polyline_pts'):
            return None

        bend_pts = self._polyline_pts
        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        # Initialize mask as zeros
        mask = np.zeros((ny, nx), dtype=np.uint8)
        half_width = 0.5 * self.wg_width

        # Build a single polyline for lead-in + bend + lead-out.
        if hasattr(self, "_lead_in_end") and hasattr(self, "_lead_out_end"):
            pts = np.vstack([self._lead_in_end, bend_pts, self._lead_out_end])
        else:
            pts = bend_pts

        # Rasterize polyline.
        min_dist = np.full((ny, nx), np.inf, dtype=np.float64)
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]

            dx_seg = p1[0] - p0[0]
            dy_seg = p1[1] - p0[1]
            seg_len_sq = dx_seg**2 + dy_seg**2

            if seg_len_sq < 1e-12:
                continue

            t = ((xx - p0[0]) * dx_seg + (yy - p0[1]) * dy_seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            closest_x = p0[0] + t * dx_seg
            closest_y = p0[1] + t * dy_seg

            dist = np.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)
            min_dist = np.minimum(min_dist, dist)

        return np.logical_or(mask, (min_dist <= half_width)).astype(np.uint8)

    # -------------------------
    # Build simulations
    # -------------------------
    def _build_sim_single(self, input_port=1, df_frac=0.1):
        """Build single-frequency simulation."""
        lam = float(self.wavelength_um)
        fcen = 1.0 / lam
        fwidth = float(df_frac) * fcen

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(float(self._in_tangent[0]), float(self._in_tangent[1]), 0)
        else:
            # Input from output port (reversed)
            src_xy = self._clamp_point_inside_nonpml((
                self.port_out.center.x + self.source_shift * self._out_tangent[0],
                self.port_out.center.y + self.source_shift * self._out_tangent[1],
            ))
            src_x = src_xy[0]
            src_y = src_xy[1]
            src_vol = mp.Volume(center=mp.Vector3(src_x, src_y, 0), size=self.port_out.size)
            k_in = mp.Vector3(float(self._out_tangent[0]), float(self._out_tangent[1]), 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_in,
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

        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_in))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_out))

        dft = sim.add_dft_fields(
            [mp.Ez],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        return sim, (m1, m2), dft, fcen

    def _build_sim_broadband(self, input_port=1, lam_min_um=1.40, lam_max_um=1.60, Nf=101):
        """Build broadband simulation."""
        lam_min_um = float(lam_min_um)
        lam_max_um = float(lam_max_um)
        Nf = int(Nf)

        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(float(self._in_tangent[0]), float(self._in_tangent[1]), 0)
        else:
            src_xy = self._clamp_point_inside_nonpml((
                self.port_out.center.x + self.source_shift * self._out_tangent[0],
                self.port_out.center.y + self.source_shift * self._out_tangent[1],
            ))
            src_x = src_xy[0]
            src_y = src_xy[1]
            src_vol = mp.Volume(center=mp.Vector3(src_x, src_y, 0), size=self.port_out.size)
            k_in = mp.Vector3(float(self._out_tangent[0]), float(self._out_tangent[1]), 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_in,
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

        m1 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_in))
        m2 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_out))

        return sim, (m1, m2), (fcen, df, Nf, freqs, lams)

    # -------------------------
    # Single-frequency S-params
    # -------------------------
    def run_sim(self, decay_tol=1e-5, dir_plus=0, dir_minus=1):
        """
        Single-frequency run at self.wavelength_um.

        Returns:
          eps_mid [ny,nx], Ez_mid [ny,nx] complex (at fcen),
          S11 complex, S21 complex, (cell_x, cell_y)
        """
        toward = {1: +1, 2: -1}

        sim, (m1, m2), dft, _fcen = self._build_sim_single(input_port=1, df_frac=0.1)
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_mid = sim.get_epsilon().T.astype(np.float32)
        Ez_mid = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)

        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        _a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)

        S11 = b1_out / a1_in
        S21 = b2_out / a1_in

        sim.reset_meep()

        self.S11 = S11
        self.S21 = S21
        return eps_mid, Ez_mid, S11, S21, (self.cell_x, self.cell_y)

    # -------------------------
    # Broadband spectral response
    # -------------------------
    def run_spectrum(
        self,
        lam_min_um=1.40,
        lam_max_um=1.60,
        Nf=101,
        decay_tol=1e-6,
        n_periods=50,
        dir_plus=0,
        dir_minus=1,
    ):
        """
        Broadband run with a Gaussian pulse; computes S11(λ), S21(λ).

        Returns:
          lams [Nf] (µm),
          S11 [Nf] complex,
          S21 [Nf] complex
        """
        toward = {1: +1, 2: -1}

        sim, (m1, m2), (_fcen, _df, _Nf, _freqs, lams) = self._build_sim_broadband(
            input_port=1, lam_min_um=lam_min_um, lam_max_um=lam_max_um, Nf=Nf
        )

        stop = mp.stop_when_fields_decayed(
            int(n_periods),
            mp.Ez,
            self.port_out.center,
            float(decay_tol),
        )
        sim.run(until_after_sources=stop)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)

        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
        _a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)

        S11 = b1_out / a1_in
        S21 = b2_out / a1_in

        sim.reset_meep()

        self.lams = lams
        self.S11_spec = S11
        self.S21_spec = S21
        return lams, S11, S21
