# sbend.py
"""
S-bend waveguides (2D effective-index) with two geometric variants:

1. EulerSBend2D: Adiabatic S-bend with gradually varying curvature (production)
2. CosineSBend2D: Layout-standard S-bend using raised cosine profile

All devices:
- Inherit from Device2DBase (default 26x26 µm cell with 1 µm PML)
- Support run_sim() for single-frequency S-parameters
- Support run_spectrum() for broadband spectral response
- Include straight input/output leads extending into PML
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


def euler_sbend_points(lateral_offset_um, R_min_um, n_pts=256):
    """
    Generate points along an Euler S-bend (adiabatic).

    Two back-to-back Euler spirals with opposite curvature.
    Curvature profile: 0 -> +kmax -> 0 -> -kmax -> 0
    """
    offset = float(lateral_offset_um)
    R = float(R_min_um)
    if R <= 0:
        raise ValueError("R_min_um must be > 0")

    # Approximate angle needed for desired offset
    # For S-bend: offset ≈ 2*R*theta^2 for small theta
    theta = np.sqrt(abs(offset) / (2.0 * R))
    sign = np.sign(offset) if offset != 0 else 1.0

    kmax = 1.0 / R
    L_half = 2.0 * theta / kmax

    n_half = max(16, int(n_pts // 2))

    # First half: curvature 0 -> kmax -> 0 (upward)
    s1 = np.linspace(0.0, L_half, n_half, dtype=np.float64)
    ds1 = float(s1[1] - s1[0]) if len(s1) > 1 else 0.0

    mid1 = 0.5 * L_half
    k1 = np.where(s1 <= mid1, kmax * (s1 / mid1), kmax * ((L_half - s1) / mid1))
    k1 *= float(sign)

    th1 = np.cumsum(k1) * ds1
    x1 = np.cumsum(np.cos(th1)) * ds1
    y1 = np.cumsum(np.sin(th1)) * ds1

    # Second half: curvature 0 -> -kmax -> 0 (return to horizontal)
    s2 = np.linspace(0.0, L_half, n_half, dtype=np.float64)
    ds2 = float(s2[1] - s2[0]) if len(s2) > 1 else 0.0

    mid2 = 0.5 * L_half
    k2 = np.where(s2 <= mid2, -kmax * (s2 / mid2), -kmax * ((L_half - s2) / mid2))
    k2 *= float(sign)

    th2 = th1[-1] + np.cumsum(k2) * ds2
    x2_rel = np.cumsum(np.cos(th2)) * ds2
    y2_rel = np.cumsum(np.sin(th2)) * ds2

    x = np.concatenate([x1, x1[-1] + x2_rel])
    y = np.concatenate([y1, y1[-1] + y2_rel])

    return x, y


def cosine_sbend_points(lateral_offset_um, length_um, n_pts=256):
    """
    Generate points along a raised cosine S-bend.

    Smooth cosine profile: y(x) = offset * (1 - cos(π*x/L)) / 2
    """
    offset = float(lateral_offset_um)
    L = float(length_um)
    if L <= 0:
        raise ValueError("length_um must be > 0")

    n = max(32, int(n_pts))
    x = np.linspace(0.0, L, n, dtype=np.float64)
    y = offset * (1.0 - np.cos(np.pi * x / L)) / 2.0

    return x, y


def polyline_to_blocks(points_xy, wg_width_um, material):
    """Convert a centerline polyline into rotated mp.Blocks."""
    blocks = []
    pts = np.asarray(points_xy, dtype=np.float64)

    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len <= 1e-9:
            continue

        t = _unit(d)
        n = np.array([-t[1], t[0]], dtype=np.float64)

        center = mp.Vector3(float(0.5 * (a[0] + b[0])), float(0.5 * (a[1] + b[1])), 0.0)
        e1 = mp.Vector3(float(t[0]), float(t[1]), 0.0)
        e2 = mp.Vector3(float(n[0]), float(n[1]), 0.0)

        size = mp.Vector3(seg_len * 1.05, float(wg_width_um), mp.inf)
        blocks.append(mp.Block(size=size, center=center, material=material, e1=e1, e2=e2))

    return blocks


# -------------------------
# Euler S-bend Device
# -------------------------
class EulerSBend2D(Device2DBase):
    """
    Euler S-bend: adiabatic, gradually varying curvature (production).
    """

    def __init__(
        self,
        wg_width=0.45,
        wavelength_um=1.55,
        lateral_offset_um=3.0,
        R_min_um=5.0,
        n_core=None,
        n_clad=1.444,
        port_y_pad=1.0,
        source_shift=0.5,
        fit_margin_um=0.5,
        cell_x=None,
        cell_y=None,
        dpml=None,
        resolution=None,
    ):
        super().__init__(cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)

        self.wg_width = float(wg_width)
        self.wavelength_um = float(wavelength_um)
        self.lateral_offset_um = float(lateral_offset_um)
        self.R_min_um = float(R_min_um)
        self.n_clad = float(n_clad)
        self.port_y_pad = float(port_y_pad)
        self.source_shift = float(source_shift)
        self.fit_margin_um = float(fit_margin_um)

        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width, self.wavelength_um))
            if n_core is None
            else float(n_core)
        )
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

        self.geometry = None
        self.port_in = None
        self.port_out = None
        self.src_vol = None
        self._sbend_pts = None

        self.build_geometry()

    def build_geometry(self):
        """Build Euler S-bend with straight leads."""
        xb, yb = euler_sbend_points(self.lateral_offset_um, self.R_min_um, n_pts=256)

        # Center in x and y
        xb -= 0.5 * (xb.min() + xb.max())
        yb -= 0.5 * (yb.min() + yb.max())
        sbend_pts = np.column_stack([xb, yb])
        self._sbend_pts = sbend_pts.copy()

        # Build geometry
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = +half_y - self.dpml
        sbend_blocks = polyline_to_blocks(sbend_pts, self.wg_width, self.core_medium)

        # Input lead
        sbend_start = sbend_pts[0]
        lead_in_len = abs(sbend_start[0] - (-half_x - self.dpml)) + 1.0
        lead_in = mp.Block(
            size=mp.Vector3(lead_in_len, self.wg_width, mp.inf),
            center=mp.Vector3(sbend_start[0] - 0.5 * lead_in_len, sbend_start[1], 0),
            material=self.core_medium,
        )

        # Output lead
        sbend_end = sbend_pts[-1]
        lead_out_len = abs((half_x + self.dpml) - sbend_end[0]) + 1.0
        lead_out = mp.Block(
            size=mp.Vector3(lead_out_len, self.wg_width, mp.inf),
            center=mp.Vector3(sbend_end[0] + 0.5 * lead_out_len, sbend_end[1], 0),
            material=self.core_medium,
        )

        self.geometry = [lead_in] + sbend_blocks + [lead_out]

        # Device window
        x0, x1 = float(sbend_pts[:, 0].min()), float(sbend_pts[:, 0].max())
        y0, y1 = float(sbend_pts[:, 1].min()), float(sbend_pts[:, 1].max())
        half_w = 0.5 * float(self.wg_width)
        x_extent = max(abs(x0), abs(x1)) + half_w + self.fit_margin_um
        y_extent = max(abs(y0), abs(y1)) + half_w + self.fit_margin_um
        if x_extent > min(abs(nonpml_left), abs(nonpml_right)):
            raise ValueError(
                f"EulerSBend2D exceeds non-PML x extent ({x_extent:.2f} um > "
                f"{min(abs(nonpml_left), abs(nonpml_right)):.2f} um). Reduce lateral_offset_um/R_min_um or increase crop."
            )
        if y_extent > min(abs(nonpml_bot), abs(nonpml_top)):
            raise ValueError(
                f"EulerSBend2D exceeds non-PML y extent ({y_extent:.2f} um > "
                f"{min(abs(nonpml_bot), abs(nonpml_top)):.2f} um). Reduce lateral_offset_um/R_min_um or increase crop."
            )
        self.dev_cx = 0.5 * (x0 + x1)
        self.dev_cy = 0.5 * (y0 + y1)
        self.dev_wx = (x1 - x0) + 2.0 * (self.fit_margin_um + half_w)
        self.dev_wy = (y1 - y0) + 2.0 * (self.fit_margin_um + half_w)

        # Ports
        port_span = self.wg_width + self.port_y_pad
        port_margin = self.dpml + 0.5

        port_in_x = -half_x + port_margin
        self.port_in = mp.Volume(
            center=mp.Vector3(port_in_x, sbend_start[1], 0),
            size=mp.Vector3(0, port_span, 0),
        )

        src_x_left = max(port_in_x - self.source_shift, nonpml_left + 0.1)
        self.src_vol = mp.Volume(
            center=mp.Vector3(src_x_left, sbend_start[1], 0),
            size=self.port_in.size,
        )

        self._src_x_left = float(src_x_left)
        self._src_x_right = float(min((half_x - port_margin) + self.source_shift, nonpml_right - 0.1))

        port_out_x = half_x - port_margin
        self.port_out = mp.Volume(
            center=mp.Vector3(port_out_x, sbend_end[1], 0),
            size=mp.Vector3(0, port_span, 0),
        )

    def core_mask(self, nx, ny, dx, dy):
        """Generate mask for waveguide core."""
        if self._sbend_pts is None:
            return None

        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        mask = np.zeros((ny, nx), dtype=np.uint8)
        half_width = 0.5 * self.wg_width

        # Input lead
        sbend_start = self._sbend_pts[0]
        in_lead = (xx <= sbend_start[0]) & (np.abs(yy - sbend_start[1]) <= half_width)
        mask = np.logical_or(mask, in_lead).astype(np.uint8)

        # S-bend polyline
        min_dist = np.full((ny, nx), np.inf, dtype=np.float64)
        for i in range(len(self._sbend_pts) - 1):
            p0, p1 = self._sbend_pts[i], self._sbend_pts[i + 1]
            dx_seg, dy_seg = p1[0] - p0[0], p1[1] - p0[1]
            seg_len_sq = dx_seg**2 + dy_seg**2
            if seg_len_sq < 1e-12:
                continue

            t = ((xx - p0[0]) * dx_seg + (yy - p0[1]) * dy_seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            closest_x = p0[0] + t * dx_seg
            closest_y = p0[1] + t * dy_seg
            dist = np.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)
            min_dist = np.minimum(min_dist, dist)

        sbend_mask = (min_dist <= half_width).astype(np.uint8)
        mask = np.logical_or(mask, sbend_mask).astype(np.uint8)

        # Output lead
        sbend_end = self._sbend_pts[-1]
        out_lead = (xx >= sbend_end[0]) & (np.abs(yy - sbend_end[1]) <= half_width)
        mask = np.logical_or(mask, out_lead).astype(np.uint8)

        return mask

    def _build_sim_single(self, input_port=1, df_frac=0.1):
        """Build single-frequency simulation."""
        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(+1, 0, 0)
        else:
            src_vol = mp.Volume(
                center=mp.Vector3(self._src_x_right, float(self.port_out.center.y), 0),
                size=self.port_out.size,
            )
            k_in = mp.Vector3(-1, 0, 0)

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
            [mp.Ez], fcen, 0, 1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        return sim, (m1, m2), dft, fcen

    def _build_sim_broadband(self, input_port=1, lam_min_um=1.40, lam_max_um=1.60, Nf=101):
        """Build broadband simulation."""
        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(+1, 0, 0)
        else:
            src_vol = mp.Volume(
                center=mp.Vector3(self._src_x_right, float(self.port_out.center.y), 0),
                size=self.port_out.size,
            )
            k_in = mp.Vector3(-1, 0, 0)

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

    def run_sim(self, decay_tol=1e-5, dir_plus=0, dir_minus=1):
        """Single-frequency S-parameters."""
        toward = {1: +1, 2: -1}

        sim, (m1, m2), dft, _ = self._build_sim_single()
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
        """Broadband spectral response."""
        toward = {1: +1, 2: -1}

        sim, (m1, m2), (_, _, _, _, lams) = self._build_sim_broadband(lam_min_um, lam_max_um, Nf)

        stop = mp.stop_when_fields_decayed(int(n_periods), mp.Ez, self.port_out.center, float(decay_tol))
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


# -------------------------
# Cosine S-bend Device
# -------------------------
class CosineSBend2D(Device2DBase):
    """
    Cosine S-bend: layout-standard, smooth closed-form.
    """

    def __init__(
        self,
        wg_width=0.45,
        wavelength_um=1.55,
        lateral_offset_um=3.0,
        length_um=12.0,
        n_core=None,
        n_clad=1.444,
        port_y_pad=1.0,
        source_shift=0.5,
        fit_margin_um=0.5,
        cell_x=None,
        cell_y=None,
        dpml=None,
        resolution=None,
    ):
        super().__init__(cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)

        self.wg_width = float(wg_width)
        self.wavelength_um = float(wavelength_um)
        self.lateral_offset_um = float(lateral_offset_um)
        self.length_um = float(length_um)
        self.n_clad = float(n_clad)
        self.port_y_pad = float(port_y_pad)
        self.source_shift = float(source_shift)
        self.fit_margin_um = float(fit_margin_um)

        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width, self.wavelength_um))
            if n_core is None
            else float(n_core)
        )
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

        self.geometry = None
        self.port_in = None
        self.port_out = None
        self.src_vol = None
        self._sbend_pts = None

        self.build_geometry()

    def build_geometry(self):
        """Build Cosine S-bend with straight leads."""
        xb, yb = cosine_sbend_points(self.lateral_offset_um, self.length_um, n_pts=256)

        # Center in x and y
        xb -= 0.5 * (xb.min() + xb.max())
        yb -= 0.5 * (yb.min() + yb.max())
        sbend_pts = np.column_stack([xb, yb])
        self._sbend_pts = sbend_pts.copy()

        # Build geometry
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = +half_y - self.dpml
        sbend_blocks = polyline_to_blocks(sbend_pts, self.wg_width, self.core_medium)

        # Input lead
        sbend_start = sbend_pts[0]
        lead_in_len = abs(sbend_start[0] - (-half_x - self.dpml)) + 1.0
        lead_in = mp.Block(
            size=mp.Vector3(lead_in_len, self.wg_width, mp.inf),
            center=mp.Vector3(sbend_start[0] - 0.5 * lead_in_len, sbend_start[1], 0),
            material=self.core_medium,
        )

        # Output lead
        sbend_end = sbend_pts[-1]
        lead_out_len = abs((half_x + self.dpml) - sbend_end[0]) + 1.0
        lead_out = mp.Block(
            size=mp.Vector3(lead_out_len, self.wg_width, mp.inf),
            center=mp.Vector3(sbend_end[0] + 0.5 * lead_out_len, sbend_end[1], 0),
            material=self.core_medium,
        )

        self.geometry = [lead_in] + sbend_blocks + [lead_out]

        # Device window
        x0, x1 = float(sbend_pts[:, 0].min()), float(sbend_pts[:, 0].max())
        y0, y1 = float(sbend_pts[:, 1].min()), float(sbend_pts[:, 1].max())
        half_w = 0.5 * float(self.wg_width)
        x_extent = max(abs(x0), abs(x1)) + half_w + self.fit_margin_um
        y_extent = max(abs(y0), abs(y1)) + half_w + self.fit_margin_um
        if x_extent > min(abs(nonpml_left), abs(nonpml_right)):
            raise ValueError(
                f"CosineSBend2D exceeds non-PML x extent ({x_extent:.2f} um > "
                f"{min(abs(nonpml_left), abs(nonpml_right)):.2f} um). Reduce length_um/offset or increase crop."
            )
        if y_extent > min(abs(nonpml_bot), abs(nonpml_top)):
            raise ValueError(
                f"CosineSBend2D exceeds non-PML y extent ({y_extent:.2f} um > "
                f"{min(abs(nonpml_bot), abs(nonpml_top)):.2f} um). Reduce lateral_offset_um or increase crop."
            )
        self.dev_cx = 0.5 * (x0 + x1)
        self.dev_cy = 0.5 * (y0 + y1)
        self.dev_wx = (x1 - x0) + 2.0 * (self.fit_margin_um + half_w)
        self.dev_wy = (y1 - y0) + 2.0 * (self.fit_margin_um + half_w)

        # Ports
        port_span = self.wg_width + self.port_y_pad
        port_margin = self.dpml + 0.5

        port_in_x = -half_x + port_margin
        self.port_in = mp.Volume(
            center=mp.Vector3(port_in_x, sbend_start[1], 0),
            size=mp.Vector3(0, port_span, 0),
        )

        src_x_left = max(port_in_x - self.source_shift, nonpml_left + 0.1)
        self.src_vol = mp.Volume(
            center=mp.Vector3(src_x_left, sbend_start[1], 0),
            size=self.port_in.size,
        )

        self._src_x_left = float(src_x_left)
        self._src_x_right = float(min((half_x - port_margin) + self.source_shift, nonpml_right - 0.1))

        port_out_x = half_x - port_margin
        self.port_out = mp.Volume(
            center=mp.Vector3(port_out_x, sbend_end[1], 0),
            size=mp.Vector3(0, port_span, 0),
        )

    def core_mask(self, nx, ny, dx, dy):
        """Generate mask for waveguide core."""
        if self._sbend_pts is None:
            return None

        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        mask = np.zeros((ny, nx), dtype=np.uint8)
        half_width = 0.5 * self.wg_width

        # Input lead
        sbend_start = self._sbend_pts[0]
        in_lead = (xx <= sbend_start[0]) & (np.abs(yy - sbend_start[1]) <= half_width)
        mask = np.logical_or(mask, in_lead).astype(np.uint8)

        # S-bend polyline
        min_dist = np.full((ny, nx), np.inf, dtype=np.float64)
        for i in range(len(self._sbend_pts) - 1):
            p0, p1 = self._sbend_pts[i], self._sbend_pts[i + 1]
            dx_seg, dy_seg = p1[0] - p0[0], p1[1] - p0[1]
            seg_len_sq = dx_seg**2 + dy_seg**2
            if seg_len_sq < 1e-12:
                continue

            t = ((xx - p0[0]) * dx_seg + (yy - p0[1]) * dy_seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            closest_x = p0[0] + t * dx_seg
            closest_y = p0[1] + t * dy_seg
            dist = np.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)
            min_dist = np.minimum(min_dist, dist)

        sbend_mask = (min_dist <= half_width).astype(np.uint8)
        mask = np.logical_or(mask, sbend_mask).astype(np.uint8)

        # Output lead
        sbend_end = self._sbend_pts[-1]
        out_lead = (xx >= sbend_end[0]) & (np.abs(yy - sbend_end[1]) <= half_width)
        mask = np.logical_or(mask, out_lead).astype(np.uint8)

        return mask

    def _build_sim_single(self, input_port=1, df_frac=0.1):
        """Build single-frequency simulation."""
        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(+1, 0, 0)
        else:
            src_vol = mp.Volume(
                center=mp.Vector3(self._src_x_right, float(self.port_out.center.y), 0),
                size=self.port_out.size,
            )
            k_in = mp.Vector3(-1, 0, 0)

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
            [mp.Ez], fcen, 0, 1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        return sim, (m1, m2), dft, fcen

    def _build_sim_broadband(self, input_port=1, lam_min_um=1.40, lam_max_um=1.60, Nf=101):
        """Build broadband simulation."""
        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        if int(input_port) == 1:
            src_vol = self.src_vol
            k_in = mp.Vector3(+1, 0, 0)
        else:
            src_vol = mp.Volume(
                center=mp.Vector3(self._src_x_right, float(self.port_out.center.y), 0),
                size=self.port_out.size,
            )
            k_in = mp.Vector3(-1, 0, 0)

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

    def run_sim(self, decay_tol=1e-5, dir_plus=0, dir_minus=1):
        """Single-frequency S-parameters."""
        toward = {1: +1, 2: -1}

        sim, (m1, m2), dft, _ = self._build_sim_single()
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
        """Broadband spectral response."""
        toward = {1: +1, 2: -1}

        sim, (m1, m2), (_, _, _, _, lams) = self._build_sim_broadband(lam_min_um, lam_max_um, Nf)

        stop = mp.stop_when_fields_decayed(int(n_periods), mp.Ez, self.port_out.center, float(decay_tol))
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

