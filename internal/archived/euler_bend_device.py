# euler_bend_device.py
"""
EulerZigZag2D (Meep, 2D effective-index) compatible with your current:
- Device2DBase (cell_x/cell_y/dpml/resolution quantization already handled there)
- utils.neff_siwire_from_tables
- utils.get_mode_alpha_2dir
- utils.pick_in_out_from_alpha
- utils.quantize_square_cell_from_crop

What this file provides:
- Deterministic crop sizing via crop_px + dpml quantization (no "different size boxes")
- Zig-zag Euler-bend polyline converted to rotated mp.Block segments
- Axis-aligned ports via rotation snapping to 0/90/180/270
- Single-frequency S11/S21 (run_sim)
- Broadband spectral S11(λ)/S21(λ) (run_spectrum)

Assumptions:
- get_mode_alpha_2dir(sim, monitor, ...) returns either (ndir,) for single-freq
  or (Nf, ndir) for broadband.
- pick_in_out_from_alpha(alpha, toward_device_sign, dir_plus, dir_minus) supports 1D and 2D alpha.
"""

from __future__ import annotations

import numpy as np
import meep as mp

from devices_base import Device2DBase
from utils import (
    neff_siwire_from_tables,
    get_mode_alpha_2dir,
    pick_in_out_from_alpha,
    quantize_square_cell_from_crop,
)


# -------------------------
# Small geometry helpers
# -------------------------
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _snap_deg_90(angle_deg: float) -> float:
    # Snap to {0,90,180,270}
    a = float(angle_deg)
    return float(int(np.round(a / 90.0)) * 90) % 360.0


def _rotate_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    th = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T


def _euler_bend_points(theta_total_rad: float, R_min_um: float, n_pts: int = 256, sign: float = 1.0):
    """
    Curvature ramps 0 -> kmax -> 0 (triangular curvature profile).
    Total bend angle theta = ∫ k(s) ds = 0.5*kmax*L  => L = 2*theta/kmax = 2*theta*R_min.
    Returns x,y with start at (0,0), end straight.
    """
    theta = float(theta_total_rad)
    if theta <= 0:
        raise ValueError("theta_total_rad must be > 0")
    R = float(R_min_um)
    if R <= 0:
        raise ValueError("R_min_um must be > 0")

    kmax = 1.0 / R
    L = 2.0 * theta / kmax  # 2*theta*R

    n = max(32, int(n_pts))
    s = np.linspace(0.0, L, n, dtype=np.float64)
    ds = float(s[1] - s[0])

    mid = 0.5 * L
    k = np.empty_like(s)
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
        a = np.asarray(a, dtype=np.float64).reshape(2)
        b = np.asarray(b, dtype=np.float64).reshape(2)
        d = b - a
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len <= 1e-9:
            continue

        t = _unit(d)
        n = np.array([-t[1], t[0]], dtype=np.float64)

        center = mp.Vector3(float(0.5 * (a[0] + b[0])), float(0.5 * (a[1] + b[1])), 0.0)
        e1 = mp.Vector3(float(t[0]), float(t[1]), 0.0)   # tangent
        e2 = mp.Vector3(float(n[0]), float(n[1]), 0.0)   # normal

        size = mp.Vector3(seg_len * 1.05, float(wg_width_um), mp.inf)
        blocks.append(mp.Block(size=size, center=center, material=material, e1=e1, e2=e2))
    return blocks


def _axis_from_tangent(t: np.ndarray):
    t = np.asarray(t, dtype=np.float64).reshape(2)
    if abs(float(t[0])) >= abs(float(t[1])):
        return mp.X, float(np.sign(float(t[0])) if float(t[0]) != 0 else 1.0)
    return mp.Y, float(np.sign(float(t[1])) if float(t[1]) != 0 else 1.0)


def _is_axis_aligned(t: np.ndarray, tol: float = 1e-3) -> bool:
    t = _unit(np.asarray(t, dtype=np.float64))
    return (abs(float(t[0])) > 1.0 - tol and abs(float(t[1])) < tol) or (abs(float(t[1])) > 1.0 - tol and abs(float(t[0])) < tol)


# -------------------------
# Device
# -------------------------
class EulerZigZag2D(Device2DBase):
    """
    Zig-zag Euler bends with 2 ports.
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wavelength_um: float = 1.55,
        n_core: float | None = None,
        n_clad: float = 1.444,
        # grid/crop
        crop_px: int = 384,
        resolution: int = 24,
        dpml_um: float = 2.0 / 3.0,
        # path controls
        n_zigs: int = 1,
        start_up: bool = True,
        bend_angle_deg: float = 90.0,
        R_min_um: float = 1.5,
        straight_x_um: float = 1.0,
        straight_y_um: float = 0.8,
        lead_in_um: float = 2.0,
        lead_out_um: float = 2.0,
        y0_um: float = 0.0,
        path_rotation_deg: float = 0.0,  # snapped to 90deg
        # ports/sources
        port_margin_um: float = 0.6,      # distance from non-PML boundary into interior
        port_span_pad_um: float = 0.25,
        source_shift_um: float = 0.5,     # source upstream from input port
        # fitting margins for device window / safety
        fit_margin_um: float = 0.25,
    ):
        crop_px = int(crop_px)
        resolution = int(resolution)
        if crop_px <= 0:
            raise ValueError("crop_px must be > 0")
        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        # Quantize dpml + cell so interior is exactly crop_px x crop_px pixels
        dpml_q, pml_px, cell_um, full_px = quantize_square_cell_from_crop(
            crop_px=crop_px, resolution=resolution, dpml_um=dpml_um
        )

        super().__init__(cell_x=cell_um, cell_y=cell_um, dpml=dpml_q, resolution=resolution)

        self.crop_px = crop_px
        self.pml_px_q = int(pml_px)

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)

        self.n_zigs = int(n_zigs)
        self.start_up = bool(start_up)
        self.bend_angle_deg = float(bend_angle_deg)
        self.R_min_um = float(R_min_um)
        self.straight_x_um = float(straight_x_um)
        self.straight_y_um = float(straight_y_um)
        self.lead_in_um = float(lead_in_um)
        self.lead_out_um = float(lead_out_um)
        self.y0_um = float(y0_um)
        self.path_rotation_deg = _snap_deg_90(path_rotation_deg)

        self.port_margin_um = float(port_margin_um)
        self.port_span_pad_um = float(port_span_pad_um)
        self.source_shift_um = float(source_shift_um)

        self.fit_margin_um = float(fit_margin_um)

        self.n_core = float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um)) if n_core is None else float(n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        # filled by build_geometry()
        self.geometry = None
        self.port_in = None
        self.port_out = None
        self.src_vol = None
        self.full_plane = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(self.cell_x, self.cell_y, 0))

        # tangents at ports
        self.tan_in = None
        self.tan_out = None
        self.axis_in = None
        self.axis_out = None
        self.sign_in = None
        self.sign_out = None

        self.build_geometry()

    def pml_px(self) -> int:
        return int(self.pml_px_q)

    def nonpml_shape(self) -> tuple[int, int]:
        # (ny, nx) of interior (crop) if you crop away PML
        return int(self.crop_px), int(self.crop_px)

    def get_port_y_span_um(self) -> float:
        return float(self.wg_width_um + 2.0 * self.port_span_pad_um)

    # -------------------------
    # Path + ports
    # -------------------------
    def build_geometry(self):
        core = mp.Medium(index=self.n_core)

        # Build local polyline: starts along +x
        pts = []
        x, y = 0.0, 0.0
        pts.append((x, y))

        def add_straight(dx_um: float, dy_um: float):
            nonlocal x, y
            L = float(np.hypot(dx_um, dy_um))
            n = max(8, int(np.ceil(L * self.resolution)))
            for t in np.linspace(0.0, 1.0, n, dtype=np.float64)[1:]:
                pts.append((x + dx_um * float(t), y + dy_um * float(t)))
            x += float(dx_um)
            y += float(dy_um)

        # Input lead
        add_straight(self.lead_in_um, 0.0)

        up = bool(self.start_up)
        for zi in range(max(0, self.n_zigs)):
            sgn = +1.0 if up else -1.0
            th = np.deg2rad(float(self.bend_angle_deg))

            xb, yb, _ = _euler_bend_points(th, self.R_min_um, n_pts=192, sign=sgn)
            for k in range(1, len(xb)):
                pts.append((x + float(xb[k]), y + float(yb[k])))
            x += float(xb[-1])
            y += float(yb[-1])

            # straight along bent direction
            add_straight(float(np.cos(sgn * th)) * self.straight_y_um,
                         float(np.sin(sgn * th)) * self.straight_y_um)

            # bend back to +x
            xb2, yb2, _ = _euler_bend_points(th, self.R_min_um, n_pts=192, sign=-sgn)
            rot = sgn * th
            c, s = float(np.cos(rot)), float(np.sin(rot))
            for k in range(1, len(xb2)):
                xx, yy = float(xb2[k]), float(yb2[k])
                xr = c * xx - s * yy
                yr = s * xx + c * yy
                pts.append((x + xr, y + yr))
            xx, yy = float(xb2[-1]), float(yb2[-1])
            xr_end = c * xx - s * yy
            yr_end = s * xx + c * yy
            x += xr_end
            y += yr_end

            # between-zig straight
            add_straight(self.straight_x_um, 0.0)

            up = not up

        # Output lead
        add_straight(self.lead_out_um, 0.0)

        pts = np.asarray(pts, dtype=np.float64)

        # Center + y-offset in local coords
        pts[:, 0] -= 0.5 * float(pts[:, 0].min() + pts[:, 0].max())
        pts[:, 1] -= 0.5 * float(pts[:, 1].min() + pts[:, 1].max())
        pts[:, 1] += self.y0_um

        # Snap-rotate (ports axis-aligned)
        pts = _rotate_points(pts, self.path_rotation_deg)

        # Tangents at ends
        tan_in = _unit(pts[1] - pts[0])
        tan_out = _unit(pts[-1] - pts[-2])

        # Must be axis-aligned after snap-rotation
        if not (_is_axis_aligned(tan_in) and _is_axis_aligned(tan_out)):
            raise ValueError("Port tangents are not axis-aligned. Set path_rotation_deg to multiples of 90.")

        self.tan_in = tan_in
        self.tan_out = tan_out
        self.axis_in, self.sign_in = _axis_from_tangent(tan_in)
        self.axis_out, self.sign_out = _axis_from_tangent(tan_out)

        # Fit check inside non-PML interior with margin
        half_x = 0.5 * float(self.cell_x)
        half_y = 0.5 * float(self.cell_y)
        left = -half_x + float(self.dpml) + float(self.fit_margin_um)
        right = +half_x - float(self.dpml) - float(self.fit_margin_um)
        bot = -half_y + float(self.dpml) + float(self.fit_margin_um)
        top = +half_y - float(self.dpml) - float(self.fit_margin_um)

        r = 0.5 * float(self.wg_width_um)
        if (pts[:, 0].min() - r) < left or (pts[:, 0].max() + r) > right or (pts[:, 1].min() - r) < bot or (pts[:, 1].max() + r) > top:
            raise ValueError("Waveguide does not fit inside non-PML interior; increase cell/crop or reduce geometry.")

        # Build geometry blocks
        self.geometry = _polyline_to_blocks(pts, wg_width_um=self.wg_width_um, material=core)

        # Device window for ML conditioning: bounding box + small margin
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        half_w = 0.5 * float(self.wg_width_um)
        self.dev_cx = 0.5 * (x0 + x1)
        self.dev_cy = 0.5 * (y0 + y1)
        self.dev_wx = (x1 - x0) + 2.0 * (float(self.fit_margin_um) + half_w)
        self.dev_wy = (y1 - y0) + 2.0 * (float(self.fit_margin_um) + half_w)

        # Ports: place them near non-PML boundary (inside) along the axis direction
        span = self.get_port_y_span_um()

        def place_port_on_interior_boundary(tangent: np.ndarray, margin: float, which: str):
            # which = "in" or "out"
            tangent = _unit(tangent)
            axis, sgn = _axis_from_tangent(tangent)

            half_x = 0.5 * float(self.cell_x)
            half_y = 0.5 * float(self.cell_y)
            nx_left = -half_x + float(self.dpml)
            nx_right = +half_x - float(self.dpml)
            ny_bot = -half_y + float(self.dpml)
            ny_top = +half_y - float(self.dpml)

            if axis == mp.X:
                # tangent is ±x
                x = (nx_left + margin) if sgn > 0 else (nx_right - margin)
                y = 0.0
                size = mp.Vector3(0, span, 0)
                kpoint = mp.Vector3(sgn, 0, 0)
            else:
                # tangent is ±y
                y = (ny_bot + margin) if sgn > 0 else (ny_top - margin)
                x = 0.0
                size = mp.Vector3(span, 0, 0)
                kpoint = mp.Vector3(0, sgn, 0)

            return mp.Vector3(x, y, 0), size, kpoint, axis, sgn

        # Input port: tangent into device is tan_in (points from port into device)
        port_in_center, port_in_size, k_in, axis_in, sgn_in = place_port_on_interior_boundary(self.tan_in, self.port_margin_um, "in")
        self.port_in = mp.Volume(center=port_in_center, size=port_in_size)

        # Source is upstream of input port (opposite tangent direction)
        if axis_in == mp.X:
            src_center = mp.Vector3(float(port_in_center.x) - float(sgn_in) * float(self.source_shift_um), float(port_in_center.y), 0)
            src_size = port_in_size
        else:
            src_center = mp.Vector3(float(port_in_center.x), float(port_in_center.y) - float(sgn_in) * float(self.source_shift_um), 0)
            src_size = port_in_size
        self.src_vol = mp.Volume(center=src_center, size=src_size)

        # Output port: tangent out of device is tan_out (points from device toward port boundary)
        port_out_center, port_out_size, k_out, axis_out, sgn_out = place_port_on_interior_boundary(self.tan_out, self.port_margin_um, "out")
        self.port_out = mp.Volume(center=port_out_center, size=port_out_size)

        # Store for simulation
        self._k_in = k_in
        self._k_out = k_out
        self.axis_in = axis_in
        self.axis_out = axis_out
        self.sign_in = sgn_in
        self.sign_out = sgn_out

    # -------------------------
    # Build sims
    # -------------------------
    def _build_sim_single(self, df_frac: float = 0.1):
        lam = float(self.wavelength_um)
        fcen = 1.0 / lam
        fwidth = float(df_frac) * fcen

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=self.src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=self._k_in,
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

        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_in, direction=self.axis_in))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_out, direction=self.axis_out))

        dft = sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=self.full_plane.center, size=self.full_plane.size)
        return sim, (m1, m2), dft, fcen

    def _build_sim_broadband(self, lam_min_um: float, lam_max_um: float, Nf: int):
        lam_min_um = float(lam_min_um)
        lam_max_um = float(lam_max_um)
        Nf = int(Nf)

        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df),
                volume=self.src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=self._k_in,
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

        m1 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_in, direction=self.axis_in))
        m2 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_out, direction=self.axis_out))
        return sim, (m1, m2), (fcen, df, Nf, freqs, lams)

    # -------------------------
    # Single-frequency S-params
    # -------------------------
    def run_sim(self, decay_tol: float = 1e-6, dir_plus: int = 0, dir_minus: int = 1):
        """
        Returns:
          eps_mid [ny,nx], Ez_mid [ny,nx] complex (at fcen),
          S11 complex, S21 complex, (cell_x, cell_y)
        """
        sim, (m1, m2), dft, _fcen = self._build_sim_single(df_frac=0.1)
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_mid = sim.get_epsilon().T.astype(np.float32)
        Ez_mid = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        # toward_device sign convention along each port axis:
        # input port incoming toward device is along +tan_in => sign_in
        toward1 = int(np.sign(self.sign_in))  # +1 means incoming is +axis
        # output port incoming toward device is opposite tan_out => -sign_out
        toward2 = int(-np.sign(self.sign_out))

        alpha1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)  # (ndir,) or (1,ndir)
        alpha2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)

        alpha1 = np.asarray(alpha1)
        alpha2 = np.asarray(alpha2)
        if alpha1.ndim == 2:
            alpha1 = alpha1[0]
        if alpha2.ndim == 2:
            alpha2 = alpha2[0]

        a1_in, b1_out = pick_in_out_from_alpha(alpha1, toward1, dir_plus=dir_plus, dir_minus=dir_minus)
        _a2_in, b2_out = pick_in_out_from_alpha(alpha2, toward2, dir_plus=dir_plus, dir_minus=dir_minus)

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
        lam_min_um: float = 1.40,
        lam_max_um: float = 1.60,
        Nf: int = 101,
        decay_tol: float = 1e-6,
        n_periods: int = 50,
        dir_plus: int = 0,
        dir_minus: int = 1,
    ):
        """
        Returns:
          lams [Nf] (um),
          S11 [Nf] complex,
          S21 [Nf] complex
        """
        sim, (m1, m2), (fcen, df, Nf, freqs, lams) = self._build_sim_broadband(lam_min_um, lam_max_um, Nf)

        stop = mp.stop_when_fields_decayed(int(n_periods), mp.Ez, self.port_out.center, float(decay_tol))
        sim.run(until_after_sources=stop)

        res1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.NO_PARITY)
        res2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.NO_PARITY)

        alpha1 = res1.alpha[:, 0, :]  # (Nf, ndir)
        alpha2 = res2.alpha[:, 0, :]  # (Nf, ndir)

        toward1 = int(np.sign(self.sign_in))
        toward2 = int(-np.sign(self.sign_out))

        a1_in, b1_out = pick_in_out_from_alpha(alpha1, toward1, dir_plus=dir_plus, dir_minus=dir_minus)
        _a2_in, b2_out = pick_in_out_from_alpha(alpha2, toward2, dir_plus=dir_plus, dir_minus=dir_minus)

        S11 = b1_out / a1_in
        S21 = b2_out / a1_in

        sim.reset_meep()

        self.lams = lams
        self.S11_spec = S11
        self.S21_spec = S21
        return lams, S11, S21
