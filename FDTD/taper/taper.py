# taper.py
import meep as mp
import numpy as np

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    neff_siwire_from_tables,
    pick_in_out_from_alpha,
    get_mode_alpha_2dir,
)
from devices_base import Device2DBase


class TaperWaveguide2D(Device2DBase):
    """
    2D linear taper waveguide with two mode ports (left/right).

    Geometry uses a single polygon/prism (angled edges) instead of stitched blocks:
      - Left lead: width = wg_width_in
      - Taper section (device window): linearly transitions to wg_width_out
      - Right lead: width = wg_width_out

    Supports:
      - run_sim(): single-frequency eps + Ez snapshot + S11/S21
      - run_spectrum(): broadband wavelength grid + S11(λ), S21(λ)
    """

    def __init__(
        self,
        wg_width_in=0.45,
        wg_width_out=1.20,
        wavelength_um=1.55,
        taper_length_um=12.0,
        n_core=None,
        n_clad=1.444,
        port_y_pad=1.0,
        source_shift=0.5,
        fit_margin_um=0.5,
        # optional sim window overrides
        cell_x=None,
        cell_y=None,
        dpml=None,
        resolution=None,
        # constraints
        clamp_widths=(0.38, 2.50),
    ):
        super().__init__(cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)

        wmin, wmax = float(clamp_widths[0]), float(clamp_widths[1])
        self.wg_width_in = float(np.clip(float(wg_width_in), wmin, wmax))
        self.wg_width_out = float(np.clip(float(wg_width_out), wmin, wmax))

        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)
        self.source_shift = float(source_shift)

        self.port_y_pad = float(port_y_pad)
        self.fit_margin_um = float(fit_margin_um)

        self.taper_length_um = float(taper_length_um)

        # Clamp taper length to fit inside non-PML interior
        L_inner = float(self.cell_x) - 2.0 * float(self.dpml)
        max_len = L_inner - 2.0 * float(self.fit_margin_um)
        self.taper_length_um = float(np.clip(self.taper_length_um, 1.0, max_len))

        # Ports: placed at non-PML boundary (inside non-PML region by small margin)
        self.port_y_span = max(self.wg_width_in, self.wg_width_out) + float(self.port_y_pad)
        port_margin = self.dpml + 0.5  # 0.5 µm inside the non-PML region

        # Cell boundaries
        half_x = 0.5 * self.cell_x

        # Input port: on left edge, inside non-PML region
        self._x_in = -half_x + port_margin
        # Output port: on right edge, inside non-PML region
        self._x_out = half_x - port_margin

        # Conditioning device window (covers taper region only)
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = float(self.taper_length_um)
        self.dev_wy = max(self.wg_width_in, self.wg_width_out) + 2.0 * (self.port_y_pad + self.fit_margin_um)

        # Effective-index core (single neff; you can refine to in/out if you want)
        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width_in, self.wavelength_um))
            if n_core is None
            else float(n_core)
        )
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)

        # For single-frequency Ez snapshots
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

    # -------------------------
    # Geometry (angled edges)
    # -------------------------
    def build_geometry(self):
        # A single polygon with tapered sidewalls spanning taper_length_um.
        xL = -0.5 * float(self.taper_length_um)
        xR = +0.5 * float(self.taper_length_um)

        w1 = float(self.wg_width_in)
        w2 = float(self.wg_width_out)

        y1 = 0.5 * w1
        y2 = 0.5 * w2

        verts = [
            mp.Vector3(xL, +y1, 0),
            mp.Vector3(xR, +y2, 0),
            mp.Vector3(xR, -y2, 0),
            mp.Vector3(xL, -y1, 0),
        ]

        taper_prism = mp.Prism(
            vertices=verts,
            height=1e9,              # not mp.inf
            material=self.core_medium,
        )

        # Add straight leads to the non-PML boundaries
        half_x = 0.5 * float(self.cell_x)
        lead_in_len = abs(xL - (-half_x - self.dpml)) + 1.0
        lead_in = mp.Block(
            size=mp.Vector3(lead_in_len, w1, mp.inf),
            center=mp.Vector3(xL - 0.5 * lead_in_len, 0.0, 0.0),
            material=self.core_medium,
        )
        lead_out_len = abs((half_x + self.dpml) - xR) + 1.0
        lead_out = mp.Block(
            size=mp.Vector3(lead_out_len, w2, mp.inf),
            center=mp.Vector3(xR + 0.5 * lead_out_len, 0.0, 0.0),
            material=self.core_medium,
        )

        self.geometry = [lead_in, taper_prism, lead_out]


        port_size = mp.Vector3(0, self.port_y_span, 0)
        self.port_in = mp.Volume(center=mp.Vector3(self._x_in, 0, 0), size=port_size)
        self.port_out = mp.Volume(center=mp.Vector3(self._x_out, 0, 0), size=port_size)

        src_center = mp.Vector3(float(self.port_in.center.x) - self.source_shift, 0, 0)
        self.src_vol = mp.Volume(center=src_center, size=self.port_in.size)

    # -------------------------
    # Masks for overlay notebooks
    # -------------------------
    def core_mask(self, nx, ny, dx, dy):
        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        xL = -0.5 * float(self.taper_length_um)
        xR = +0.5 * float(self.taper_length_um)
        w1 = float(self.wg_width_in)
        w2 = float(self.wg_width_out)

        # Piecewise half-width: constant -> linear taper -> constant
        half = np.empty_like(xx, dtype=float)
        half[xx <= xL] = 0.5 * w1
        half[xx >= xR] = 0.5 * w2
        mid = (xx > xL) & (xx < xR)
        if np.any(mid):
            t = (xx[mid] - xL) / (xR - xL)
            w = (1.0 - t) * w1 + t * w2
            half[mid] = 0.5 * w

        return (np.abs(yy) <= half).astype(np.uint8)

    # -------------------------
    # Helpers (broadband grid)
    # -------------------------
    @staticmethod
    def _spectrum_from_lams(lam_min_um, lam_max_um, Nf):
        lam_min_um = float(lam_min_um)
        lam_max_um = float(lam_max_um)
        Nf = int(Nf)
        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)
        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs
        return fcen, df, Nf, freqs, lams

    @staticmethod
    def _stop_when_decayed(tol, n_periods, at, component=mp.Ez):
        return mp.stop_when_fields_decayed(int(n_periods), component, at, float(tol))

    # -------------------------
    # Build simulations
    # -------------------------
    def _build_sim_single(self, input_port=1, df_frac=0.1):
        lam = float(self.wavelength_um)
        fcen = 1.0 / lam
        fwidth = float(df_frac) * fcen

        if int(input_port) == 1:
            src_center = mp.Vector3(float(self.port_in.center.x) - self.source_shift, 0, 0)
            src_vol = mp.Volume(center=src_center, size=self.port_in.size)
            k = mp.Vector3(+1, 0, 0)
        else:
            src_center = mp.Vector3(float(self.port_out.center.x) + self.source_shift, 0, 0)
            src_vol = mp.Volume(center=src_center, size=self.port_out.size)
            k = mp.Vector3(-1, 0, 0)

        self.src_vol = src_vol

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k,
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
        fcen, df, Nf, freqs, lams = self._spectrum_from_lams(lam_min_um, lam_max_um, Nf)

        if int(input_port) == 1:
            src_center = mp.Vector3(float(self.port_in.center.x) - self.source_shift, 0, 0)
            src_vol = mp.Volume(center=src_center, size=self.port_in.size)
            k = mp.Vector3(+1, 0, 0)
        else:
            src_center = mp.Vector3(float(self.port_out.center.x) + self.source_shift, 0, 0)
            src_vol = mp.Volume(center=src_center, size=self.port_out.size)
            k = mp.Vector3(-1, 0, 0)

        self.src_vol = src_vol

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k,
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
        toward = {1: +1, 2: -1}

        sim, (m1, m2), (_fcen, _df, _Nf, _freqs, lams) = self._build_sim_broadband(
            input_port=1, lam_min_um=lam_min_um, lam_max_um=lam_max_um, Nf=Nf
        )

        stop = self._stop_when_decayed(
            tol=decay_tol,
            n_periods=n_periods,
            at=self.port_out.center,
            component=mp.Ez,
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
