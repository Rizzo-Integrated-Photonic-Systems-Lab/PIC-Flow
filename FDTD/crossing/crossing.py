# crossing.py
"""
Waveguide crossings (2D effective-index) - 4-port devices at 90°.

Two types of crossings:
1. UniformCrossing2D: Plain uniform-width 90° crossing (baseline)
2. TaperedCrossing2D: Low-loss tapered/expanded-intersection with symmetric width expansion

All devices:
- 4 ports: Port 1 (left), Port 2 (right), Port 3 (bottom), Port 4 (top)
- Inherit from Device2DBase (default 26x26 µm cell with 1 µm PML)
- Support run_sim() for single-frequency S-parameters (4-port)
- Support run_spectrum() for broadband spectral response
- Can excite from any port
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
# Uniform 90° Crossing
# -------------------------
class UniformCrossing2D(Device2DBase):
    """
    Plain uniform-width 90° crossing (baseline).

    Two waveguides intersect at right angles:
    - WG_H: horizontal (port_1 left, port_2 right)
    - WG_V: vertical (port_3 bottom, port_4 top)

    4-port device with uniform waveguide width throughout.
    """

    def __init__(
        self,
        wg_width_h=0.45,
        wg_width_v=0.45,
        wavelength_um=1.55,
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

        self.wg_width_h = float(wg_width_h)
        self.wg_width_v = float(wg_width_v)
        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)
        self.port_y_pad = float(port_y_pad)
        self.source_shift = float(source_shift)
        self.fit_margin_um = float(fit_margin_um)

        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width_h, self.wavelength_um))
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
        self.port_1 = None  # Left
        self.port_2 = None  # Right
        self.port_3 = None  # Bottom
        self.port_4 = None  # Top
        self.src_vol_1 = None
        self.src_vol_2 = None
        self.src_vol_3 = None
        self.src_vol_4 = None

        self.build_geometry()

    def build_geometry(self):
        """Build uniform 90° crossing geometry."""
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y

        # Horizontal waveguide (infinite along x)
        wg_h = mp.Block(
            size=mp.Vector3(mp.inf, self.wg_width_h, mp.inf),
            center=mp.Vector3(0, 0, 0),
            material=self.core_medium,
        )

        # Vertical waveguide (infinite along y)
        wg_v = mp.Block(
            size=mp.Vector3(self.wg_width_v, mp.inf, mp.inf),
            center=mp.Vector3(0, 0, 0),
            material=self.core_medium,
        )

        self.geometry = [wg_h, wg_v]

        # Device window: covers crossing region
        crossing_size = max(self.wg_width_h, self.wg_width_v) + 2.0 * self.fit_margin_um
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = crossing_size
        self.dev_wy = crossing_size

        # Ports: 4 ports at edges
        port_span = max(self.wg_width_h, self.wg_width_v) + self.port_y_pad
        port_margin = self.dpml + 0.5

        # Port 1: left edge (horizontal WG, +x direction)
        port_1_x = -half_x + port_margin
        self.port_1 = mp.Volume(
            center=mp.Vector3(port_1_x, 0, 0),
            size=mp.Vector3(0, port_span, 0),
        )
        self.src_vol_1 = mp.Volume(
            center=mp.Vector3(port_1_x - self.source_shift, 0, 0),
            size=self.port_1.size,
        )

        # Port 2: right edge (horizontal WG, -x direction for incoming)
        port_2_x = half_x - port_margin
        self.port_2 = mp.Volume(
            center=mp.Vector3(port_2_x, 0, 0),
            size=mp.Vector3(0, port_span, 0),
        )
        self.src_vol_2 = mp.Volume(
            center=mp.Vector3(port_2_x + self.source_shift, 0, 0),
            size=self.port_2.size,
        )

        # Port 3: bottom edge (vertical WG, +y direction)
        port_3_y = -half_y + port_margin
        self.port_3 = mp.Volume(
            center=mp.Vector3(0, port_3_y, 0),
            size=mp.Vector3(port_span, 0, 0),
        )
        self.src_vol_3 = mp.Volume(
            center=mp.Vector3(0, port_3_y - self.source_shift, 0),
            size=self.port_3.size,
        )

        # Port 4: top edge (vertical WG, -y direction for incoming)
        port_4_y = half_y - port_margin
        self.port_4 = mp.Volume(
            center=mp.Vector3(0, port_4_y, 0),
            size=mp.Vector3(port_span, 0, 0),
        )
        self.src_vol_4 = mp.Volume(
            center=mp.Vector3(0, port_4_y + self.source_shift, 0),
            size=self.port_4.size,
        )

    def get_ports(self):
        """Override to provide all 4 ports."""
        return {
            "port_1": self.port_1,
            "port_2": self.port_2,
            "port_3": self.port_3,
            "port_4": self.port_4,
        }

    def get_sources(self):
        """Override to provide all 4 source volumes."""
        return {
            "src_1": self.src_vol_1,
            "src_2": self.src_vol_2,
            "src_3": self.src_vol_3,
            "src_4": self.src_vol_4,
        }

    def core_mask(self, nx, ny, dx, dy):
        """Generate mask for waveguide core (horizontal + vertical)."""
        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        # Horizontal waveguide
        mask_h = (np.abs(yy) <= 0.5 * self.wg_width_h).astype(np.uint8)

        # Vertical waveguide
        mask_v = (np.abs(xx) <= 0.5 * self.wg_width_v).astype(np.uint8)

        # Combine masks
        mask = np.logical_or(mask_h, mask_v).astype(np.uint8)
        return mask

    def _build_sim_single(self, input_port=1, df_frac=0.1):
        """Build single-frequency simulation."""
        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen

        # Select source and k-vector based on input port
        if int(input_port) == 1:
            src_vol = self.src_vol_1
            k_vec = mp.Vector3(+1, 0, 0)
        elif int(input_port) == 2:
            src_vol = self.src_vol_2
            k_vec = mp.Vector3(-1, 0, 0)
        elif int(input_port) == 3:
            src_vol = self.src_vol_3
            k_vec = mp.Vector3(0, +1, 0)
        else:  # port 4
            src_vol = self.src_vol_4
            k_vec = mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_vec,
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
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        return sim, (m1, m2, m3, m4), dft, fcen

    def _build_sim_broadband(self, input_port=1, lam_min_um=1.40, lam_max_um=1.60, Nf=101):
        """Build broadband simulation."""
        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        # Select source and k-vector based on input port
        if int(input_port) == 1:
            src_vol = self.src_vol_1
            k_vec = mp.Vector3(+1, 0, 0)
        elif int(input_port) == 2:
            src_vol = self.src_vol_2
            k_vec = mp.Vector3(-1, 0, 0)
        elif int(input_port) == 3:
            src_vol = self.src_vol_3
            k_vec = mp.Vector3(0, +1, 0)
        else:  # port 4
            src_vol = self.src_vol_4
            k_vec = mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_vec,
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

        m1 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_1))
        m2 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_2))
        m3 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_3))
        m4 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_4))

        return sim, (m1, m2, m3, m4), (fcen, df, Nf, freqs, lams)

    def run_sim(self, input_port=1, decay_tol=1e-5, dir_plus=0, dir_minus=1):
        """
        Single-frequency S-parameters for 4-port device.

        Returns:
          eps_mid, Ez_mid, S-matrix dict, (cell_x, cell_y)
        """
        # Toward convention: direction from port toward device interior
        # Port 1 (left, +x): toward[1] = +1
        # Port 2 (right, -x): toward[2] = -1
        # Port 3 (bottom, +y): toward[3] = +1
        # Port 4 (top, -y): toward[4] = -1
        toward = {1: +1, 2: -1, 3: +1, 4: -1}

        sim, (m1, m2, m3, m4), dft, _ = self._build_sim_single(input_port=input_port)
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_mid = sim.get_epsilon().T.astype(np.float32)
        Ez_mid = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        # Get mode coefficients for all 4 ports
        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        alpha_3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        alpha_4 = get_mode_alpha_2dir(sim, m4, band=1, eig_parity=mp.NO_PARITY)

        # Extract incoming and outgoing waves
        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
        a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)
        a3_in, b3_out = pick_in_out_from_alpha(alpha_3, toward[3], dir_plus=dir_plus, dir_minus=dir_minus)
        a4_in, b4_out = pick_in_out_from_alpha(alpha_4, toward[4], dir_plus=dir_plus, dir_minus=dir_minus)

        # Determine which port is the input and build S-matrix
        if int(input_port) == 1:
            a_in = a1_in
            S = {
                'S11': b1_out / a_in,
                'S21': b2_out / a_in,
                'S31': b3_out / a_in,
                'S41': b4_out / a_in,
            }
        elif int(input_port) == 2:
            a_in = a2_in
            S = {
                'S12': b1_out / a_in,
                'S22': b2_out / a_in,
                'S32': b3_out / a_in,
                'S42': b4_out / a_in,
            }
        elif int(input_port) == 3:
            a_in = a3_in
            S = {
                'S13': b1_out / a_in,
                'S23': b2_out / a_in,
                'S33': b3_out / a_in,
                'S43': b4_out / a_in,
            }
        else:  # port 4
            a_in = a4_in
            S = {
                'S14': b1_out / a_in,
                'S24': b2_out / a_in,
                'S34': b3_out / a_in,
                'S44': b4_out / a_in,
            }

        sim.reset_meep()

        # Store S-parameters
        for key, val in S.items():
            setattr(self, key, val)

        return eps_mid, Ez_mid, S, (self.cell_x, self.cell_y)

    def run_spectrum(
        self,
        input_port=1,
        lam_min_um=1.40,
        lam_max_um=1.60,
        Nf=101,
        decay_tol=1e-6,
        n_periods=50,
        dir_plus=0,
        dir_minus=1,
    ):
        """Broadband spectral response for 4-port device."""
        toward = {1: +1, 2: -1, 3: +1, 4: -1}

        sim, (m1, m2, m3, m4), (_, _, _, _, lams) = self._build_sim_broadband(
            input_port=input_port, lam_min_um=lam_min_um, lam_max_um=lam_max_um, Nf=Nf
        )

        # Use opposite port for decay detection (through transmission)
        decay_ports = {1: self.port_2, 2: self.port_1, 3: self.port_4, 4: self.port_3}
        stop = mp.stop_when_fields_decayed(int(n_periods), mp.Ez, decay_ports[input_port].center, float(decay_tol))
        sim.run(until_after_sources=stop)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        alpha_3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        alpha_4 = get_mode_alpha_2dir(sim, m4, band=1, eig_parity=mp.NO_PARITY)

        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
        a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)
        a3_in, b3_out = pick_in_out_from_alpha(alpha_3, toward[3], dir_plus=dir_plus, dir_minus=dir_minus)
        a4_in, b4_out = pick_in_out_from_alpha(alpha_4, toward[4], dir_plus=dir_plus, dir_minus=dir_minus)

        if int(input_port) == 1:
            a_in = a1_in
            S = {
                'S11': b1_out / a_in,
                'S21': b2_out / a_in,
                'S31': b3_out / a_in,
                'S41': b4_out / a_in,
            }
        elif int(input_port) == 2:
            a_in = a2_in
            S = {
                'S12': b1_out / a_in,
                'S22': b2_out / a_in,
                'S32': b3_out / a_in,
                'S42': b4_out / a_in,
            }
        elif int(input_port) == 3:
            a_in = a3_in
            S = {
                'S13': b1_out / a_in,
                'S23': b2_out / a_in,
                'S33': b3_out / a_in,
                'S43': b4_out / a_in,
            }
        else:
            a_in = a4_in
            S = {
                'S14': b1_out / a_in,
                'S24': b2_out / a_in,
                'S34': b3_out / a_in,
                'S44': b4_out / a_in,
            }

        sim.reset_meep()

        self.lams = lams
        for key, val in S.items():
            setattr(self, key + '_spec', val)

        return lams, S


# -------------------------
# Tapered 90° Crossing
# -------------------------
class TaperedCrossing2D(Device2DBase):
    """
    Low-loss tapered/expanded-intersection 90° crossing.

    Symmetric width expansion at crossing: w_in → w_max → w_in
    Taper length L_taper on each side of crossing.

    4-port device with tapered intersection for reduced loss/crosstalk.
    """

    def __init__(
        self,
        wg_width_h=0.45,
        wg_width_v=0.45,
        crossing_width=1.2,  # Maximum width at crossing center
        taper_length_um=3.0,  # Taper length on each side
        wavelength_um=1.55,
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

        self.wg_width_h = float(wg_width_h)
        self.wg_width_v = float(wg_width_v)
        self.crossing_width = float(crossing_width)
        self.taper_length_um = float(taper_length_um)
        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)
        self.port_y_pad = float(port_y_pad)
        self.source_shift = float(source_shift)
        self.fit_margin_um = float(fit_margin_um)

        self.n_core = (
            float(neff_siwire_from_tables(self.wg_width_h, self.wavelength_um))
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
        self.port_1 = None
        self.port_2 = None
        self.port_3 = None
        self.port_4 = None
        self.src_vol_1 = None
        self.src_vol_2 = None
        self.src_vol_3 = None
        self.src_vol_4 = None

        self.build_geometry()

    def build_geometry(self):
        """Build tapered 90° crossing with expanded intersection."""
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        nonpml_half_x = half_x - self.dpml
        nonpml_half_y = half_y - self.dpml

        min_crossing_width = max(self.wg_width_h, self.wg_width_v)
        if self.crossing_width < min_crossing_width:
            raise ValueError(
                f"TaperedCrossing2D: crossing_width ({self.crossing_width:.3f}) must be >= "
                f"max(wg_width_h, wg_width_v) ({min_crossing_width:.3f})."
            )
        max_taper_x = nonpml_half_x - self.fit_margin_um
        max_taper_y = nonpml_half_y - self.fit_margin_um
        if self.taper_length_um <= 0:
            raise ValueError("TaperedCrossing2D: taper_length_um must be > 0.")
        if self.taper_length_um >= max_taper_x or self.taper_length_um >= max_taper_y:
            raise ValueError(
                f"TaperedCrossing2D: taper_length_um ({self.taper_length_um:.3f}) leaves no room for straight sections "
                f"inside the non-PML region ({max_taper_x:.3f} x {max_taper_y:.3f} um available)."
            )

        # Horizontal waveguide with taper at center
        # Left section: uniform
        wg_h_left = mp.Block(
            size=mp.Vector3(half_x - self.taper_length_um, self.wg_width_h, mp.inf),
            center=mp.Vector3(-0.5 * (half_x + self.taper_length_um), 0, 0),
            material=self.core_medium,
        )

        # Right section: uniform
        wg_h_right = mp.Block(
            size=mp.Vector3(half_x - self.taper_length_um, self.wg_width_h, mp.inf),
            center=mp.Vector3(0.5 * (half_x + self.taper_length_um), 0, 0),
            material=self.core_medium,
        )

        # Horizontal taper (left): wg_width_h → crossing_width
        wg_h_taper_left = mp.Prism(
            vertices=[
                mp.Vector3(-self.taper_length_um, -0.5 * self.wg_width_h, 0),
                mp.Vector3(0, -0.5 * self.crossing_width, 0),
                mp.Vector3(0, 0.5 * self.crossing_width, 0),
                mp.Vector3(-self.taper_length_um, 0.5 * self.wg_width_h, 0),
            ],
            height=1e9,
            material=self.core_medium,
        )

        # Horizontal taper (right): crossing_width → wg_width_h
        wg_h_taper_right = mp.Prism(
            vertices=[
                mp.Vector3(0, -0.5 * self.crossing_width, 0),
                mp.Vector3(self.taper_length_um, -0.5 * self.wg_width_h, 0),
                mp.Vector3(self.taper_length_um, 0.5 * self.wg_width_h, 0),
                mp.Vector3(0, 0.5 * self.crossing_width, 0),
            ],
            height=1e9,
            material=self.core_medium,
        )

        # Vertical waveguide with taper at center
        # Bottom section: uniform
        wg_v_bottom = mp.Block(
            size=mp.Vector3(self.wg_width_v, half_y - self.taper_length_um, mp.inf),
            center=mp.Vector3(0, -0.5 * (half_y + self.taper_length_um), 0),
            material=self.core_medium,
        )

        # Top section: uniform
        wg_v_top = mp.Block(
            size=mp.Vector3(self.wg_width_v, half_y - self.taper_length_um, mp.inf),
            center=mp.Vector3(0, 0.5 * (half_y + self.taper_length_um), 0),
            material=self.core_medium,
        )

        # Vertical taper (bottom): wg_width_v → crossing_width
        wg_v_taper_bottom = mp.Prism(
            vertices=[
                mp.Vector3(-0.5 * self.wg_width_v, -self.taper_length_um, 0),
                mp.Vector3(-0.5 * self.crossing_width, 0, 0),
                mp.Vector3(0.5 * self.crossing_width, 0, 0),
                mp.Vector3(0.5 * self.wg_width_v, -self.taper_length_um, 0),
            ],
            height=1e9,
            material=self.core_medium,
        )

        # Vertical taper (top): crossing_width → wg_width_v
        wg_v_taper_top = mp.Prism(
            vertices=[
                mp.Vector3(-0.5 * self.crossing_width, 0, 0),
                mp.Vector3(-0.5 * self.wg_width_v, self.taper_length_um, 0),
                mp.Vector3(0.5 * self.wg_width_v, self.taper_length_um, 0),
                mp.Vector3(0.5 * self.crossing_width, 0, 0),
            ],
            height=1e9,
            material=self.core_medium,
        )

        self.geometry = [
            wg_h_left, wg_h_right, wg_h_taper_left, wg_h_taper_right,
            wg_v_bottom, wg_v_top, wg_v_taper_bottom, wg_v_taper_top,
        ]

        # Device window: covers crossing + taper region
        crossing_size = 2.0 * self.taper_length_um + 2.0 * self.fit_margin_um
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = crossing_size
        self.dev_wy = crossing_size

        # Ports: same as uniform crossing
        port_span = max(self.wg_width_h, self.wg_width_v) + self.port_y_pad
        port_margin = self.dpml + 0.5
        nonpml_left = -half_x + self.dpml
        nonpml_right = half_x - self.dpml
        nonpml_bot = -half_y + self.dpml
        nonpml_top = half_y - self.dpml

        port_1_x = -half_x + port_margin
        self.port_1 = mp.Volume(
            center=mp.Vector3(port_1_x, 0, 0),
            size=mp.Vector3(0, port_span, 0),
        )
        self.src_vol_1 = mp.Volume(
            center=mp.Vector3(max(port_1_x - self.source_shift, nonpml_left + 0.1), 0, 0),
            size=self.port_1.size,
        )

        port_2_x = half_x - port_margin
        self.port_2 = mp.Volume(
            center=mp.Vector3(port_2_x, 0, 0),
            size=mp.Vector3(0, port_span, 0),
        )
        self.src_vol_2 = mp.Volume(
            center=mp.Vector3(min(port_2_x + self.source_shift, nonpml_right - 0.1), 0, 0),
            size=self.port_2.size,
        )

        port_3_y = -half_y + port_margin
        self.port_3 = mp.Volume(
            center=mp.Vector3(0, port_3_y, 0),
            size=mp.Vector3(port_span, 0, 0),
        )
        self.src_vol_3 = mp.Volume(
            center=mp.Vector3(0, max(port_3_y - self.source_shift, nonpml_bot + 0.1), 0),
            size=self.port_3.size,
        )

        port_4_y = half_y - port_margin
        self.port_4 = mp.Volume(
            center=mp.Vector3(0, port_4_y, 0),
            size=mp.Vector3(port_span, 0, 0),
        )
        self.src_vol_4 = mp.Volume(
            center=mp.Vector3(0, min(port_4_y + self.source_shift, nonpml_top - 0.1), 0),
            size=self.port_4.size,
        )

    def get_ports(self):
        """Override to provide all 4 ports."""
        return {
            "port_1": self.port_1,
            "port_2": self.port_2,
            "port_3": self.port_3,
            "port_4": self.port_4,
        }

    def get_sources(self):
        """Override to provide all 4 source volumes."""
        return {
            "src_1": self.src_vol_1,
            "src_2": self.src_vol_2,
            "src_3": self.src_vol_3,
            "src_4": self.src_vol_4,
        }

    def core_mask(self, nx, ny, dx, dy):
        """Generate mask for waveguide core (tapered crossing)."""
        x = (np.arange(nx) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")

        mask = np.zeros((ny, nx), dtype=np.uint8)

        # Horizontal waveguide with tapers
        # Left uniform section
        mask_h_left = (xx <= -self.taper_length_um) & (np.abs(yy) <= 0.5 * self.wg_width_h)
        mask = np.logical_or(mask, mask_h_left).astype(np.uint8)

        # Right uniform section
        mask_h_right = (xx >= self.taper_length_um) & (np.abs(yy) <= 0.5 * self.wg_width_h)
        mask = np.logical_or(mask, mask_h_right).astype(np.uint8)

        # Left taper: linear interpolation
        in_left_taper = (xx >= -self.taper_length_um) & (xx <= 0)
        t_left = (xx + self.taper_length_um) / self.taper_length_um
        t_left = np.clip(t_left, 0, 1)
        w_left = self.wg_width_h + t_left * (self.crossing_width - self.wg_width_h)
        mask_h_taper_left = in_left_taper & (np.abs(yy) <= 0.5 * w_left)
        mask = np.logical_or(mask, mask_h_taper_left).astype(np.uint8)

        # Right taper
        in_right_taper = (xx >= 0) & (xx <= self.taper_length_um)
        t_right = xx / self.taper_length_um
        t_right = np.clip(t_right, 0, 1)
        w_right = self.crossing_width + t_right * (self.wg_width_h - self.crossing_width)
        mask_h_taper_right = in_right_taper & (np.abs(yy) <= 0.5 * w_right)
        mask = np.logical_or(mask, mask_h_taper_right).astype(np.uint8)

        # Vertical waveguide with tapers
        # Bottom uniform section
        mask_v_bottom = (yy <= -self.taper_length_um) & (np.abs(xx) <= 0.5 * self.wg_width_v)
        mask = np.logical_or(mask, mask_v_bottom).astype(np.uint8)

        # Top uniform section
        mask_v_top = (yy >= self.taper_length_um) & (np.abs(xx) <= 0.5 * self.wg_width_v)
        mask = np.logical_or(mask, mask_v_top).astype(np.uint8)

        # Bottom taper
        in_bottom_taper = (yy >= -self.taper_length_um) & (yy <= 0)
        t_bottom = (yy + self.taper_length_um) / self.taper_length_um
        t_bottom = np.clip(t_bottom, 0, 1)
        w_bottom = self.wg_width_v + t_bottom * (self.crossing_width - self.wg_width_v)
        mask_v_taper_bottom = in_bottom_taper & (np.abs(xx) <= 0.5 * w_bottom)
        mask = np.logical_or(mask, mask_v_taper_bottom).astype(np.uint8)

        # Top taper
        in_top_taper = (yy >= 0) & (yy <= self.taper_length_um)
        t_top = yy / self.taper_length_um
        t_top = np.clip(t_top, 0, 1)
        w_top = self.crossing_width + t_top * (self.wg_width_v - self.crossing_width)
        mask_v_taper_top = in_top_taper & (np.abs(xx) <= 0.5 * w_top)
        mask = np.logical_or(mask, mask_v_taper_top).astype(np.uint8)

        return mask

    # Same _build_sim_single, _build_sim_broadband, run_sim, run_spectrum as UniformCrossing2D
    def _build_sim_single(self, input_port=1, df_frac=0.1):
        """Build single-frequency simulation."""
        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen

        if int(input_port) == 1:
            src_vol = self.src_vol_1
            k_vec = mp.Vector3(+1, 0, 0)
        elif int(input_port) == 2:
            src_vol = self.src_vol_2
            k_vec = mp.Vector3(-1, 0, 0)
        elif int(input_port) == 3:
            src_vol = self.src_vol_3
            k_vec = mp.Vector3(0, +1, 0)
        else:
            src_vol = self.src_vol_4
            k_vec = mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_vec,
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
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        return sim, (m1, m2, m3, m4), dft, fcen

    def _build_sim_broadband(self, input_port=1, lam_min_um=1.40, lam_max_um=1.60, Nf=101):
        """Build broadband simulation."""
        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs

        if int(input_port) == 1:
            src_vol = self.src_vol_1
            k_vec = mp.Vector3(+1, 0, 0)
        elif int(input_port) == 2:
            src_vol = self.src_vol_2
            k_vec = mp.Vector3(-1, 0, 0)
        elif int(input_port) == 3:
            src_vol = self.src_vol_3
            k_vec = mp.Vector3(0, +1, 0)
        else:
            src_vol = self.src_vol_4
            k_vec = mp.Vector3(0, -1, 0)

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df),
                volume=src_vol,
                eig_band=1,
                eig_parity=mp.NO_PARITY,
                eig_match_freq=True,
                eig_kpoint=k_vec,
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

        m1 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_1))
        m2 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_2))
        m3 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_3))
        m4 = sim.add_mode_monitor(fcen, df, Nf, mp.ModeRegion(volume=self.port_4))

        return sim, (m1, m2, m3, m4), (fcen, df, Nf, freqs, lams)

    def run_sim(self, input_port=1, decay_tol=1e-5, dir_plus=0, dir_minus=1):
        """Single-frequency S-parameters for 4-port device."""
        toward = {1: +1, 2: -1, 3: +1, 4: -1}

        sim, (m1, m2, m3, m4), dft, _ = self._build_sim_single(input_port=input_port)
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_mid = sim.get_epsilon().T.astype(np.float32)
        Ez_mid = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        alpha_3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        alpha_4 = get_mode_alpha_2dir(sim, m4, band=1, eig_parity=mp.NO_PARITY)

        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
        a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)
        a3_in, b3_out = pick_in_out_from_alpha(alpha_3, toward[3], dir_plus=dir_plus, dir_minus=dir_minus)
        a4_in, b4_out = pick_in_out_from_alpha(alpha_4, toward[4], dir_plus=dir_plus, dir_minus=dir_minus)

        if int(input_port) == 1:
            a_in = a1_in
            S = {
                'S11': b1_out / a_in,
                'S21': b2_out / a_in,
                'S31': b3_out / a_in,
                'S41': b4_out / a_in,
            }
        elif int(input_port) == 2:
            a_in = a2_in
            S = {
                'S12': b1_out / a_in,
                'S22': b2_out / a_in,
                'S32': b3_out / a_in,
                'S42': b4_out / a_in,
            }
        elif int(input_port) == 3:
            a_in = a3_in
            S = {
                'S13': b1_out / a_in,
                'S23': b2_out / a_in,
                'S33': b3_out / a_in,
                'S43': b4_out / a_in,
            }
        else:
            a_in = a4_in
            S = {
                'S14': b1_out / a_in,
                'S24': b2_out / a_in,
                'S34': b3_out / a_in,
                'S44': b4_out / a_in,
            }

        sim.reset_meep()

        for key, val in S.items():
            setattr(self, key, val)

        return eps_mid, Ez_mid, S, (self.cell_x, self.cell_y)

    def run_spectrum(
        self,
        input_port=1,
        lam_min_um=1.40,
        lam_max_um=1.60,
        Nf=101,
        decay_tol=1e-6,
        n_periods=50,
        dir_plus=0,
        dir_minus=1,
    ):
        """Broadband spectral response for 4-port device."""
        toward = {1: +1, 2: -1, 3: +1, 4: -1}

        sim, (m1, m2, m3, m4), (_, _, _, _, lams) = self._build_sim_broadband(
            input_port=input_port, lam_min_um=lam_min_um, lam_max_um=lam_max_um, Nf=Nf
        )

        decay_ports = {1: self.port_2, 2: self.port_1, 3: self.port_4, 4: self.port_3}
        stop = mp.stop_when_fields_decayed(int(n_periods), mp.Ez, decay_ports[input_port].center, float(decay_tol))
        sim.run(until_after_sources=stop)

        alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        alpha_3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        alpha_4 = get_mode_alpha_2dir(sim, m4, band=1, eig_parity=mp.NO_PARITY)

        a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
        a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)
        a3_in, b3_out = pick_in_out_from_alpha(alpha_3, toward[3], dir_plus=dir_plus, dir_minus=dir_minus)
        a4_in, b4_out = pick_in_out_from_alpha(alpha_4, toward[4], dir_plus=dir_plus, dir_minus=dir_minus)

        if int(input_port) == 1:
            a_in = a1_in
            S = {
                'S11': b1_out / a_in,
                'S21': b2_out / a_in,
                'S31': b3_out / a_in,
                'S41': b4_out / a_in,
            }
        elif int(input_port) == 2:
            a_in = a2_in
            S = {
                'S12': b1_out / a_in,
                'S22': b2_out / a_in,
                'S32': b3_out / a_in,
                'S42': b4_out / a_in,
            }
        elif int(input_port) == 3:
            a_in = a3_in
            S = {
                'S13': b1_out / a_in,
                'S23': b2_out / a_in,
                'S33': b3_out / a_in,
                'S43': b4_out / a_in,
            }
        else:
            a_in = a4_in
            S = {
                'S14': b1_out / a_in,
                'S24': b2_out / a_in,
                'S34': b3_out / a_in,
                'S44': b4_out / a_in,
            }

        sim.reset_meep()

        self.lams = lams
        for key, val in S.items():
            setattr(self, key + '_spec', val)

        return lams, S
