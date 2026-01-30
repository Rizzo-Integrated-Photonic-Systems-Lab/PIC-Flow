import meep as mp
import numpy as np
from utils import neff_siwire_from_tables
from devices_base import Device2DBase

class StraightWaveguide2D(Device2DBase):
    def __init__(self,
                 wg_width=0.45,
                 wavelength=1.55,
                 dev_length_um=12.0,     
                 n_core=None,
                 n_clad=1.444,
                 port_y_pad=1.0,
                 source_shift=0.5,
                 fit_margin_um=0.5):
        super().__init__()  # inherits cell_x, cell_y, dpml, resolution defaults

        self.wg_width = wg_width
        self.wavelength = wavelength
        self.n_clad = n_clad
        self.source_shift = source_shift
        self.port_y_span = wg_width + port_y_pad

        self.fit_margin_um = fit_margin_um
        self.dev_length_um = float(dev_length_um)

        # clamp so it fits inside non-PML interior
        L_inner = self.cell_x - 2 * self.dpml
        max_len = L_inner - 2 * self.fit_margin_um
        self.dev_length_um = float(np.clip(self.dev_length_um, 1.0, max_len))

        # device window definition (conditioning/mask)
        self.dev_cx = 0.0
        self.dev_cy = 0.0
        self.dev_wx = self.dev_length_um
        self.dev_wy = self.wg_width + 2 * (port_y_pad + self.fit_margin_um)

        self.n_core = neff_siwire_from_tables(wg_width, wavelength) if n_core is None else n_core
        self.clad_medium = mp.Medium(index=self.n_clad)

        self.build_geometry()

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)

        # infinite waveguide geometry
        self.geometry = [mp.Block(size=mp.Vector3(mp.inf, self.wg_width, mp.inf),
                                  center=mp.Vector3(0, 0, 0),
                                  material=core)]

        # ports at device-window ends
        x_in = -0.5 * self.dev_length_um
        x_out = +0.5 * self.dev_length_um

        port_size = mp.Vector3(0, self.port_y_span, 0)
        self.port_in = mp.Volume(center=mp.Vector3(x_in, 0), size=port_size)
        self.port_out = mp.Volume(center=mp.Vector3(x_out, 0), size=port_size)

        self.src_vol = mp.Volume(center=mp.Vector3(x_in - self.source_shift, 0), size=port_size)


    def run_sim(self, decay_tol: float = 1e-3):
        """Run Meep, compute S21, and cache eps/Ez."""
        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=self.src_vol,
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

        m_in = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_in))
        m_out = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_out))

        dft_fields = sim.add_dft_fields(
            [mp.Ez],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        res_in = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=mp.NO_PARITY)
        res_out = sim.get_eigenmode_coefficients(m_out, [1], eig_parity=mp.NO_PARITY)

        a_in = res_in.alpha[0, 0, 0]
        a_out = res_out.alpha[0, 0, 0]

        self.a_in = a_in
        self.a_out = a_out

        self.S21 = a_out / a_in

        print("=== 2D straight waveguide S21 ===")
        print("S21 =", self.S21)
        print("|S21|^2 =", abs(self.S21) ** 2)
        print("a_in_ref =", a_in)
        print("|a_in_ref|^2 =", abs(a_in)**2)

        eps_2d = sim.get_epsilon()            # [nx, ny]
        self.eps_mid = eps_2d.T              # [ny, nx]

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0)  # [nx, ny]
        self.Ez_mid = Ez_mid.T

        # No design mask: return raw eps/Ez and cell size for downstream use
        return self.eps_mid, self.Ez_mid, self.S21, (self.cell_x, self.cell_y)
