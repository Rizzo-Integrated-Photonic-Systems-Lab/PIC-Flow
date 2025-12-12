import meep as mp
import numpy as np
from utils import neff_siwire_from_tables
from devices_base import Device2DBase

class StraightWaveguide2D(Device2DBase):
    """
    2D effective-index straight waveguide, Tidy3D-style:

      - Core: index n_core, width wg_width_um (y), infinite in x (extends into PML)
      - Background: cladding index n_clad
      - PML on all sides
      - Domain length (wg_length_um) is distance between inner PML faces
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wg_length_um: float = 20.0,    # physical length between inner PML faces [µm]
        wavelength_um: float = 1.55,
        resolution: int = 40,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 1.0,
        pad_y_um: float = 2.0,
        port_margin_um: float = 2.0,   # distance from inner PML to ports [µm]
        source_shift_um: float = 0.5,  # distance source is upstream of input port [µm]
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
    ):
        # Use base defaults unless overrides provided
        cx = 25.0 if cell_x_um is None else cell_x_um
        cy = 4.0 if cell_y_um is None else cell_y_um
        super().__init__(cell_x_um=cx, cell_y_um=cy, dpml=dpml, resolution=resolution)

        self.wg_width_um = wg_width_um
        self.domain_length_um = wg_length_um
        self.wavelength_um = wavelength_um
        if n_core is None:
            # Look up geometry‑ and wavelength‑dependent n_eff from tables
            self.n_core = neff_siwire_from_tables(self.wg_width_um, self.wavelength_um)
        else:
            self.n_core = n_core
        self.n_clad = n_clad
        self.pad_y_um = pad_y_um
        self.port_margin_um = port_margin_um
        self.source_shift_um = source_shift_um

        # to be filled by build_geometry()
        self.geometry = None
        self.port_in = None
        self.port_out = None
        self.src_vol = None
        self.full_plane = None
        self.clad_medium = None

        # results
        self.eps_mid = None
        self.Ez_mid = None
        self.S21 = None
        self.a_in = None
        self.a_out = None

        self.build_geometry()

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        # straight waveguide extending through PML in x
        self.geometry = [
            mp.Block(
                size=mp.Vector3(mp.inf, self.wg_width_um, mp.inf),
                center=mp.Vector3(0, 0, 0),
                material=core,
            )
        ]

        # inner non-PML region: [-L/2, +L/2]
        L = self.domain_length_um
        x_left = -0.5 * L
        x_right = +0.5 * L

        # ports inside domain, offset from PML
        x_in = x_left + self.port_margin_um
        x_out = x_right - self.port_margin_um

        port_y_span = self.wg_width_um + 1.0
        port_size = mp.Vector3(0, port_y_span, 0)

        self.port_in = mp.Volume(center=mp.Vector3(x_in, 0), size=port_size)
        self.port_out = mp.Volume(center=mp.Vector3(x_out, 0), size=port_size)

        # source a bit upstream of the input monitor
        self.src_vol = mp.Volume(
            center=mp.Vector3(x_in - self.source_shift_um, 0),
            size=port_size,
        )

        # full xy slice for DFT fields
        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

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
