# waveguide_crossing.py
import meep as mp
import numpy as np

from utils import neff_siwire_from_tables
from devices_base import (
    Device2DBase,
    DEFAULT_CELL_X_UM,
    DEFAULT_CELL_Y_UM,
    DEFAULT_DPML_UM,
    DEFAULT_RESOLUTION,
)


class WaveguideCrossing2D(Device2DBase):
    """
    Orthogonal waveguide crossing (effective-index 2D).

    Port numbering (looking down +x, y up):
        1: left   (west)  -> +x propagation
        2: right  (east)  -> +x propagation
        3: bottom (south) -> +y propagation
        4: top    (north) -> +y propagation

    Geometry:
      - Horizontal straight core along x at y=0
      - Vertical straight core along y at x=0
      - Both extend through the PML (mp.inf) for cleaner ports
      - Same default cell size as your coupler (32 × 6.4 µm) for UNet consistency
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,
        wavelength_um: float = 1.55,
        resolution: int | None = None,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 1.0,
        source_shift_um: float = 0.5,  # how far “upstream” of the input port the source plane is
        port_margin_um: float = 0.5,   # margin from non-PML boundary for monitor planes
        port_span_pad_um: float = 0.25, # extra span beyond wg width for mode monitors
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
    ):
        super().__init__(
            cell_x_um=DEFAULT_CELL_X_UM if cell_x_um is None else float(cell_x_um),
            cell_y_um=DEFAULT_CELL_Y_UM if cell_y_um is None else float(cell_y_um),
            dpml=DEFAULT_DPML_UM if dpml is None else dpml,
            resolution=DEFAULT_RESOLUTION if resolution is None else int(resolution),
        )

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.n_clad = float(n_clad)
        self.dpml = float(dpml)

        self.source_shift_um = float(source_shift_um)
        self.port_margin_um = float(port_margin_um)
        self.port_span_pad_um = float(port_span_pad_um)

        if n_core is None:
            self.n_core = neff_siwire_from_tables(self.wg_width_um, self.wavelength_um)
        else:
            self.n_core = float(n_core)

        # to be filled by build_geometry()
        self.geometry = None
        self.clad_medium = None

        # ports + sources
        self.port_1 = None
        self.port_2 = None
        self.port_3 = None
        self.port_4 = None

        self.src_1 = None  # excite from left
        self.src_3 = None  # excite from bottom

        self.full_plane = None

        # results
        self.eps_mid = None
        self.Ez_mid = None
        self.Hx_mid = None
        self.Hy_mid = None
        self.S = None

        self.build_geometry()

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        geometry = []

        # Two orthogonal cores crossing at (0,0)
        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, self.wg_width_um, mp.inf),  # horizontal
                center=mp.Vector3(0.0, 0.0, 0.0),
                material=core,
            )
        )
        geometry.append(
            mp.Block(
                size=mp.Vector3(self.wg_width_um, mp.inf, mp.inf),  # vertical
                center=mp.Vector3(0.0, 0.0, 0.0),
                material=core,
            )
        )

        self.geometry = geometry

        # Non-PML bounds
        half_x = 0.5 * self.cell_x
        half_y = 0.5 * self.cell_y
        x_pml_left = -half_x
        x_pml_right = +half_x
        y_pml_bot = -half_y
        y_pml_top = +half_y

        nonpml_left = x_pml_left + self.dpml
        nonpml_right = x_pml_right - self.dpml
        nonpml_bot = y_pml_bot + self.dpml
        nonpml_top = y_pml_top - self.dpml

        # Place monitor planes safely inside non-PML
        port_x_left = nonpml_left + self.port_margin_um
        port_x_right = nonpml_right - self.port_margin_um
        port_y_bot = nonpml_bot + self.port_margin_um
        port_y_top = nonpml_top - self.port_margin_um

        self.x_port_left_um = port_x_left
        self.x_port_right_um = port_x_right
        self.y_port_bot_um = port_y_bot
        self.y_port_top_um = port_y_top

        # Port cross-sections:
        # - x-propagating ports: a line at fixed x spanning y
        # - y-propagating ports: a line at fixed y spanning x
        span = self.wg_width_um + 2.0 * self.port_span_pad_um

        port_size_x = mp.Vector3(0, span, 0)  # for ports 1 & 2 (direction=mp.X)
        port_size_y = mp.Vector3(span, 0, 0)  # for ports 3 & 4 (direction=mp.Y)

        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, 0.0, 0.0), size=port_size_x)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_right, 0.0, 0.0), size=port_size_x)
        self.port_3 = mp.Volume(center=mp.Vector3(0.0, port_y_bot, 0.0), size=port_size_y)
        self.port_4 = mp.Volume(center=mp.Vector3(0.0, port_y_top, 0.0), size=port_size_y)

        # Source planes: upstream of input ports but still inside non-PML
        src_x_left = max(nonpml_left + 0.1, port_x_left - self.source_shift_um)
        src_y_bot = max(nonpml_bot + 0.1, port_y_bot - self.source_shift_um)

        self.src_1 = mp.Volume(center=mp.Vector3(src_x_left, 0.0, 0.0), size=port_size_x)
        self.src_3 = mp.Volume(center=mp.Vector3(0.0, src_y_bot, 0.0), size=port_size_y)

        # Full xy slice for DFT fields
        self.full_plane = mp.Volume(
            center=mp.Vector3(0.0, 0.0, 0.0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0.0),
        )

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        """
        Run Meep, excite either port 1 (left) or port 3 (bottom),
        compute S_{•,input_port} and cache eps/Ez/Hx/Hy.

        Assumes:
          - Single guided mode per arm (band 1)
        """
        if input_port not in (1, 3):
            raise ValueError("input_port must be 1 (left) or 3 (bottom).")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

        if input_port == 1:
            src_vol = self.src_1
            eig_kpoint = mp.Vector3(1, 0, 0)   # propagate +x
        else:
            src_vol = self.src_3
            eig_kpoint = mp.Vector3(0, 1, 0)   # propagate +y

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

        # Mode monitors on all 4 ports
        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=mp.X))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=mp.X))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3, direction=mp.Y))
        m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_4, direction=mp.Y))

        # DFT Ez/Hx/Hy on the whole cross-section
        dft_fields = sim.add_dft_fields(
            [mp.Ez, mp.Hx, mp.Hy],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        # eigenmode coefficients: alpha[..., 0] = forward, alpha[..., 1] = backward
        res1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.NO_PARITY)
        res2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.NO_PARITY)
        res3 = sim.get_eigenmode_coefficients(m3, [1], eig_parity=mp.NO_PARITY)
        res4 = sim.get_eigenmode_coefficients(m4, [1], eig_parity=mp.NO_PARITY)

        a1_fwd, a1_bwd = res1.alpha[0, 0, 0], res1.alpha[0, 0, 1]
        a2_fwd, a2_bwd = res2.alpha[0, 0, 0], res2.alpha[0, 0, 1]
        a3_fwd, a3_bwd = res3.alpha[0, 0, 0], res3.alpha[0, 0, 1]
        a4_fwd, a4_bwd = res4.alpha[0, 0, 0], res4.alpha[0, 0, 1]

        S = {}

        if input_port == 1:
            # incident from left: forward at port 1 is incoming (+x)
            a_in = a1_fwd

            # reflections / transmissions:
            # port 1 (left): outgoing to -x is backward at port 1
            S[(1, 1)] = a1_bwd / a_in

            # port 2 (right): outgoing to +x is forward at port 2
            S[(2, 1)] = a2_fwd / a_in

            # port 3 (bottom): outgoing to -y is backward at port 3 (since forward is +y)
            S[(3, 1)] = a3_bwd / a_in

            # port 4 (top): outgoing to +y is forward at port 4
            S[(4, 1)] = a4_fwd / a_in

        else:
            # incident from bottom: forward at port 3 is incoming (+y)
            a_in = a3_fwd

            # port 3 (bottom): outgoing to -y is backward at port 3
            S[(3, 3)] = a3_bwd / a_in

            # port 4 (top): outgoing to +y is forward at port 4
            S[(4, 3)] = a4_fwd / a_in

            # port 1 (left): outgoing to -x is backward at port 1
            S[(1, 3)] = a1_bwd / a_in

            # port 2 (right): outgoing to +x is forward at port 2
            S[(2, 3)] = a2_fwd / a_in

        self.S = S

        print(f"=== 2D waveguide crossing, excitation at port {input_port} ===")
        power_sum = 0.0
        outs = (1, 2, 3, 4)
        for p in outs:
            key = (p, input_port)
            if key not in S:
                continue
            s_val = S[key]
            p_val = abs(s_val) ** 2
            power_sum += p_val
            print(f"S{p}{input_port} = {s_val:.4g}, |S{p}{input_port}|^2 = {p_val:.4g}")
        print(f"Sum of guided powers Σ|S·{input_port}|^2 = {power_sum:.4g}")

        # store fields
        eps_2d = sim.get_epsilon()     # [nx, ny]
        self.eps_mid = eps_2d.T        # [ny, nx]

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0)  # [nx, ny]
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0)
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0)

        self.Ez_mid = Ez_mid.T
        self.Hx_mid = Hx_mid.T
        self.Hy_mid = Hy_mid.T

        return self.eps_mid, self.Ez_mid, self.Hx_mid, self.Hy_mid, S, (self.cell_x, self.cell_y)
