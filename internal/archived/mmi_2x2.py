# mmi_2x2.py
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


class MMI2x2PowerSplitter2D(Device2DBase):
    """
    Symmetric 2×2 MMI power splitter (effective-index 2D).

    Port numbering (looking down +x, y up):

        1: left  top input
        2: left  bottom input
        3: right top output
        4: right bottom output

    Geometry (modeled after common 2x2 MMI recipes):
      - Two left access guides at larger vertical offset (ports),
        cosine S-bends bring them inward to a small-gap pair at the MMI input face.
      - A rectangular MMI multimode section (width w_mmi, length l_mmi).
      - Cosine S-bends on the right fan the guides back outward to the output ports.
      - All waveguides + MMI are the same “core” medium; background is cladding medium.
      - Curves are approximated by many short Blocks (“Manhattan curve” along a smooth centerline).

    Notes on S-parameters in this 2D Meep setup:
      - For left ports, the *outgoing* wave is in the -x direction, so we use the
        backward eigenmode coefficient for S(1,·) and S(2,·).
      - For right ports, outgoing is +x, so we use the forward coefficient for S(3,·), S(4,·).
    """

    def __init__(
        self,
        # --- platform / sim ---
        wg_width_um: float = 0.40,
        wavelength_um: float = 1.55,
        resolution: int | None = None,
        n_core: float | None = None,
        n_clad: float = 1.444,
        dpml: float = 1.0,
        pad_y_um: float = 2.0,
        source_shift_um: float = 0.5,
        # --- MMI design ---
        w_mmi_um: float = 1.0,
        l_mmi_um: float = 3.0,
        gap_um: float = 0.20,             # gap between the two access guides at the MMI faces
        # --- access routing ---
        l_input_um: float = 2.0,
        l_output_um: float = 3.0,
        s_bend_offset_um: float = 1.0,    # how far ports are displaced outward vs MMI-face guides
        s_bend_length_um: float = 4.0,
        bend_n_segments: int = 120,
        # --- domain / ports ---
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        port_y_span_um: float | None = None,
    ):
        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.resolution = DEFAULT_RESOLUTION if resolution is None else int(resolution)

        if n_core is None:
            self.n_core = float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
        else:
            self.n_core = float(n_core)

        self.n_clad = float(n_clad)
        self.dpml = float(dpml)
        self.pad_y_um = float(pad_y_um)
        self.source_shift_um = float(source_shift_um)

        self.w_mmi_um = float(w_mmi_um)
        self.l_mmi_um = float(l_mmi_um)
        self.gap_um = float(gap_um)

        self.l_input_um = float(l_input_um)
        self.l_output_um = float(l_output_um)
        self.s_bend_offset_um = float(s_bend_offset_um)
        self.s_bend_length_um = float(s_bend_length_um)
        self.bend_n_segments = int(bend_n_segments)

        # Derived y-positions (ports are “far”, MMI-face guides are “near”)
        self.y_near = 0.5 * self.wg_width_um + 0.5 * self.gap_um
        self.y_far = self.y_near + self.s_bend_offset_um

        super().__init__(
            cell_x_um=DEFAULT_CELL_X_UM if cell_x_um is None else cell_x_um,
            cell_y_um=DEFAULT_CELL_Y_UM if cell_y_um is None else cell_y_um,
            dpml=DEFAULT_DPML_UM if dpml is None else dpml,
            resolution=self.resolution,
        )


        self.port_y_span_um = port_y_span_um

        # filled by build_geometry()
        self.geometry = None
        self.clad_medium = None

        self.port_1 = None
        self.port_2 = None
        self.port_3 = None
        self.port_4 = None

        self.src_1 = None
        self.src_2 = None
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

    @staticmethod
    def _cosine_interp(y0: float, y1: float, u: float) -> float:
        """
        Smooth (zero-slope endpoints) interpolation y(u), u in [0,1]:
            y = y0 + (y1-y0) * (1 - cos(pi u))/2
        """
        return y0 + (y1 - y0) * (0.5 - 0.5 * np.cos(np.pi * u))

    def _cosine_s_bend_blocks(self, x0: float, x1: float, y0: float, y1: float, w: float, n_seg: int, core: mp.Medium):
        """Approximate a cosine S-bend centerline with short Blocks."""
        L = x1 - x0
        dx = L / n_seg
        blocks = []
        for i in range(n_seg):
            u = (i + 0.5) / n_seg
            x = x0 + u * L
            y = self._cosine_interp(y0, y1, u)
            blocks.append(
                mp.Block(
                    size=mp.Vector3(dx * 1.05, w, mp.inf),
                    center=mp.Vector3(float(x), float(y), 0),
                    material=core,
                )
            )
        return blocks

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        geometry = []

        # non-PML x-range
        half_cell = 0.5 * self.cell_x
        x_pml_left = -half_cell
        x_pml_right = half_cell
        x_npml_left = x_pml_left + self.dpml
        x_npml_right = x_pml_right - self.dpml

        # Required device length (excluding PML)
        L_req = self.l_input_um + self.s_bend_length_um + self.l_mmi_um + self.s_bend_length_um + self.l_output_um
        margin = 0.6
        avail = (x_npml_right - margin) - (x_npml_left + margin)
        if L_req > avail:
            raise ValueError(
                f"MMI2x2PowerSplitter2D does not fit in cell_x={self.cell_x:.3g}um "
                f"(needs {L_req:.3g}um, has {avail:.3g}um). Increase cell_x_um or reduce lengths."
            )

        # Place device so right end has margin
        x0 = (x_npml_right - margin) - L_req

        # Key x landmarks
        x_in0 = x0
        x_in1 = x_in0 + self.l_input_um

        x_sb_in0 = x_in1
        x_sb_in1 = x_sb_in0 + self.s_bend_length_um

        x_mmi0 = x_sb_in1
        x_mmi1 = x_mmi0 + self.l_mmi_um

        x_sb_out0 = x_mmi1
        x_sb_out1 = x_sb_out0 + self.s_bend_length_um

        x_out0 = x_sb_out1
        x_out1 = x_out0 + self.l_output_um

        # --- Left straight access guides (ports are at y=±y_far) ---
        # Extend all the way into left PML for clean mode launching.
        if x_in1 > x_pml_left:
            stem_len = x_in1 - x_pml_left
            stem_center = 0.5 * (x_pml_left + x_in1)
            for y in (+self.y_far, -self.y_far):
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(stem_len, self.wg_width_um, mp.inf),
                        center=mp.Vector3(float(stem_center), float(y), 0),
                        material=core,
                    )
                )

        # --- Input cosine S-bends: y_far -> y_near over x_sb_in0..x_sb_in1 ---
        geometry += self._cosine_s_bend_blocks(x_sb_in0, x_sb_in1, +self.y_far, +self.y_near, self.wg_width_um, self.bend_n_segments, core)
        geometry += self._cosine_s_bend_blocks(x_sb_in0, x_sb_in1, -self.y_far, -self.y_near, self.wg_width_um, self.bend_n_segments, core)

        # --- Short straight “near” guides to ensure overlap at MMI face (optional but helps continuity) ---
        # Here we just add a tiny overlap block at the MMI face region.
        overlap = 0.10
        for y in (+self.y_near, -self.y_near):
            geometry.append(
                mp.Block(
                    size=mp.Vector3(overlap, self.wg_width_um, mp.inf),
                    center=mp.Vector3(float(x_mmi0 + 0.5 * overlap), float(y), 0),
                    material=core,
                )
            )

        # --- MMI multimode section ---
        geometry.append(
            mp.Block(
                size=mp.Vector3(self.l_mmi_um, self.w_mmi_um, mp.inf),
                center=mp.Vector3(float(0.5 * (x_mmi0 + x_mmi1)), 0.0, 0),
                material=core,
            )
        )

        # --- Output cosine S-bends: y_near -> y_far over x_sb_out0..x_sb_out1 ---
        geometry += self._cosine_s_bend_blocks(x_sb_out0, x_sb_out1, +self.y_near, +self.y_far, self.wg_width_um, self.bend_n_segments, core)
        geometry += self._cosine_s_bend_blocks(x_sb_out0, x_sb_out1, -self.y_near, -self.y_far, self.wg_width_um, self.bend_n_segments, core)

        # --- Right straight output guides into right PML ---
        if x_out0 < x_pml_right:
            out_len = x_pml_right - x_out0
            out_center = 0.5 * (x_out0 + x_pml_right)
            for y in (+self.y_far, -self.y_far):
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(out_len, self.wg_width_um, mp.inf),
                        center=mp.Vector3(float(out_center), float(y), 0),
                        material=core,
                    )
                )

        self.geometry = geometry

        # --- Ports (place in straight sections, away from bends) ---
        port_margin = 0.8
        # left: put it before input bend region
        port_x_left = x_npml_left + port_margin
        port_x_left = min(port_x_left, x_sb_in0 - 0.30)
        # right: put it after output bend region
        port_x_right = x_npml_right - port_margin
        port_x_right = max(port_x_right, x_sb_out1 + 0.30)

        # clamp just in case (avoid placing in PML)
        port_x_left = max(port_x_left, x_npml_left + 0.10)
        port_x_right = min(port_x_right, x_npml_right - 0.10)

        self.x_port_left_um = float(port_x_left)
        self.x_port_right_um = float(port_x_right)

        if self.port_y_span_um is None:
            port_y_span = self.wg_width_um + 0.40
        else:
            port_y_span = float(self.port_y_span_um)
        port_size = mp.Vector3(0, port_y_span, 0)

        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, +self.y_far, 0), size=port_size)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_left, -self.y_far, 0), size=port_size)
        self.port_3 = mp.Volume(center=mp.Vector3(port_x_right, +self.y_far, 0), size=port_size)
        self.port_4 = mp.Volume(center=mp.Vector3(port_x_right, -self.y_far, 0), size=port_size)

        # sources upstream of the left ports, still inside non-PML
        def make_src(port_center_y: float):
            src_x = port_x_left - self.source_shift_um
            min_src_x = x_npml_left + 0.10
            if src_x < min_src_x:
                src_x = min_src_x
            return mp.Volume(center=mp.Vector3(float(src_x), float(port_center_y), 0), size=port_size)

        self.src_1 = make_src(+self.y_far)
        self.src_2 = make_src(-self.y_far)

        # full-plane DFT
        self.full_plane = mp.Volume(center=mp.Vector3(0, 0), size=mp.Vector3(self.cell_x, self.cell_y, 0))

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6, fwidth_frac: float = 0.10):
        """
        Run Meep, excite one input (port 1 or 2), compute S_{•,in} and cache eps/Ez/Hx/Hy.
        """
        if input_port not in (1, 2):
            raise ValueError("MMI2x2PowerSplitter2D: input_port must be 1 (top-left) or 2 (bottom-left).")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = float(fwidth_frac) * fcen

        src_vol = self.src_1 if input_port == 1 else self.src_2

        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=df_source),
                volume=src_vol,
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

        # Mode monitors at all ports (direction=+x axis)
        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=mp.X))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=mp.X))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3, direction=mp.X))
        m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_4, direction=mp.X))

        dft_fields = sim.add_dft_fields(
            [mp.Ez, mp.Hx, mp.Hy],
            fcen,
            0,
            1,
            center=self.full_plane.center,
            size=self.full_plane.size,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=decay_tol))

        # Get eigenmode coefficients
        r1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.NO_PARITY)
        r2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.NO_PARITY)
        r3 = sim.get_eigenmode_coefficients(m3, [1], eig_parity=mp.NO_PARITY)
        r4 = sim.get_eigenmode_coefficients(m4, [1], eig_parity=mp.NO_PARITY)

        # alpha[..., 0] = forward (+x), alpha[..., 1] = backward (-x)
        a1_fwd, a1_bwd = r1.alpha[0, 0, 0], r1.alpha[0, 0, 1]
        a2_fwd, a2_bwd = r2.alpha[0, 0, 0], r2.alpha[0, 0, 1]
        a3_fwd, a3_bwd = r3.alpha[0, 0, 0], r3.alpha[0, 0, 1]
        a4_fwd, a4_bwd = r4.alpha[0, 0, 0], r4.alpha[0, 0, 1]

        a_in = a1_fwd if input_port == 1 else a2_fwd
        if abs(a_in) == 0:
            raise RuntimeError("Input mode amplitude is zero; check port placement / source placement / geometry.")

        # Assemble S(·,in)
        # Left ports: use backward as outgoing; Right ports: use forward as outgoing
        S = {}
        if input_port == 1:
            S[(1, 1)] = a1_bwd / a_in
            S[(2, 1)] = a2_bwd / a_in
            S[(3, 1)] = a3_fwd / a_in
            S[(4, 1)] = a4_fwd / a_in
        else:
            S[(1, 2)] = a1_bwd / a_in
            S[(2, 2)] = a2_bwd / a_in
            S[(3, 2)] = a3_fwd / a_in
            S[(4, 2)] = a4_fwd / a_in

        self.S = S

        print(f"=== 2D 2×2 MMI, excitation at port {input_port} ===")
        guided_power_sum = 0.0
        for p in (1, 2, 3, 4):
            key = (p, input_port)
            if key not in S:
                continue
            s_val = S[key]
            p_val = abs(s_val) ** 2
            guided_power_sum += p_val
            print(f"S{p}{input_port} = {s_val:.4g}, |S{p}{input_port}|^2 = {p_val:.4g}")
        print(f"Σ guided powers (measured at ports) = {guided_power_sum:.4g}")

        # Cache epsilon + fields (2D arrays)
        eps_2d = sim.get_epsilon()
        self.eps_mid = eps_2d.T

        Ez_mid = sim.get_dft_array(dft_fields, mp.Ez, 0)
        Hx_mid = sim.get_dft_array(dft_fields, mp.Hx, 0)
        Hy_mid = sim.get_dft_array(dft_fields, mp.Hy, 0)

        self.Ez_mid = Ez_mid.T
        self.Hx_mid = Hx_mid.T
        self.Hy_mid = Hy_mid.T

        return self.eps_mid, self.Ez_mid, self.Hx_mid, self.Hy_mid, S, (self.cell_x, self.cell_y)


