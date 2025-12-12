import meep as mp
import numpy as np
from devices_base import Device2DBase
from utils import plot_geometry_xy, plot_geometry_yz


class DirectionalCoupler3D(Device2DBase):
    """
    3D symmetric 2×2 directional coupler with two parallel rectangular waveguides.

    Coordinates (looking down +x, y up, z out of plane):

        1: left,  top arm
        2: left,  bottom arm
        3: right, top arm
        4: right, bottom arm

    Geometry:
      - Two straight parallel cores of width `wg_width_um` (y) and height `wg_height_um` (z),
        separated by gap `gap_um` (edge–to–edge in y) in the coupling region.
      - Background cladding index n_clad (above and below).
      - PML on all sides with thickness `dpml`.
      - `wg_length_um` is the physical length between inner PML faces (coupling region).
      - Optional S-bends on input/output to widen lead spacing.
    """

    def __init__(
        self,
        wg_width_um: float = 0.45,       # width in y [µm]
        wg_height_um: float = 0.22,      # thickness in z [µm]
        gap_um: float = 0.2,             # edge–to–edge gap between guides [µm]
        wg_length_um: float = 20.0,      # length of central coupling region [µm]
        wavelength_um: float = 1.55,
        resolution: int = 30,
        n_core: float | None = None,     # core index; default ~Si at 1550 nm if None
        n_clad: float = 1.444,           # cladding index (e.g. SiO2)
        dpml: float = 1.0,
        pad_y_um: float = 2.0,           # cladding above top and below bottom guide [µm]
        pad_z_um: float = 1.0,           # cladding above and below waveguide in z [µm]
        source_shift_um: float = 0.5,    # distance source is upstream of input port [µm]
        lead_extra_gap_um: float = 1.0,  # extra separation between IO leads [µm]
        bend_length_um: float = 4.0,     # S-bend length [µm]
        bend_n_segments: int = 24,       # number of segments to approximate bend
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        cell_z_um: float | None = None,
    ):
        # Default lateral cell size if not provided
        cx = 25 if cell_x_um is None else cell_x_um
        cy = 8 if cell_y_um is None else cell_y_um

        # First call the 2D base to set up x–y and bookkeeping
        super().__init__(cell_x_um=cx, cell_y_um=cy, dpml=dpml, resolution=resolution)

        # Now upgrade to 3D: choose cell_z so that we have
        #   half_z = 0.5 * wg_height + pad_z + dpml
        # of space from center to each PML.
        if cell_z_um is None:
            cz = wg_height_um + 2 * pad_z_um + 2 * dpml
        else:
            cz = cell_z_um

        self.wg_width_um = wg_width_um
        self.wg_height_um = wg_height_um
        self.gap_um = gap_um
        self.domain_length_um = wg_length_um
        self.wavelength_um = wavelength_um

        if n_core is None:
            # crude default: silicon around 1.55 µm
            self.n_core = 3.48
        else:
            self.n_core = n_core

        self.n_clad = n_clad
        self.pad_y_um = pad_y_um
        self.pad_z_um = pad_z_um
        self.source_shift_um = source_shift_um

        self.coupling_length_um = wg_length_um

        self.lead_extra_gap_um = lead_extra_gap_um
        self.bend_length_um = bend_length_um
        self.bend_n_segments = bend_n_segments

        # override the cell to be 3D
        self.cell_z = cz
        self.cell = mp.Vector3(self.cell_x, self.cell_y, self.cell_z)

        # geometry & mediums
        self.geometry = None
        self.clad_medium = None

        # ports & sources
        self.port_1 = None
        self.port_2 = None
        self.port_3 = None
        self.port_4 = None

        self.src_1 = None
        self.src_2 = None

        self.full_plane = None  # here: actually a full 3D volume

        # x positions for plotting
        self.x_in_um = None
        self.x_out_um = None
        self.x_port_left_um = None
        self.x_port_right_um = None

        # results
        self.eps_mid = None   # mid-plane ε slice [ny, nx]
        self.Ez_mid = None    # mid-plane Ez slice [ny, nx]
        self.Hx_mid = None    # mid-plane Hx slice [ny, nx]
        self.Hy_mid = None    # mid-plane Hy slice [ny, nx]
        self.S = None         # dict keyed by (out_port, in_port)

        self.build_geometry()

    def build_geometry(self):
        core = mp.Medium(index=self.n_core)
        clad = mp.Medium(index=self.n_clad)
        self.clad_medium = clad

        # vertical placement in y (same as 2D)
        core_half_y = 0.5 * self.wg_width_um
        center_offset_y = 0.5 * (self.wg_width_um + self.gap_um)
        self.center_offset_y = center_offset_y

        # lead separation:
        lead_center_offset_y = center_offset_y + 0.5 * self.lead_extra_gap_um
        self.lead_center_offset_y = lead_center_offset_y

        # center in z at 0, with finite thickness
        core_half_z = 0.5 * self.wg_height_um

        geometry = []

        # inner non-PML region in x: [-L/2, +L/2]
        Lc = self.coupling_length_um
        x_in = -0.5 * Lc
        x_out = +0.5 * Lc

        self.x_in_um = x_in
        self.x_out_um = x_out

        # straight parallel guides in central region
        geometry.append(
            mp.Block(
                size=mp.Vector3(Lc, self.wg_width_um, self.wg_height_um),
                center=mp.Vector3(0.0, +center_offset_y, 0.0),
                material=core,
            )
        )
        geometry.append(
            mp.Block(
                size=mp.Vector3(Lc, self.wg_width_um, self.wg_height_um),
                center=mp.Vector3(0.0, -center_offset_y, 0.0),
                material=core,
            )
        )

        # S-bend helper, now with finite thickness in z
        def s_bend_blocks(
            y_start: float,
            y_end: float,
            x_start: float,
            x_end: float,
            n_seg: int,
        ):
            """
            Approximate a smooth cosine S-bend in the x–y plane and extrude it in z.

                y(u) = y_start + Δy * 0.5 * (1 - cos(π u)),  u ∈ [0, 1]
            """
            dx = (x_end - x_start) / n_seg
            dy = (y_end - y_start)
            L = x_end - x_start

            blocks = []
            for i in range(n_seg):
                u = (i + 0.5) / n_seg
                x0 = x_start + u * L
                y0 = y_start + 0.5 * dy * (1.0 - np.cos(np.pi * u))

                blocks.append(
                    mp.Block(
                        size=mp.Vector3(dx * 1.05, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x0, y0, 0.0),
                        material=core,
                    )
                )
            return blocks

        # cell boundaries in x
        half_cell_x = 0.5 * self.cell_x
        x_pml_left = -half_cell_x
        x_pml_right = half_cell_x

        # keep bends inside non-PML with margin
        x_left_lim = -half_cell_x + self.dpml + 0.5
        x_right_lim = half_cell_x - self.dpml - 0.5

        # clamp bend lengths
        L_bend = min(self.bend_length_um, x_in - x_left_lim, x_right_lim - x_out)

        port_x_left = x_in
        port_x_right = x_out
        port_y_offset = center_offset_y
        src_y_offset = center_offset_y

        if L_bend <= 0:
            # no S-bends: straight leads extending to PML
            if x_in > x_pml_left:
                lead_len_left = x_in - x_pml_left
                x_center_left = 0.5 * (x_pml_left + x_in)
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_left, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_left, +center_offset_y, 0.0),
                        material=core,
                    )
                )
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_left, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_left, -center_offset_y, 0.0),
                        material=core,
                    )
                )

            if x_out < x_pml_right:
                lead_len_right = x_pml_right - x_out
                x_center_right = 0.5 * (x_out + x_pml_right)
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_right, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_right, +center_offset_y, 0.0),
                        material=core,
                    )
                )
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_right, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_right, -center_offset_y, 0.0),
                        material=core,
                    )
                )

            self.geometry = geometry
        else:
            x_left_beg = x_in - L_bend
            x_left_end = x_in
            x_right_beg = x_out
            x_right_end = x_out + L_bend

            # left bends: wide leads -> tight coupling region
            geometry += s_bend_blocks(
                y_start=+lead_center_offset_y,
                y_end=+center_offset_y,
                x_start=x_left_beg,
                x_end=x_left_end,
                n_seg=self.bend_n_segments,
            )
            geometry += s_bend_blocks(
                y_start=-lead_center_offset_y,
                y_end=-center_offset_y,
                x_start=x_left_beg,
                x_end=x_left_end,
                n_seg=self.bend_n_segments,
            )

            # right bends: tight coupling region -> wide leads
            geometry += s_bend_blocks(
                y_start=+center_offset_y,
                y_end=+lead_center_offset_y,
                x_start=x_right_beg,
                x_end=x_right_end,
                n_seg=self.bend_n_segments,
            )
            geometry += s_bend_blocks(
                y_start=-center_offset_y,
                y_end=-lead_center_offset_y,
                x_start=x_right_beg,
                x_end=x_right_end,
                n_seg=self.bend_n_segments,
            )

            # straight leads that extend into the PML, matching the straight WG
            if x_left_beg > x_pml_left:
                lead_len_left = x_left_beg - x_pml_left
                x_center_left = 0.5 * (x_pml_left + x_left_beg)
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_left, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_left, +lead_center_offset_y, 0.0),
                        material=core,
                    )
                )
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_left, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_left, -lead_center_offset_y, 0.0),
                        material=core,
                    )
                )

            if x_right_end < x_pml_right:
                lead_len_right = x_pml_right - x_right_end
                x_center_right = 0.5 * (x_right_end + x_pml_right)
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_right, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_right, +lead_center_offset_y, 0.0),
                        material=core,
                    )
                )
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(lead_len_right, self.wg_width_um, self.wg_height_um),
                        center=mp.Vector3(x_center_right, -lead_center_offset_y, 0.0),
                        material=core,
                    )
                )

            self.geometry = geometry

            # Port positions: on straight leads, symmetric about x=0, inside non-PML
            half_cell_x = 0.5 * self.cell_x
            nonpml_left_start = -half_cell_x + self.dpml
            nonpml_right_end = +half_cell_x - self.dpml

            desired_margin = 0.5
            candidate_left = nonpml_left_start + desired_margin
            max_left_before_bend = x_left_beg - 0.2
            port_x_left = min(candidate_left, max_left_before_bend)

            # symmetric about x=0
            port_x_right = -port_x_left

            port_y_offset = lead_center_offset_y
            src_y_offset = lead_center_offset_y

        # record port x-locations
        self.x_port_left_um = port_x_left
        self.x_port_right_um = port_x_right

        # port cross-section: cover a single arm in y and full waveguide in z (+ some cladding)
        port_y_span = self.wg_width_um + 0.5 * self.gap_um
        port_z_span = self.wg_height_um + 2 * 0.5 * self.pad_z_um
        port_size = mp.Vector3(0, port_y_span, port_z_span)

        # source x: slightly upstream of left ports but inside non-PML region
        src_x_left = port_x_left - self.source_shift_um
        if L_bend > 0:
            half_cell_x = 0.5 * self.cell_x
            nonpml_left_start = -half_cell_x + self.dpml
            min_src_x = nonpml_left_start + 0.1
            if src_x_left < min_src_x:
                src_x_left = min_src_x

        # ports (volumes where eigenmode expansion is done)
        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, +port_y_offset, 0.0),
                                size=port_size)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_left, -port_y_offset, 0.0),
                                size=port_size)
        self.port_3 = mp.Volume(center=mp.Vector3(port_x_right, +port_y_offset, 0.0),
                                size=port_size)
        self.port_4 = mp.Volume(center=mp.Vector3(port_x_right, -port_y_offset, 0.0),
                                size=port_size)

        # sources slightly upstream of left ports
        self.src_1 = mp.Volume(
            center=mp.Vector3(src_x_left, +src_y_offset, 0.0),
            size=port_size,
        )
        self.src_2 = mp.Volume(
            center=mp.Vector3(src_x_left, -src_y_offset, 0.0),
            size=port_size,
        )

        # full 3D volume for DFT Ez fields (we'll slice later)
        self.full_plane = mp.Volume(
            center=mp.Vector3(0.0, 0.0, 0.0),
            size=mp.Vector3(self.cell_x, self.cell_y, self.cell_z),
        )

    def plot_xy(self, z_um: float = 0.0, title: str | None = None):
        """
        Plot an x–y slice of the 3D geometry at the specified z.
        """
        if title is None:
            title = f"3D coupler geometry (x–y @ z={z_um:g} µm)"

        plot_geometry_xy(
            cell_size=self.cell,
            geometry=self.geometry,
            dpml=self.dpml,
            resolution=self.resolution,
            z_um=z_um,
            title=title,
        )

    def plot_yz(self, x_um: float = 0.0, title: str | None = None):
        """
        Plot a y–z slice of the 3D geometry at the specified x.
        """
        if title is None:
            title = f"3D coupler geometry (y–z @ x={x_um:g} µm)"

        plot_geometry_yz(
            cell_size=self.cell,
            geometry=self.geometry,
            dpml=self.dpml,
            resolution=self.resolution,
            x_um=x_um,
            title=title,
        )

    def run_sim(self, input_port: int = 1, decay_tol: float = 1e-6):
        """
        Run 3D Meep, excite either port 1 or 2, compute S_{•,input_port} and cache eps/Ez mid-plane.

        Assumes:
          - Single guided mode per arm (band 1).
        """
        if input_port not in (1, 2):
            raise ValueError("input_port must be 1 (top left) or 2 (bottom left).")

        lam = self.wavelength_um
        fcen = 1.0 / lam
        df_source = 0.1 * fcen

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

        # Mode monitors on all four ports (propagation along +x)
        m1 = sim.add_mode_monitor(
            fcen, 0, 1, mp.ModeRegion(volume=self.port_1, direction=mp.X)
        )
        m2 = sim.add_mode_monitor(
            fcen, 0, 1, mp.ModeRegion(volume=self.port_2, direction=mp.X)
        )
        m3 = sim.add_mode_monitor(
            fcen, 0, 1, mp.ModeRegion(volume=self.port_3, direction=mp.X)
        )
        m4 = sim.add_mode_monitor(
            fcen, 0, 1, mp.ModeRegion(volume=self.port_4, direction=mp.X)
        )

        # 3D DFT Ez over the whole cell
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

        a1_fwd = res1.alpha[0, 0, 0]
        a1_bwd = res1.alpha[0, 0, 1]

        a2_fwd = res2.alpha[0, 0, 0]
        a2_bwd = res2.alpha[0, 0, 1]

        a3_fwd = res3.alpha[0, 0, 0]
        a3_bwd = res3.alpha[0, 0, 1]

        a4_fwd = res4.alpha[0, 0, 0]
        a4_bwd = res4.alpha[0, 0, 1]

        S = {}

        if input_port == 1:
            a_in = a1_fwd
        else:
            a_in = a2_fwd

        S[(1, input_port)] = a1_bwd / a_in   # reflection at port 1
        S[(2, input_port)] = a2_bwd / a_in   # reflection at port 2
        S[(3, input_port)] = a3_fwd / a_in   # transmission to port 3
        S[(4, input_port)] = a4_fwd / a_in   # transmission to port 4

        self.S = S

        print(f"=== 3D 2×2 coupler, excitation at port {input_port} ===")
        power_sum = 0.0
        for p in (1, 2, 3, 4):
            key = (p, input_port)
            s_val = S[key]
            p_val = abs(s_val) ** 2
            power_sum += p_val
            print(f"S{p}{input_port} = {s_val:.4g}, |S{p}{input_port}|^2 = {p_val:.4g}")
        print(f"Sum of guided powers Σ|S·{input_port}|^2 = {power_sum:.4g}")

        # store mid-plane ε and Ez (z=0 slice) for downstream use
        eps_3d = np.array(sim.get_epsilon())       # [nx, ny, nz]
        nx, ny, nz = eps_3d.shape
        k_mid = nz // 2
        eps_mid = eps_3d[:, :, k_mid]              # [nx, ny]
        self.eps_mid = eps_mid.T                   # [ny, nx]

        Ez_3d = np.array(sim.get_dft_array(dft_fields, mp.Ez, 0))  # [nx, ny, nz]
        Hx_3d = np.array(sim.get_dft_array(dft_fields, mp.Hx, 0))
        Hy_3d = np.array(sim.get_dft_array(dft_fields, mp.Hy, 0))

        Ez_mid = Ez_3d[:, :, k_mid]                # [nx, ny]
        Hx_mid = Hx_3d[:, :, k_mid]
        Hy_mid = Hy_3d[:, :, k_mid]

        self.Ez_mid = Ez_mid.T                     # [ny, nx]
        self.Hx_mid = Hx_mid.T
        self.Hy_mid = Hy_mid

        return (
            self.eps_mid,
            self.Ez_mid,
            self.Hx_mid,
            self.Hy_mid,
            S,
            (self.cell_x, self.cell_y, self.cell_z),
        )
