# devices_base.py
import meep as mp
import numpy as np

from utils import plot_geometry_2d

DEFAULT_CELL_X_UM = 26.0  # 24.0 μm inner + 2*1.0 μm PML (square cell)
DEFAULT_CELL_Y_UM = 26.0  # 24.0 μm inner + 2*1.0 μm PML (square cell)
DEFAULT_DPML_UM = 1.0
DEFAULT_RESOLUTION = 14


class Device2DBase:
    """
    Base class for 2D Meep devices with:
      - Pixel-quantized dpml + interior sizes (prevents off-by-one crops)
      - Standardized device window (conditioning/mask only)
      - Common broadband spectral grid helpers
      - Optional standardized port/source accessors for generic plotting
      - Optional core_mask hook for device-agnostic overlays
    """

    def __init__(
        self,
        cell_x=None,
        cell_y=None,
        dpml=None,
        resolution=None,
        dev_wx=None,
        dev_wy=None,
        dev_cx=0.0,
        dev_cy=0.0,
    ):
        # -------------------------
        # Resolution and pixel size
        # -------------------------
        self.resolution = DEFAULT_RESOLUTION if resolution is None else int(resolution)
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        self.dx_um = 1.0 / float(self.resolution)

        # -------------------------
        # User-requested cell size
        # -------------------------
        cell_x = DEFAULT_CELL_X_UM if cell_x is None else float(cell_x)
        cell_y = DEFAULT_CELL_Y_UM if cell_y is None else float(cell_y)
        dpml = DEFAULT_DPML_UM if dpml is None else float(dpml)

        if cell_x <= 0 or cell_y <= 0:
            raise ValueError("cell_x and cell_y must be positive")
        if dpml < 0:
            raise ValueError("dpml must be non-negative")

        # -------------------------
        # Quantize dpml to integer pixels
        # -------------------------
        self.pml_px = int(round(dpml * self.resolution))
        self.dpml = float(self.pml_px) / float(self.resolution)

        # -------------------------
        # Quantize inner (non-PML) region to integer pixels
        # -------------------------
        inner_x_req = cell_x - 2.0 * self.dpml
        inner_y_req = cell_y - 2.0 * self.dpml
        if inner_x_req <= 0 or inner_y_req <= 0:
            raise ValueError("cell too small relative to dpml after quantization")

        inner_x_px = int(round(inner_x_req * self.resolution))
        inner_y_px = int(round(inner_y_req * self.resolution))
        inner_x_px = max(inner_x_px, 1)
        inner_y_px = max(inner_y_px, 1)

        self.inner_x = float(inner_x_px) / float(self.resolution)
        self.inner_y = float(inner_y_px) / float(self.resolution)

        # Recompute total cell size from quantized inner + dpml
        self.cell_x = self.inner_x + 2.0 * self.dpml
        self.cell_y = self.inner_y + 2.0 * self.dpml
        self.cell = mp.Vector3(self.cell_x, self.cell_y, 0)

        # Grid sizes (full including PML)
        self.nx = int(round(self.cell_x * self.resolution))
        self.ny = int(round(self.cell_y * self.resolution))

        # -------------------------
        # Device window (conditioning/mask only)
        # -------------------------
        self.dev_cx = float(dev_cx)
        self.dev_cy = float(dev_cy)
        self.dev_wx = self.cell_x if dev_wx is None else float(dev_wx)
        self.dev_wy = self.cell_y if dev_wy is None else float(dev_wy)

        if self.dev_wx <= 0 or self.dev_wy <= 0:
            raise ValueError("dev_wx and dev_wy must be positive")

        # Core geometry list (Meep primitives) set by subclass
        self.geometry = None

        # Optional standardized members that subclasses may set
        # - ports: dict[str, mp.Volume]
        # - sources: dict[str, mp.Volume]
        self.ports = {}
        self.sources = {}

    # -------------------------
    # Standard windows / volumes
    # -------------------------
    def device_window(self):
        """Returns (center, size) mp.Vector3 for the conditioning/device window."""
        return mp.Vector3(self.dev_cx, self.dev_cy, 0), mp.Vector3(self.dev_wx, self.dev_wy, 0)

    def pml_thickness(self):
        return float(self.dpml)

    def inner_window(self):
        """Returns non-PML interior window (center, size)."""
        return mp.Vector3(0.0, 0.0, 0.0), mp.Vector3(self.inner_x, self.inner_y, 0.0)

    def full_window(self):
        """Returns full cell window (center, size)."""
        return mp.Vector3(0.0, 0.0, 0.0), mp.Vector3(self.cell_x, self.cell_y, 0.0)

    # -------------------------
    # Plotting convenience
    # -------------------------
    def plot_geometry(self, center=None, size=None, title="Geometry (ε_r)"):
        if center is None or size is None:
            center, size = self.device_window()
        plot_geometry_2d(
            cell_size=self.cell,
            geometry=self.geometry,
            dpml=self.dpml,
            resolution=self.resolution,
            center=center,
            size=size,
            title=title,
            xlabel="x (µm)",
            ylabel="y (µm)",
        )

    # -------------------------
    # Port/source standardized access
    # -------------------------
    def get_ports(self):
        """
        Returns dict[str, mp.Volume].
        Subclasses can either:
          - populate self.ports, OR
          - set attributes like port_in/port_out and override this.
        """
        # If subclass uses self.ports, prefer it.
        if isinstance(self.ports, dict) and len(self.ports) > 0:
            return self.ports

        # Fallback: common attribute names
        out = {}
        for k in ["port_in", "port_out", "port_1", "port_2", "port_3", "port_4"]:
            v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out

    def get_sources(self):
        """
        Returns dict[str, mp.Volume].
        Subclasses can either:
          - populate self.sources, OR
          - set attribute src_vol and override this.
        """
        if isinstance(self.sources, dict) and len(self.sources) > 0:
            return self.sources

        out = {}
        v = getattr(self, "src_vol", None)
        if v is not None:
            out["src"] = v
        return out

    # -------------------------
    # Optional hook for overlays
    # -------------------------
    def core_mask(self, nx, ny, dx, dy):
        """
        Optional: subclasses can override to return a uint8 mask of "core" regions.
        Default returns None.

        Convention:
          - coordinates are physical (µm) with origin at cell center
          - output array shape should be (ny, nx) to match imshow(origin="lower")
        """
        return None

    # -------------------------
    # Spectral grid helpers
    # -------------------------
    @staticmethod
    def spectrum_from_lams(lam_min_um, lam_max_um, Nf=101):
        """
        Returns: (fcen, df, Nf, freqs, lams)
        Meep uses frequency f = 1 / wavelength (in µm^-1 when wavelength is in µm).
        """
        lam_min_um = float(lam_min_um)
        lam_max_um = float(lam_max_um)
        Nf = int(Nf)

        if lam_min_um <= 0 or lam_max_um <= 0:
            raise ValueError("lam_min_um and lam_max_um must be positive")
        if lam_min_um >= lam_max_um:
            raise ValueError("lam_min_um must be < lam_max_um")
        if Nf < 1:
            raise ValueError("Nf must be >= 1")

        fmin = 1.0 / lam_max_um
        fmax = 1.0 / lam_min_um
        fcen = 0.5 * (fmin + fmax)
        df = (fmax - fmin)

        freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, Nf)
        lams = 1.0 / freqs
        return fcen, df, Nf, freqs, lams

    @staticmethod
    def stop_condition_broadband(tol=1e-6, n_periods=50, at=None, component=mp.Ez):
        """
        Standard Meep stop condition for broadband pulse runs.
        """
        if at is None:
            at = mp.Vector3(0, 0, 0)
        return mp.stop_when_fields_decayed(int(n_periods), component, at, float(tol))

    # -------------------------
    # Optional: backward-compatible eigenmode alpha helper
    # -------------------------
    @staticmethod
    def mode_alpha(sim, monitor, band=1, eig_parity=mp.NO_PARITY):
        """
        Backward-compatible extraction: returns alpha[:, 0, :] for (nfreq, nbands=1, ndir=2).
        Prefer utils.get_mode_alpha_2dir for shape normalization across Meep variants.
        """
        res = sim.get_eigenmode_coefficients(monitor, [int(band)], eig_parity=eig_parity)
        a = res.alpha
        if a.ndim != 3:
            raise RuntimeError(f"Unexpected alpha shape {a.shape}, expected 3D array")
        return a[:, 0, :]
