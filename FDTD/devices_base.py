import meep as mp
from utils import plot_geometry_2d

# Global defaults shared by all 2D devices
DEFAULT_CELL_X_UM = 32.0
DEFAULT_CELL_Y_UM = 8.0
DEFAULT_DPML_UM = 1.0
DEFAULT_RESOLUTION = 32


class Device2DBase:
    """
    Base for 2D Meep devices: owns common cell sizing, PML, and resolution.
    Subclasses should call super().__init__ before building geometry.
    """
    
    def __init__(
        self,
        cell_x_um: float | None = None,
        cell_y_um: float | None = None,
        dpml: float | None = None,
        resolution: int | None = None,
    ):
        # Shared simulation extents and discretization
        self.cell_x = DEFAULT_CELL_X_UM if cell_x_um is None else cell_x_um
        self.cell_y = DEFAULT_CELL_Y_UM if cell_y_um is None else cell_y_um
        self.cell = mp.Vector3(self.cell_x, self.cell_y, 0)
        self.dpml = DEFAULT_DPML_UM if dpml is None else dpml
        self.resolution = DEFAULT_RESOLUTION if resolution is None else resolution

        # Geometry is defined by subclasses
        self.geometry = None

    def plot_geometry(
        self,
        center: mp.Vector3 | None = None,
        size: mp.Vector3 | None = None,
        title: str = "Geometry (ε_r)",
        xlabel: str = "x (µm)",
        ylabel: str = "y (µm)",
    ):
        if center is None:
            center = mp.Vector3(0, 0)
        if size is None:
            size = mp.Vector3(self.cell_x, self.cell_y, 0)

        plot_geometry_2d(
            cell_size=self.cell,
            geometry=self.geometry,
            dpml=self.dpml,
            resolution=self.resolution,
            center=center,
            size=size,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
