import meep as mp
from utils import plot_geometry_2d


class Device2DBase:
    """
    Base for 2D Meep devices: owns common cell sizing, PML, and resolution.
    Subclasses should call super().__init__ before building geometry.
    """

    def __init__(
        self,
        cell_x_um: float = 25.0,
        cell_y_um: float = 4.0,
        dpml: float = 1.0,
        resolution: int = 40,
    ):
        # Shared simulation extents and discretization
        self.cell_x = cell_x_um
        self.cell_y = cell_y_um
        self.cell = mp.Vector3(self.cell_x, self.cell_y, 0)
        self.dpml = dpml
        self.resolution = resolution

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
