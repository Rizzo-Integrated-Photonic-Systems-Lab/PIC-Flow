# utils.py
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path


def plot_geometry_2d(
    cell_size: mp.Vector3,
    geometry,
    dpml: float,
    resolution: int,
    center: mp.Vector3 | None = None,
    size: mp.Vector3 | None = None,
    title: str = "Geometry (ε_r)",
    xlabel: str = "x (µm)",
    ylabel: str = "y (µm)",
):
    if center is None:
        center = mp.Vector3(0, 0, 0)

    if size is None:
        size = mp.Vector3(cell_size.x, cell_size.y, 0)

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        # let Meep's default medium be air, or pass explicitly if you want
        default_material=mp.Medium(index=1.0),
    )

    vol = mp.Volume(center=center, size=size)

    mp.plot2D(sim, output_plane=vol)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()



def plot_fields_2d(
    eps_mid,
    Ez_mid,
    Hx_mid=None,
    Hy_mid=None,
    cell_x_um: float | None = None,
    cell_y_um: float | None = None,
    resolution: int | None = None,
    x_left_um: float | None = None,   # optional: override x-range
    x_right_um: float | None = None,
    title_prefix: str = "2D device",
    abs_scale: str = "gamma",         # "linear", "log", or "gamma"
    abs_vmin_rel: float = 1e-2,       # for log/gamma: min as fraction of max
    show_eps_contour: bool = True,
    eps_contour_level: float | None = None,
):
    """
    Field plotter for 2D simulations with optional physical coords and
    optional geometry overlay.

    eps_mid : [ny, nx]   (ε_r)
    Ez_mid  : [ny, nx]   (complex Ez at some frequency)
    """
    ez_real = np.real(Ez_mid)
    ez_imag = np.imag(Ez_mid)
    ez_abs = np.abs(Ez_mid)
    ny, nx = eps_mid.shape

    # Build list of fields to plot
    entries = [
        ("Dielectric profile (ε_r)", eps_mid, None, "linear"),
        ("Re(Ez)", ez_real, "RdBu", "linear"),
        ("Im(Ez)", ez_imag, "RdBu", "linear"),
    ]

    # |Ez| uses magnitude handling later
    entries.append(("|Ez|", ez_abs, "magma", abs_scale))

    if Hx_mid is not None:
        entries.append(("Re(Hx)", np.real(Hx_mid), "RdBu", "linear"))
        entries.append(("Im(Hx)", np.imag(Hx_mid), "RdBu", "linear"))
    if Hy_mid is not None:
        entries.append(("Re(Hy)", np.real(Hy_mid), "RdBu", "linear"))
        entries.append(("Im(Hy)", np.imag(Hy_mid), "RdBu", "linear"))

    n_plots = len(entries)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.8 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else (axes,)

    # --- decide whether to use physical coordinates or indices ---
    if cell_x_um is None or cell_y_um is None:
        extent = None
        x_label = "x index"
        y_label = "y index"
    else:
        # use actual cell extents, optionally cropped
        x_min = -0.5 * cell_x_um if x_left_um is None else x_left_um
        x_max =  0.5 * cell_x_um if x_right_um is None else x_right_um
        y_min = -0.5 * cell_y_um
        y_max =  0.5 * cell_y_um
        extent = [x_min, x_max, y_min, y_max]
        x_label = "x (µm)"
        y_label = "y (µm)"

    for ax, (title, data, cmap, scale_mode) in zip(axes_flat, entries):
        if title == "|Ez|":
            ez_max = ez_abs.max() if ez_abs.size > 0 else 1.0
            if scale_mode == "log":
                vmin = max(abs_vmin_rel * ez_max, 1e-12)
                norm = colors.LogNorm(vmin=vmin, vmax=ez_max)
                plot_data = ez_abs
            elif scale_mode == "gamma":
                gamma = 0.25
                plot_data = (ez_abs / ez_max) ** gamma * ez_max
                norm = None
            else:
                plot_data = ez_abs
                norm = None
        else:
            plot_data = data
            norm = None

        im = ax.imshow(
            plot_data,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            extent=extent,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if title == "|Ez|" and show_eps_contour:
            if eps_contour_level is None:
                eps_min = float(eps_mid.min())
                eps_max = float(eps_mid.max())
                eps_contour_level = 0.5 * (eps_min + eps_max)
            ax.contour(
                eps_mid,
                levels=[eps_contour_level],
                colors="white",
                linewidths=0.7,
                alpha=0.7,
                extent=extent,
            )

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    plt.show()




NEFF_COEFFS_450x220 = np.array([
    -1.2345,
    3.4567,
    0.8910
])

LAMBDA_MIN_UM = 1.30
LAMBDA_MAX_UM = 1.80


def neff_siwire_450x220(lam_um: float) -> float:
    """Effective index of 450×220 nm Si wire fundamental TE mode vs λ.

    Kept for backward compatibility with older scripts that assume a fixed
    450×220 nm waveguide. New geometry‑dependent simulations should prefer
    `neff_siwire_from_tables`, which looks up n_eff(λ, width) from CSV tables.
    """
    lam_clamped = np.clip(lam_um, LAMBDA_MIN_UM, LAMBDA_MAX_UM)
    return float(np.polyval(NEFF_COEFFS_450x220, lam_clamped))


# ---------------------------------------------------------------------------
# Geometry‑ and wavelength‑dependent n_eff lookup using precomputed tables
# ---------------------------------------------------------------------------

_NEFF_TABLE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _load_neff_table_for_width(
    wg_width_um: float,
    tables_dir: str | Path = "neff_tables",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (and cache) the n_eff(λ) table for a given waveguide width.

    Parameters
    ----------
    wg_width_um : float
        Waveguide width in µm (e.g. 0.45).
    tables_dir : str or Path
        Directory containing CSV files named
        'neff_siwire_w{w_code}_t0220.csv', where w_code is an integer
        like 045 for 0.45 µm.

    Returns
    -------
    lambda_grid : np.ndarray
        Array of wavelengths (µm).
    neff_vals : np.ndarray
        Corresponding effective indices.
    """
    # Clamp width to table range [0.38, 0.60] and quantize to 0.01 µm grid
    width_clamped = float(np.clip(wg_width_um, 0.38, 0.60))
    width_rounded = float(np.round(width_clamped, 2))
    w_code = int(round(width_rounded * 100))  # 0.45 -> 45

    if w_code in _NEFF_TABLE_CACHE:
        return _NEFF_TABLE_CACHE[w_code]

    w_str = f"{w_code:03d}"  # 45 -> "045"

    base_dir = Path(tables_dir)
    if not base_dir.is_absolute():
        # Resolve relative to this file's directory
        base_dir = Path(__file__).resolve().parent / base_dir

    csv_path = base_dir / f"neff_siwire_w{w_str}_t0220.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"n_eff table not found for width {width_rounded:.2f} µm at {csv_path}")

    # CSV layout:
    #   lambda_um,neff
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        # Single row → expand to 2D
        data = data[None, :]

    lambda_grid = data[:, 0].astype(float)
    neff_vals = data[:, 1].astype(float)

    _NEFF_TABLE_CACHE[w_code] = (lambda_grid, neff_vals)
    return lambda_grid, neff_vals


def neff_siwire_from_tables(
    wg_width_um: float,
    lam_um: float,
    tables_dir: str | Path = "neff_tables",
) -> float:
    """
    Effective index n_eff(λ, width) from precomputed CSV tables.

    Parameters
    ----------
    wg_width_um : float
        Waveguide width in µm (e.g. between 0.38 and 0.60).
    lam_um : float
        Wavelength in µm (e.g. between 1.40 and 1.60).
    tables_dir : str or Path
        Directory containing the `neff_siwire_w###_t0220.csv` files.

    Returns
    -------
    float
        Interpolated n_eff at the requested (width, λ).
    """
    lambda_grid, neff_vals = _load_neff_table_for_width(wg_width_um, tables_dir=tables_dir)

    # 1D linear interpolation in λ; clamp to table range at the ends.
    lam_clamped = float(np.clip(lam_um, lambda_grid.min(), lambda_grid.max()))
    neff_interp = np.interp(lam_clamped, lambda_grid, neff_vals)
    return float(neff_interp)