# utils.py
import os
import json
from pathlib import Path

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# =============================================================================
# Plotting (device-agnostic)
# =============================================================================
def plot_geometry_2d(cell_size, geometry, dpml, resolution,
                     center=None, size=None,
                     title="Geometry (ε_r)",
                     xlabel="x (µm)", ylabel="y (µm)"):
    if center is None:
        center = mp.Vector3(0, 0, 0)
    if size is None:
        size = mp.Vector3(cell_size.x, cell_size.y, 0)

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=int(resolution),
        boundary_layers=[mp.PML(float(dpml))],
        geometry=geometry,
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


def plot_fields_2d(eps_mid, Ez_mid,
                   Hx_mid=None, Hy_mid=None,
                   cell_x_um=None, cell_y_um=None,
                   x_left_um=None, x_right_um=None,
                   title_prefix="2D device",
                   abs_scale="gamma",
                   abs_vmin_rel=1e-2,
                   show_eps_contour=True,
                   eps_contour_level=None):
    ez_real = np.real(Ez_mid)
    ez_imag = np.imag(Ez_mid)
    ez_abs = np.abs(Ez_mid)

    entries = [
        ("Dielectric profile (ε_r)", eps_mid, None, "linear"),
        ("Re(Ez)", ez_real, "RdBu", "linear"),
        ("Im(Ez)", ez_imag, "RdBu", "linear"),
        ("|Ez|", ez_abs, "magma", abs_scale),
    ]

    if Hx_mid is not None:
        entries += [("Re(Hx)", np.real(Hx_mid), "RdBu", "linear"),
                    ("Im(Hx)", np.imag(Hx_mid), "RdBu", "linear")]
    if Hy_mid is not None:
        entries += [("Re(Hy)", np.real(Hy_mid), "RdBu", "linear"),
                    ("Im(Hy)", np.imag(Hy_mid), "RdBu", "linear")]

    n_plots = len(entries)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.8 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else (axes,)

    if cell_x_um is None or cell_y_um is None:
        extent = None
        x_label, y_label = "x index", "y index"
    else:
        x_min = -0.5 * float(cell_x_um) if x_left_um is None else float(x_left_um)
        x_max = +0.5 * float(cell_x_um) if x_right_um is None else float(x_right_um)
        y_min = -0.5 * float(cell_y_um)
        y_max = +0.5 * float(cell_y_um)
        extent = [x_min, x_max, y_min, y_max]
        x_label, y_label = "x (µm)", "y (µm)"

    for ax, (title, data, cmap, scale_mode) in zip(axes_flat, entries):
        if title == "|Ez|":
            ez_max = float(ez_abs.max()) if ez_abs.size else 1.0
            if scale_mode == "log":
                vmin = max(float(abs_vmin_rel) * ez_max, 1e-12)
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

        im = ax.imshow(plot_data, origin="lower", aspect="equal",
                       cmap=cmap, extent=extent, norm=norm)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if title == "|Ez|" and show_eps_contour:
            lvl = eps_contour_level
            if lvl is None:
                eps_min = float(np.min(eps_mid))
                eps_max = float(np.max(eps_mid))
                lvl = 0.5 * (eps_min + eps_max)
            ax.contour(eps_mid, levels=[lvl], colors="white",
                       linewidths=0.7, alpha=0.7, extent=extent)

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    plt.show()


# =============================================================================
# n_eff lookup (device-agnostic)
# =============================================================================
NEFF_COEFFS_450x220 = np.array([-1.2345, 3.4567, 0.8910], dtype=float)
LAMBDA_MIN_UM = 1.30
LAMBDA_MAX_UM = 1.80


def neff_siwire_450x220(lam_um):
    lam = float(np.clip(float(lam_um), LAMBDA_MIN_UM, LAMBDA_MAX_UM))
    return float(np.polyval(NEFF_COEFFS_450x220, lam))


_NEFF_TABLE_CACHE = {}  # w_code -> (lambda_grid, neff_vals)


def _load_neff_table_for_width(wg_width_um, tables_dir="neff_tables"):
    width = float(np.clip(float(wg_width_um), 0.38, 0.60))
    width = float(np.round(width, 2))
    w_code = int(round(width * 100))  # 0.45 -> 45

    if w_code in _NEFF_TABLE_CACHE:
        return _NEFF_TABLE_CACHE[w_code]

    w_str = f"{w_code:03d}"
    base_dir = Path(tables_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).resolve().parent / base_dir

    csv_path = base_dir / f"neff_siwire_w{w_str}_t0220.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"n_eff table not found for width {width:.2f} µm at {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    lambda_grid = data[:, 0].astype(float)
    neff_vals = data[:, 1].astype(float)

    _NEFF_TABLE_CACHE[w_code] = (lambda_grid, neff_vals)
    return lambda_grid, neff_vals


def neff_siwire_from_tables(wg_width_um, lam_um, tables_dir="neff_tables"):
    lambda_grid, neff_vals = _load_neff_table_for_width(wg_width_um, tables_dir=tables_dir)
    lam = float(np.clip(float(lam_um), float(lambda_grid.min()), float(lambda_grid.max())))
    return float(np.interp(lam, lambda_grid, neff_vals))


# =============================================================================
# Dataset / shard utilities (device-agnostic)
# =============================================================================
def atomic_write_json(path, obj, indent=2):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def write_shard_npz(out_path, samples, compress=True):
    """
    samples: list of (arrays_dict, meta_dict)
      - arrays_dict values must be np.ndarray (or np scalar arrays)
      - meta_dict values will be wrapped with np.array(...)
    """
    out_path = Path(out_path)
    save_dict = {}
    for i, (arrays, meta) in enumerate(samples):
        prefix = f"s{i}/"
        for k, v in arrays.items():
            save_dict[prefix + k] = v
        for k, v in meta.items():
            save_dict[prefix + k] = np.array(v)

    if compress:
        np.savez_compressed(out_path, **save_dict)
    else:
        np.savez(out_path, **save_dict)


def decode_npz_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        try:
            return decode_npz_str(x.item())
        except Exception:
            return str(x)
    return str(x)


def safe_rm(path):
    try:
        path = Path(path)
        if path.exists():
            path.unlink()
    except Exception:
        pass


def quantize_square_cell_from_crop(crop_px, resolution, dpml_um):
    """
    Euler-style quantization:
      pml_px = round(dpml_um * resolution)
      dpml_um_q = pml_px / resolution
      full_px = crop_px + 2*pml_px
      cell_um = full_px / resolution   (square)
    Returns: (dpml_um_q, pml_px, cell_um, full_px)
    """
    crop_px = int(crop_px)
    resolution = int(resolution)
    pml_px = int(np.round(float(dpml_um) * float(resolution)))
    dpml_um_q = float(pml_px) / float(resolution)
    full_px = int(crop_px + 2 * pml_px)
    cell_um = float(full_px) / float(resolution)
    return dpml_um_q, pml_px, cell_um, full_px


def validate_crop_square(cell_um, resolution, pml_px, crop_px):
    nx_full = int(np.round(float(cell_um) * float(resolution)))
    nx_crop = nx_full - 2 * int(pml_px)
    if nx_crop != int(crop_px):
        raise ValueError(
            f"Crop mismatch: expected {int(crop_px)} but got {nx_crop}. "
            f"Check dpml/resolution/crop-px."
        )


def shard_writer_generic(q,
                         shards_root,
                         shard_size,
                         compress,
                         dataset_name,
                         decode_tmp_npz,
                         index_name="index.json",
                         save_index_every_shard=True):
    """
    Generic streaming shard writer.

    Queue protocol:
      q.put(("OK", tmp_npz_path_str))
      q.put(("ERR", err_string))  # ignored here
      q.put(None)                # flush and exit

    decode_tmp_npz(tmp_path: Path) -> list of (arrays_dict, meta_dict)
      - meta_dict must include 'tag'
      - if meta_dict lacks 'dataset', it will be set to dataset_name
    """
    shards_root = Path(shards_root)
    shards_root.mkdir(parents=True, exist_ok=True)

    index = []
    index_path = shards_root / index_name

    buffer = []
    shard_id = 0

    def write_one_shard(batch):
        nonlocal shard_id, index
        shard_name = f"shard_{shard_id:05d}.npz"
        shard_path = shards_root / shard_name

        write_shard_npz(shard_path, batch, compress=compress)

        for slot, (_, meta) in enumerate(batch):
            index.append({"tag": str(meta["tag"]), "shard": shard_name, "slot": int(slot)})

        shard_id += 1
        if save_index_every_shard:
            atomic_write_json(index_path, index)

    while True:
        msg = q.get()
        if msg is None:
            break

        status, payload = msg
        if status != "OK":
            continue

        tmp_path = Path(payload)
        try:
            samples = decode_tmp_npz(tmp_path)
            for arrays, meta in samples:
                if "tag" not in meta:
                    raise RuntimeError("decode_tmp_npz must set meta['tag']")
                if "dataset" not in meta:
                    meta["dataset"] = dataset_name
                buffer.append((arrays, meta))

            while len(buffer) >= int(shard_size):
                batch = buffer[:int(shard_size)]
                buffer = buffer[int(shard_size):]
                write_one_shard(batch)

        finally:
            safe_rm(tmp_path)

    if buffer:
        write_one_shard(buffer)

    if not save_index_every_shard:
        atomic_write_json(index_path, index)

# =============================================================================
# Eigenmode S-parameter utilities (device-agnostic)
# =============================================================================

def pick_in_out_from_alpha(alpha_dir, toward_device, dir_plus=0, dir_minus=1):
    """
    alpha_dir can be:
      - shape (2,)             for single-frequency (ndir=2)
      - shape (Nf, 2)          for broadband (Nf freqs, ndir=2)

    toward_device:
      +1 means "incoming toward device is +x"
      -1 means "incoming toward device is -x"
    """
    a = np.asarray(alpha_dir)

    if a.ndim == 1:
        # (2,)
        if int(toward_device) == +1:
            a_in  = a[dir_plus]
            b_out = a[dir_minus]
        else:
            a_in  = a[dir_minus]
            b_out = a[dir_plus]
        return a_in, b_out

    if a.ndim == 2:
        # (Nf, 2) -> index LAST axis for direction
        if int(toward_device) == +1:
            a_in  = a[:, dir_plus]
            b_out = a[:, dir_minus]
        else:
            a_in  = a[:, dir_minus]
            b_out = a[:, dir_plus]
        return a_in, b_out

    raise ValueError(f"alpha_dir must be 1D or 2D, got shape {a.shape}")



def get_mode_alpha_2dir(sim, mode_monitor, band=1, eig_parity=mp.NO_PARITY):
    res = sim.get_eigenmode_coefficients(mode_monitor, [int(band)], eig_parity=eig_parity)
    a = res.alpha
    # Meep returns alpha with different shapes:
    #   - Single freq: (nfreq, nbands, 2) -> (1, 1, 2)
    #   - Broadband: (nbands, nfreq, 2) -> (1, nfreq, 2) for our case
    # We return:
    #   - (2,) for single-frequency monitors
    #   - (nfreq, 2) for broadband monitors
    if a.ndim != 3:
        raise RuntimeError(f"Unexpected alpha shape {a.shape}, expected 3D array")

    if a.shape[0] == 1 and a.shape[1] > 1:
        # Broadband case: (1, nfreq, 2) -> extract (nfreq, 2)
        alpha = a[0, :, :]  # Shape: (nfreq, 2)
    else:
        # Single freq or other cases: (nfreq, nbands, 2) -> extract (nfreq, 2)
        alpha = a[:, 0, :]  # Shape: (nfreq, 2)

    nfreq = int(alpha.shape[0])
    # Return (2,) for single freq, (nfreq, 2) for broadband
    return alpha[0, :] if nfreq == 1 else alpha


# =============================================================================
# Mask helpers (device-agnostic) + overlay plotting
# =============================================================================
def rect_mask(nx, ny, dx, dy, cx, cy, wx, wy):
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return ((np.abs(xx - cx) <= 0.5 * wx) & (np.abs(yy - cy) <= 0.5 * wy)).astype(np.uint8)

def vol_mask(nx, ny, dx, dy, vol, slab_px=2):
    cx, cy = float(vol.center.x), float(vol.center.y)
    sx, sy = float(vol.size.x), float(vol.size.y)
    if sx == 0:
        sx = slab_px * dx
    if sy == 0:
        sy = slab_px * dy
    return rect_mask(nx, ny, dx, dy, cx, cy, sx, sy)

def pml_mask(nx, ny, dx, dy, cell_x, cell_y, dpml):
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return ((np.abs(xx) >= cell_x / 2.0 - dpml) | (np.abs(yy) >= cell_y / 2.0 - dpml)).astype(np.uint8)

def mask_cmap(rgba):
    return ListedColormap([(0.0, 0.0, 0.0, 0.0), rgba])

DEFAULT_OVERLAY_COLORS = {
    "core": (0.10, 0.10, 0.10, 1.00),
    "dev": (1.00, 0.70, 0.00, 0.22),
    "pml": (0.00, 0.00, 0.00, 0.08),
    "src": (0.10, 0.75, 0.20, 0.95),
    "port_in": (0.10, 0.45, 0.95, 0.95),
    "port_out": (0.95, 0.20, 0.20, 0.95),
    "port_1": (0.20, 0.60, 0.95, 0.95),
    "port_2": (0.20, 0.80, 0.55, 0.95),
    "port_3": (0.95, 0.45, 0.20, 0.95),
    "port_4": (0.85, 0.20, 0.60, 0.95),
    "src_1": (0.10, 0.85, 0.20, 0.95),
    "src_2": (0.10, 0.75, 0.35, 0.95),
    "src_3": (0.25, 0.75, 0.20, 0.95),
    "src_4": (0.15, 0.65, 0.30, 0.95),
}

DEFAULT_DRAW_ORDER = [
    "core",
    "dev",
    "pml",
    "port_in",
    "port_out",
    "port_1",
    "port_2",
    "port_3",
    "port_4",
    "src",
    "src_1",
    "src_2",
    "src_3",
    "src_4",
]

DEFAULT_LEGEND_LABELS = {
    "core": "Waveguide/core",
    "dev": "Device window",
    "pml": "PML region",
    "src": "Source",
    "port_in": "Input port",
    "port_out": "Output port",
    "port_1": "Port 1",
    "port_2": "Port 2",
    "port_3": "Port 3",
    "port_4": "Port 4",
    "src_1": "Source 1",
    "src_2": "Source 2",
    "src_3": "Source 3",
    "src_4": "Source 4",
}


def legend_handles_from_masks(masks, colors=None, labels=None, draw_order=None):
    """
    Create legend handles for overlay masks.

    Args:
        masks: dict of mask names to mask arrays
        colors: dict mapping mask names to RGBA colors (default: DEFAULT_OVERLAY_COLORS)
        labels: dict mapping mask names to legend labels (default: DEFAULT_LEGEND_LABELS)

    Returns:
        List of matplotlib Patch objects for legend
    """
    if colors is None:
        colors = DEFAULT_OVERLAY_COLORS
    if labels is None:
        labels = DEFAULT_LEGEND_LABELS

    order = DEFAULT_DRAW_ORDER if draw_order is None else draw_order
    handles = []
    for k in order:
        if k in masks:
            handles.append(mpatches.Patch(color=colors[k], label=labels.get(k, k)))
    return handles

def plot_overlay_masks(ax, masks, colors=DEFAULT_OVERLAY_COLORS, draw_order=DEFAULT_DRAW_ORDER,
                       title=None, show_legend=False):
    ax.set_facecolor((1,1,1,1))
    for name in draw_order:
        if name not in masks:
            continue
        ax.imshow(
            masks[name],
            origin="lower",
            aspect="equal",
            cmap=mask_cmap(colors[name]),
            vmin=0, vmax=1,
            interpolation="nearest",
        )
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    if show_legend:
        handles = [mpatches.Patch(color=colors[k], label=k) for k in draw_order if k in masks]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def build_device_overlay_masks(
    dev,
    include_core=True,
    include_dev=True,
    include_pml=True,
    include_ports=True,
    include_sources=True,
):
    # Use display coordinates if device supports orientation
    if hasattr(dev, "get_display_grid_size"):
        nx, ny = dev.get_display_grid_size()
    else:
        nx, ny = int(dev.nx), int(dev.ny)
    dx = 1.0 / float(dev.resolution)
    dy = 1.0 / float(dev.resolution)

    masks = {}

    if include_core:
        core = None
        if hasattr(dev, "core_mask"):
            core = dev.core_mask(nx, ny, dx, dy)
        if core is None:
            # Use get_device_window_um if available (handles orientation), else direct attributes
            if hasattr(dev, "get_device_window_um"):
                dev_cx, dev_cy, dev_wx, dev_wy = dev.get_device_window_um()
            else:
                dev_cx, dev_cy, dev_wx, dev_wy = dev.dev_cx, dev.dev_cy, dev.dev_wx, dev.dev_wy
            core = rect_mask(nx, ny, dx, dy, dev_cx, dev_cy, dev_wx, dev_wy)
        masks["core"] = core

    if include_dev:
        # Use get_device_window_um if available (handles orientation), else direct attributes
        if hasattr(dev, "get_device_window_um"):
            dev_cx, dev_cy, dev_wx, dev_wy = dev.get_device_window_um()
        else:
            dev_cx, dev_cy, dev_wx, dev_wy = dev.dev_cx, dev.dev_cy, dev.dev_wx, dev.dev_wy
        masks["dev"] = rect_mask(nx, ny, dx, dy, dev_cx, dev_cy, dev_wx, dev_wy)
    if include_pml:
        # Use display cell size if device supports orientation
        if hasattr(dev, "get_display_cell_size"):
            cell_x_display, cell_y_display = dev.get_display_cell_size()
        else:
            cell_x_display, cell_y_display = dev.cell_x, dev.cell_y
        masks["pml"] = pml_mask(nx, ny, dx, dy, cell_x_display, cell_y_display, dev.dpml)

    if include_ports and hasattr(dev, "get_ports"):
        ports = dev.get_ports() or {}
        for k, v in ports.items():
            masks[k] = vol_mask(nx, ny, dx, dy, v)

    if include_sources and hasattr(dev, "get_sources"):
        srcs = dev.get_sources() or {}
        for k, v in srcs.items():
            masks[k] = vol_mask(nx, ny, dx, dy, v)

    return masks

