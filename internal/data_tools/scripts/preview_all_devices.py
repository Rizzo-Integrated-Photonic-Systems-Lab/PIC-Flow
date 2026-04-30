"""
Preview all 5 device types at maximum parameter values from unified_sweep.

Instantiates each device using the unified sweep cell dimensions, extracts
epsilon via sim.init_sim() + sim.get_epsilon() (no FDTD run), overlays
source and port masks, and saves individual + combined grid PNGs.

Usage:
    cd FDTD && python ../scripts/preview_all_devices.py
"""
import sys
import os

# Add FDTD directory and device subdirectories to path
FDTD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FDTD")
sys.path.insert(0, FDTD_DIR)
for sub in ["straight_waveguide", "mmi", "sbend", "ybranch", "directional_coupler"]:
    sys.path.insert(0, os.path.join(FDTD_DIR, sub))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import meep as mp

from straight_waveguide.straight import StraightWaveguide2D
from mmi.mmi import MMI2x2
from sbend.sbend import EulerSBend2D
from ybranch.ybranch import YBranch2D
from directional_coupler.directional_coupler import DirectionalCoupler2D
from unified_sweep import PARAM_RANGES, build_device, get_device_masks

# ── Unified sweep cell parameters ──
RESOLUTION = 18
DPML_RAW = 2.0 / 3.0
PML_PX = int(round(DPML_RAW * RESOLUTION))          # 12 px
DPML = float(PML_PX) / float(RESOLUTION)             # 0.6667 µm
CROP_X_PX = 512
CROP_Y_PX = 192
CELL_X = float(CROP_X_PX + 2 * PML_PX) / float(RESOLUTION)  # ≈ 29.778 µm
CELL_Y = float(CROP_Y_PX + 2 * PML_PX) / float(RESOLUTION)  # = 12.0 µm
WAVELENGTH = 1.55

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_previews")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Max parameters per device ──
def max_params(device_type):
    """Extract max values from PARAM_RANGES."""
    return {k: v[1] for k, v in PARAM_RANGES[device_type].items()}


def get_epsilon_fast(dev):
    """Extract epsilon grid without running FDTD (init_sim only)."""
    sim = mp.Simulation(
        cell_size=dev.cell,
        resolution=dev.resolution,
        boundary_layers=[mp.PML(dev.dpml)],
        geometry=dev.geometry,
        default_material=mp.Medium(index=1.444),
        sources=[],
    )
    sim.init_sim()
    eps = sim.get_epsilon().T.astype(np.float32)  # [ny, nx]
    sim.reset_meep()
    return eps


def crop_pml(arr, pml_px):
    """Crop PML border from a 2D array."""
    if pml_px > 0:
        return arr[pml_px:-pml_px, pml_px:-pml_px]
    return arr


def plot_device(ax, eps, src_mask, port_masks, port_ids, interior_x, interior_y, title):
    """Plot epsilon with source/port mask overlays on a given axes."""
    extent = [-interior_x / 2, interior_x / 2, -interior_y / 2, interior_y / 2]

    # Epsilon as grayscale background
    ax.imshow(eps, origin="lower", cmap="gray_r", extent=extent, aspect="auto",
              interpolation="nearest")

    # Source mask overlay (red, semi-transparent)
    src_rgba = np.zeros((*src_mask.shape, 4), dtype=np.float32)
    src_rgba[..., 0] = 1.0  # red
    src_rgba[..., 3] = src_mask * 0.6
    ax.imshow(src_rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    # Port mask overlays (distinct colors)
    port_colors = [
        (0.0, 0.5, 1.0),   # blue
        (0.0, 0.8, 0.2),   # green
        (1.0, 0.6, 0.0),   # orange
        (0.8, 0.0, 0.8),   # purple
    ]
    for i in range(port_masks.shape[0]):
        pm = port_masks[i]
        rgba = np.zeros((*pm.shape, 4), dtype=np.float32)
        c = port_colors[i % len(port_colors)]
        rgba[..., 0] = c[0]
        rgba[..., 1] = c[1]
        rgba[..., 2] = c[2]
        rgba[..., 3] = pm * 0.5
        ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    # PML boundary rectangle (dashed)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor="red", alpha=0.6, label="Source")]
    for i, pid in enumerate(port_ids):
        c = port_colors[i % len(port_colors)]
        legend_items.append(Patch(facecolor=c, alpha=0.5, label=f"Port {pid}"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.7)


# ── Build all devices ──
DEVICES = [
    ("straight", "Straight Waveguide"),
    ("mmi", "2x2 MMI"),
    ("sbend", "S-Bend (Euler)"),
    ("ybranch", "Y-Branch"),
    ("directional_coupler", "Directional Coupler"),
]

interior_x = CELL_X - 2 * DPML
interior_y = CELL_Y - 2 * DPML
ny_crop = CROP_Y_PX
nx_crop = CROP_X_PX

results = []  # (name, eps, src_mask, port_masks, port_ids, title)

for device_type, display_name in DEVICES:
    params = max_params(device_type)
    print(f"\n{'='*60}")
    print(f"Building {display_name} with MAX params:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    dev = build_device(device_type, params, WAVELENGTH, RESOLUTION, DPML,
                       CELL_X, CELL_Y, CROP_X_PX, CROP_Y_PX)

    print(f"  Cell: {dev.cell_x:.3f} x {dev.cell_y:.3f} µm")
    print(f"  Grid: {dev.nx} x {dev.ny} px (full)")

    # Get epsilon without running FDTD
    eps_full = get_epsilon_fast(dev)
    eps = crop_pml(eps_full, PML_PX)
    ny, nx = eps.shape
    print(f"  Interior grid: {nx} x {ny} px")

    # Get masks (input_port=1 for all)
    src_mask, port_ids, port_masks = get_device_masks(
        dev, device_type, input_port=1,
        cell_x=CELL_X, cell_y=CELL_Y, dpml=DPML, resolution=RESOLUTION,
        ny=ny, nx=nx, thickness_px=3,
    )

    param_str = ", ".join(f"{k.replace('_um','')}={v:.2f}" for k, v in params.items())
    title = f"{display_name} (MAX)\n{param_str}\n[{nx}x{ny} px, {interior_x:.1f}x{interior_y:.1f} µm]"

    results.append((device_type, eps, src_mask, port_masks, port_ids, title))

# ── Save individual images ──
file_names = {
    "straight": "straight_max.png",
    "mmi": "mmi_max.png",
    "sbend": "sbend_max.png",
    "ybranch": "ybranch_max.png",
    "directional_coupler": "coupler_max.png",
}

for device_type, eps, src_mask, port_masks, port_ids, title in results:
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    plot_device(ax, eps, src_mask, port_masks, port_ids, interior_x, interior_y, title)
    out_path = os.path.join(OUT_DIR, file_names[device_type])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Combined grid ──
n_devices = len(results)
fig, axes = plt.subplots(n_devices, 1, figsize=(20, 4.5 * n_devices))
if n_devices == 1:
    axes = [axes]

for idx, (device_type, eps, src_mask, port_masks, port_ids, title) in enumerate(results):
    plot_device(axes[idx], eps, src_mask, port_masks, port_ids, interior_x, interior_y, title)

grid_path = os.path.join(OUT_DIR, "all_devices_grid.png")
fig.tight_layout()
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined grid saved: {grid_path}")
print(f"All images in: {OUT_DIR}")
