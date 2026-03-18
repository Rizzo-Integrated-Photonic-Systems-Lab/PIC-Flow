"""
Preview all 9 device types at MAXIMUM parameter values for the final dataset.

Uses res=20 domain sizes:
  Rectangular: 480x160 px interior (24x8 µm)
  Square:      320x320 px interior (16x16 µm)

Confirms that the largest geometries fit within their domains.

Usage:
    python scripts/preview_final_dataset_devices.py
"""
import sys
import os

FDTD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FDTD")
sys.path.insert(0, FDTD_DIR)
for sub in ["straight_waveguide", "mmi", "sbend", "ybranch",
            "directional_coupler", "taper", "euler_bend", "circular_bend", "crossing"]:
    sys.path.insert(0, os.path.join(FDTD_DIR, sub))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import meep as mp

from unified_sweep import PARAM_RANGES, DEVICE_DOMAIN, build_device, get_device_masks

# ── Final dataset domain parameters (res=20) ──
RESOLUTION = 20
DPML_RAW = 1.0
PML_PX = int(round(DPML_RAW * RESOLUTION))         # 20 px
DPML = float(PML_PX) / float(RESOLUTION)            # 1.0 µm
WAVELENGTH = 1.55

# Rectangular domain
RECT_CROP_X = 480
RECT_CROP_Y = 160
RECT_CELL_X = float(RECT_CROP_X + 2 * PML_PX) / float(RESOLUTION)  # 26.0 µm
RECT_CELL_Y = float(RECT_CROP_Y + 2 * PML_PX) / float(RESOLUTION)  # 10.0 µm
RECT_INT_X = RECT_CELL_X - 2 * DPML  # 24.0 µm
RECT_INT_Y = RECT_CELL_Y - 2 * DPML  #  8.0 µm

# Square domain
SQ_CROP = 320
SQ_CELL = float(SQ_CROP + 2 * PML_PX) / float(RESOLUTION)  # 18.0 µm
SQ_INT = SQ_CELL - 2 * DPML  # 16.0 µm

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "figures", "final_dataset_max_params.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def max_params(device_type):
    return {k: v[1] for k, v in PARAM_RANGES[device_type].items()}


def get_epsilon_fast(dev):
    sim = mp.Simulation(
        cell_size=dev.cell,
        resolution=dev.resolution,
        boundary_layers=[mp.PML(dev.dpml)],
        geometry=dev.geometry,
        default_material=mp.Medium(index=1.444),
        sources=[],
    )
    sim.init_sim()
    eps = sim.get_epsilon().T.astype(np.float32)
    sim.reset_meep()
    return eps


def crop_pml(arr, pml_px):
    if pml_px > 0:
        return arr[pml_px:-pml_px, pml_px:-pml_px]
    return arr


def plot_device(ax, eps, src_mask, port_masks, port_ids, int_x, int_y, title):
    extent = [-int_x / 2, int_x / 2, -int_y / 2, int_y / 2]
    ax.imshow(eps, origin="lower", cmap="gray_r", extent=extent, aspect="equal",
              interpolation="nearest")

    # Source overlay (red)
    src_rgba = np.zeros((*src_mask.shape, 4), dtype=np.float32)
    src_rgba[..., 0] = 1.0
    src_rgba[..., 3] = src_mask * 0.6
    ax.imshow(src_rgba, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

    # Port overlays
    port_colors = [(0.0, 0.5, 1.0), (0.0, 0.8, 0.2), (1.0, 0.6, 0.0), (0.8, 0.0, 0.8)]
    for i in range(port_masks.shape[0]):
        rgba = np.zeros((*port_masks[i].shape, 4), dtype=np.float32)
        c = port_colors[i % len(port_colors)]
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c
        rgba[..., 3] = port_masks[i] * 0.5
        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    ax.set_ylabel("y (µm)", fontsize=8)
    ax.tick_params(labelsize=7)

    legend_items = [Patch(facecolor="red", alpha=0.6, label="Source")]
    for i, pid in enumerate(port_ids):
        c = port_colors[i % len(port_colors)]
        legend_items.append(Patch(facecolor=c, alpha=0.5, label=f"Port {pid}"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=6, framealpha=0.7)


# ── All 9 device types ──
RECT_DEVICES = [
    ("straight", "Straight Waveguide"),
    ("taper", "Taper"),
    ("sbend", "S-Bend (Euler)"),
    ("ybranch", "Y-Branch"),
    ("directional_coupler", "Directional Coupler"),
    ("mmi", "2x2 MMI"),
]

SQ_DEVICES = [
    ("euler_bend", "Euler Bend (90°)"),
    ("circular_bend", "Circular Bend (90°)"),
    ("crossing", "Waveguide Crossing"),
]

results = []  # (eps, src_mask, port_masks, port_ids, int_x, int_y, title, domain)

for device_type, display_name in RECT_DEVICES + SQ_DEVICES:
    params = max_params(device_type)
    is_sq = DEVICE_DOMAIN[device_type] == "square"

    if is_sq:
        cell_x, cell_y = SQ_CELL, SQ_CELL
        crop_x, crop_y = SQ_CROP, SQ_CROP
        int_x, int_y = SQ_INT, SQ_INT
    else:
        cell_x, cell_y = RECT_CELL_X, RECT_CELL_Y
        crop_x, crop_y = RECT_CROP_X, RECT_CROP_Y
        int_x, int_y = RECT_INT_X, RECT_INT_Y

    param_str = ", ".join(f"{k.replace('_um','')}={v:.2f}" for k, v in params.items())
    dom_tag = f"{'sq' if is_sq else 'rect'} {crop_x}x{crop_y}px, {int_x:.0f}x{int_y:.0f}µm"
    title = f"{display_name} (MAX)\n{param_str}\n[{dom_tag}]"

    print(f"\nBuilding {display_name} ({device_type}) with MAX params:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    try:
        dev = build_device(device_type, params, WAVELENGTH, RESOLUTION, DPML,
                           cell_x, cell_y, crop_x, crop_y)

        eps_full = get_epsilon_fast(dev)
        eps = crop_pml(eps_full, PML_PX)
        ny, nx = eps.shape
        print(f"  Interior grid: {nx}x{ny} px (expected {crop_x}x{crop_y})")

        src_mask, port_ids, port_masks = get_device_masks(
            dev, device_type, input_port=1,
            cell_x=cell_x, cell_y=cell_y, dpml=DPML, resolution=RESOLUTION,
            ny=ny, nx=nx, thickness_px=3,
        )

        results.append((eps, src_mask, port_masks, port_ids, int_x, int_y, title, "sq" if is_sq else "rect"))
        print(f"  OK")

    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((None, None, None, None, int_x, int_y, title + "\n[FAILED]", "sq" if is_sq else "rect"))

# ── Plot: rect devices on left column, square on right ──
n_rect = len(RECT_DEVICES)
n_sq = len(SQ_DEVICES)
n_rows = max(n_rect, n_sq)

fig = plt.figure(figsize=(24, 4.0 * n_rows))

# Use gridspec: left column wider (3:2 ratio for rect:sq aspect)
gs = fig.add_gridspec(n_rows, 2, width_ratios=[3, 2], hspace=0.4, wspace=0.3)

for i in range(n_rect):
    ax = fig.add_subplot(gs[i, 0])
    eps, src_mask, port_masks, port_ids, int_x, int_y, title, _ = results[i]
    if eps is not None:
        plot_device(ax, eps, src_mask, port_masks, port_ids, int_x, int_y, title)
    else:
        ax.set_title(title, fontsize=9, color="red")
        ax.text(0.5, 0.5, "Build failed", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="red")

for i in range(n_sq):
    ax = fig.add_subplot(gs[i, 1])
    idx = n_rect + i
    eps, src_mask, port_masks, port_ids, int_x, int_y, title, _ = results[idx]
    if eps is not None:
        plot_device(ax, eps, src_mask, port_masks, port_ids, int_x, int_y, title)
    else:
        ax.set_title(title, fontsize=9, color="red")
        ax.text(0.5, 0.5, "Build failed", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="red")

# Blank any unused grid cells
for i in range(n_sq, n_rows):
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")

fig.suptitle("Final Dataset: All 9 Devices at MAX Parameters (res=20)", fontsize=14, y=1.01)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
