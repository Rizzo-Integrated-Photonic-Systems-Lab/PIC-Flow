"""
Preview the half-ring coupler device at several parameter combinations.

Instantiates each device, extracts epsilon via sim.init_sim() + sim.get_epsilon()
(no FDTD run), overlays source and port masks, and saves PNGs.

Usage:
    python scripts/preview_half_ring.py
"""
import sys
import os

FDTD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FDTD")
sys.path.insert(0, FDTD_DIR)
for sub in ["half_ring_coupler"]:
    sys.path.insert(0, os.path.join(FDTD_DIR, sub))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import meep as mp

sys.path.insert(0, os.path.join(FDTD_DIR, "half_ring_coupler"))
from half_ring_coupler import HalfRingCoupler2D

# ── Cell parameters (match unified_sweep) ──
RESOLUTION = 14
DPML_RAW = 5.0 / 7.0
PML_PX = int(round(DPML_RAW * RESOLUTION))
DPML = float(PML_PX) / float(RESOLUTION)
CROP_X_PX = 336
CROP_Y_PX = 112
WAVELENGTH = 1.55

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_previews")
os.makedirs(OUT_DIR, exist_ok=True)


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


def draw_thick_line_mask(ny, nx, x0, y0, x1, y1, thickness_px=3):
    x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
    vx = x1f - x0f
    vy = y1f - y0f
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        m = np.zeros((ny, nx), dtype=np.float32)
        return m
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    t = ((xx - x0f) * vx + (yy - y0f) * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    px = x0f + t * vx
    py = y0f + t * vy
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    return (d2 <= thickness_px ** 2).astype(np.float32)


def get_masks(dev, input_port, ny, nx):
    src_px = dev.get_source_region_px(input_port=input_port, crop_pml=True)
    src_mask = draw_thick_line_mask(
        ny, nx,
        src_px["line_start_px"][0], src_px["line_start_px"][1],
        src_px["line_end_px"][0], src_px["line_end_px"][1],
    )
    port_ids = [1, 2, 3, 4]
    port_masks = []
    for p in port_ids:
        pr = dev.get_port_region_px(p, crop_pml=True)
        pm = draw_thick_line_mask(
            ny, nx,
            pr["line_start_px"][0], pr["line_start_px"][1],
            pr["line_end_px"][0], pr["line_end_px"][1],
        )
        port_masks.append(pm)
    return src_mask, np.array(port_ids), np.stack(port_masks, axis=0)


def plot_device(ax, eps, src_mask, port_masks, port_ids, interior_x, interior_y, title):
    extent = [-interior_x / 2, interior_x / 2, -interior_y / 2, interior_y / 2]
    ax.imshow(eps, origin="lower", cmap="gray_r", extent=extent, aspect="auto",
              interpolation="nearest")

    # Source overlay (red)
    src_rgba = np.zeros((*src_mask.shape, 4), dtype=np.float32)
    src_rgba[..., 0] = 1.0
    src_rgba[..., 3] = src_mask * 0.6
    ax.imshow(src_rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    # Port overlays
    port_colors = [(0.0, 0.5, 1.0), (0.0, 0.8, 0.2), (1.0, 0.6, 0.0), (0.8, 0.0, 0.8)]
    for i in range(port_masks.shape[0]):
        pm = port_masks[i]
        rgba = np.zeros((*pm.shape, 4), dtype=np.float32)
        c = port_colors[i % len(port_colors)]
        rgba[..., 0] = c[0]
        rgba[..., 1] = c[1]
        rgba[..., 2] = c[2]
        rgba[..., 3] = pm * 0.5
        ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")

    legend_items = [Patch(facecolor="red", alpha=0.6, label="Source")]
    for i, pid in enumerate(port_ids):
        c = port_colors[i % len(port_colors)]
        legend_items.append(Patch(facecolor=c, alpha=0.5, label=f"Port {pid}"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.7)


# ── Parameter combinations to preview ──
CONFIGS = [
    {
        "name": "small_gap_short",
        "params": dict(wg_width_um=0.5, gap_um=0.15, coupling_length_um=5.0,
                       bend_length_um=4.0, ring_offset_um=1.0),
    },
    {
        "name": "medium",
        "params": dict(wg_width_um=0.45, gap_um=0.2, coupling_length_um=6.0,
                       bend_length_um=4.5, ring_offset_um=1.5),
    },
    {
        "name": "large_gap_long",
        "params": dict(wg_width_um=0.5, gap_um=0.3, coupling_length_um=8.0,
                       bend_length_um=3.5, ring_offset_um=2.0),
    },
    {
        "name": "tight_coupling",
        "params": dict(wg_width_um=0.43, gap_um=0.11, coupling_length_um=4.0,
                       bend_length_um=5.0, ring_offset_um=1.2),
    },
]

interior_x = float(CROP_X_PX) / float(RESOLUTION)
interior_y = float(CROP_Y_PX) / float(RESOLUTION)

results = []

for cfg in CONFIGS:
    name = cfg["name"]
    params = cfg["params"]
    print(f"\nBuilding half_ring_coupler ({name}):")
    for k, v in params.items():
        print(f"  {k} = {v}")

    dev = HalfRingCoupler2D(
        **params,
        wavelength_um=WAVELENGTH,
        resolution=RESOLUTION,
        dpml=DPML,
        crop_x_px=CROP_X_PX,
        crop_y_px=CROP_Y_PX,
        quantize_grid=True,
        fit_margin_um=0.5,
    )

    print(f"  Cell: {dev.cell_x:.3f} x {dev.cell_y:.3f} um")
    print(f"  Bus y: {dev.bus_y_um:.3f}, Ring lead y: {dev.ring_lead_y_um:.3f}")
    print(f"  Bend length: {dev.L_bend_um:.3f}")

    eps_full = get_epsilon_fast(dev)
    eps = crop_pml(eps_full, PML_PX)
    ny, nx = eps.shape
    print(f"  Interior grid: {nx} x {ny} px")

    src_mask, port_ids, port_masks = get_masks(dev, input_port=1, ny=ny, nx=nx)

    param_str = ", ".join(f"{k.replace('_um','')}={v:.2f}" for k, v in params.items())
    title = f"Half-Ring Coupler ({name})\n{param_str}\n[{nx}x{ny} px, {interior_x:.1f}x{interior_y:.1f} um]"

    results.append((name, eps, src_mask, port_masks, port_ids, title))

# ── Save individual images ──
for name, eps, src_mask, port_masks, port_ids, title in results:
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    plot_device(ax, eps, src_mask, port_masks, port_ids, interior_x, interior_y, title)
    out_path = os.path.join(OUT_DIR, f"half_ring_{name}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Combined grid ──
n = len(results)
fig, axes = plt.subplots(n, 1, figsize=(20, 4.5 * n))
if n == 1:
    axes = [axes]

for idx, (name, eps, src_mask, port_masks, port_ids, title) in enumerate(results):
    plot_device(axes[idx], eps, src_mask, port_masks, port_ids, interior_x, interior_y, title)

grid_path = os.path.join(OUT_DIR, "half_ring_grid.png")
fig.tight_layout()
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined grid saved: {grid_path}")
