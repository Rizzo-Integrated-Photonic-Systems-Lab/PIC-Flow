"""
Preview all 9 device types in a SINGLE rectangular domain (480x160 = 24x8 µm @ res=20).

Bends use reduced R to fit in 8 µm height. Crossing uses the rect cell directly.

Usage:
    python scripts/preview_all_rect.py
"""
import sys, os

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

from unified_sweep import build_device, get_device_masks

# ── Rectangular domain @ res=20 ──
RESOLUTION = 20
DPML = 1.0
PML_PX = 20
CROP_X, CROP_Y = 480, 160
CELL_X = float(CROP_X + 2 * PML_PX) / RESOLUTION   # 26.0 µm
CELL_Y = float(CROP_Y + 2 * PML_PX) / RESOLUTION   # 10.0 µm
INT_X = CELL_X - 2 * DPML  # 24.0 µm
INT_Y = CELL_Y - 2 * DPML  #  8.0 µm
WL = 1.55

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "figures", "all_devices_rect_max.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def get_epsilon_fast(dev):
    sim = mp.Simulation(cell_size=dev.cell, resolution=dev.resolution,
                        boundary_layers=[mp.PML(dev.dpml)],
                        geometry=dev.geometry,
                        default_material=mp.Medium(index=1.444), sources=[])
    sim.init_sim()
    eps = sim.get_epsilon().T.astype(np.float32)
    sim.reset_meep()
    return eps


def crop_pml(arr, pml_px):
    return arr[pml_px:-pml_px, pml_px:-pml_px] if pml_px > 0 else arr


# ── Device definitions: all at MAX params that fit 24x8 µm ──
DEVICES = [
    ("straight", "Straight Waveguide", {
        "wg_width_um": 0.575, "dev_length_um": 18.0,
    }),
    ("taper", "Taper", {
        "wg_width_in": 0.575, "wg_width_out": 2.0, "taper_length_um": 15.0,
    }),
    ("sbend", "S-Bend (Euler)", {
        "wg_width_um": 0.575, "lateral_offset_um": 5.5, "R_min_um": 7.0,
    }),
    ("ybranch", "Y-Branch", {
        "wg_width_um": 0.575, "l_junction_um": 3.0, "l_bend_um": 7.0,
        "h_bend_um": 2.5, "l_out_um": 4.0,
    }),
    ("directional_coupler", "Directional Coupler", {
        "wg_width_um": 0.575, "gap_um": 0.35, "wg_length_um": 8.0,
        "bend_length_um": 6.0, "lead_extra_gap_um": 2.0,
    }),
    ("mmi", "2x2 MMI", {
        "wg_width_um": 0.575, "mmi_width_um": 5.5, "mmi_length_um": 15.0,
        "taper_width_um": 1.5, "taper_length_um": 3.0,
    }),
    # Bends: R reduced to fit 8 µm height
    # 90° bend vertical extent ≈ R. With 8 µm, R_max ≈ 3.5 µm leaves ~0.5 µm margin
    # for port placement + waveguide width on each side.
    ("euler_bend", "Euler Bend (90°)", {
        "wg_width": 0.575, "R_min_um": 3.5,
    }),
    ("circular_bend", "Circular Bend (90°)", {
        "wg_width": 0.575, "bend_radius_um": 3.5,
    }),
    # Crossing: fits trivially, vertical WG just has 8 µm instead of 16 µm
    ("crossing", "Waveguide Crossing", {
        "wg_width_h": 0.575, "wg_width_v": 0.575,
    }),
]

results = []
for device_type, name, params in DEVICES:
    param_str = ", ".join(f"{k.replace('_um','')}={v:.2f}" for k, v in params.items())
    title = f"{name} (MAX)\n{param_str}"
    print(f"\nBuilding {name} ({device_type})...")
    for k, v in params.items():
        print(f"  {k} = {v}")

    try:
        dev = build_device(device_type, params, WL, RESOLUTION, DPML,
                           CELL_X, CELL_Y, CROP_X, CROP_Y)
        eps = crop_pml(get_epsilon_fast(dev), PML_PX)
        ny, nx = eps.shape
        print(f"  Grid: {nx}x{ny} px  OK")

        src_mask, port_ids, port_masks = get_device_masks(
            dev, device_type, input_port=1,
            cell_x=CELL_X, cell_y=CELL_Y, dpml=DPML, resolution=RESOLUTION,
            ny=ny, nx=nx, thickness_px=3)

        results.append((eps, src_mask, port_masks, port_ids, title, True))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((None, None, None, None, title + f"\nFAILED: {e}", False))

# ── Plot ──
n = len(results)
fig, axes = plt.subplots(n, 1, figsize=(20, 2.8 * n))
if n == 1:
    axes = [axes]

extent = [-INT_X / 2, INT_X / 2, -INT_Y / 2, INT_Y / 2]
port_colors = [(0.0, 0.5, 1.0), (0.0, 0.8, 0.2), (1.0, 0.6, 0.0), (0.8, 0.0, 0.8)]

for idx, (eps, src_mask, port_masks, port_ids, title, ok) in enumerate(results):
    ax = axes[idx]
    if not ok:
        ax.set_title(title, fontsize=9, color="red")
        ax.text(0.5, 0.5, "Build failed", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="red")
        continue

    ax.imshow(eps, origin="lower", cmap="gray_r", extent=extent, aspect="auto",
              interpolation="nearest")

    # Source (red)
    rgba = np.zeros((*src_mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0; rgba[..., 3] = src_mask * 0.6
    ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    # Ports
    for i in range(port_masks.shape[0]):
        rgba = np.zeros((*port_masks[i].shape, 4), dtype=np.float32)
        c = port_colors[i % len(port_colors)]
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c
        rgba[..., 3] = port_masks[i] * 0.5
        ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (µm)", fontsize=8)
    ax.set_ylabel("y (µm)", fontsize=8)
    ax.tick_params(labelsize=7)

    legend = [Patch(facecolor="red", alpha=0.6, label="Source")]
    for i, pid in enumerate(port_ids):
        legend.append(Patch(facecolor=port_colors[i % len(port_colors)], alpha=0.5, label=f"Port {pid}"))
    ax.legend(handles=legend, loc="upper right", fontsize=6, framealpha=0.7)

fig.suptitle("All 9 Devices in Rectangular Domain (480x160 px = 24x8 µm, res=20)", fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
