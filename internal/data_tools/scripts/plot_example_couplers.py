"""Generate example directional coupler geometry images at min/max parameter extremes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FDTD"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FDTD", "directional_coupler"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from directional_coupler import DirectionalCoupler2D

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "example_couplers")
os.makedirs(OUT_DIR, exist_ok=True)

# Parameter extremes
CASES = {
    "min_all": dict(
        wg_width_um=0.38, gap_um=0.10, wg_length_um=5.0,
        bend_length_um=4.0, lead_extra_gap_um=0.8,
        label="MIN: w=0.38, gap=0.10, Lc=5.0, bend=4.0, lead=0.8",
    ),
    "max_all": dict(
        wg_width_um=0.60, gap_um=0.35, wg_length_um=15.0,
        bend_length_um=6.0, lead_extra_gap_um=2.5,
        label="MAX: w=0.60, gap=0.35, Lc=15.0, bend=6.0, lead=2.5",
    ),
    "tight_gap_long": dict(
        wg_width_um=0.45, gap_um=0.10, wg_length_um=15.0,
        bend_length_um=5.0, lead_extra_gap_um=1.5,
        label="Tight gap + long coupling: gap=0.10, Lc=15.0",
    ),
    "wide_gap_short": dict(
        wg_width_um=0.45, gap_um=0.35, wg_length_um=5.0,
        bend_length_um=5.0, lead_extra_gap_um=1.5,
        label="Wide gap + short coupling: gap=0.35, Lc=5.0",
    ),
    "typical_3dB": dict(
        wg_width_um=0.45, gap_um=0.20, wg_length_um=10.0,
        bend_length_um=5.0, lead_extra_gap_um=1.0,
        label="Typical 3dB coupler: w=0.45, gap=0.20, Lc=10.0",
    ),
}

RESOLUTION = 20
DPML = 2.0 / 3.0
CROP_X_PX = 640
CROP_Y_PX = 128
WAVELENGTH = 1.55

fig, axes = plt.subplots(len(CASES), 1, figsize=(18, 3 * len(CASES)))
if len(CASES) == 1:
    axes = [axes]

for idx, (name, cfg) in enumerate(CASES.items()):
    label = cfg.pop("label")
    print(f"Building {name}: {label}")

    dc = DirectionalCoupler2D(
        wavelength_um=WAVELENGTH,
        resolution=RESOLUTION,
        dpml=DPML,
        crop_x_px=CROP_X_PX,
        crop_y_px=CROP_Y_PX,
        quantize_grid=True,
        fit_margin_um=0.5,
        **cfg,
    )

    eps, (cx, cy) = dc.get_eps_and_cell(crop_pml=True)
    ny, nx = eps.shape

    print(f"  Grid: {nx}x{ny} px, cell: {cx:.2f}x{cy:.2f} um")

    # Plot individual image
    fig_i, ax_i = plt.subplots(1, 1, figsize=(18, 3))
    extent = [-cx / 2, cx / 2, -cy / 2, cy / 2]
    ax_i.imshow(eps, origin="lower", cmap="gray_r", extent=extent, aspect="auto")
    ax_i.set_title(f"{label}  [{nx}x{ny} px]", fontsize=11)
    ax_i.set_xlabel("x (µm)")
    ax_i.set_ylabel("y (µm)")
    out_path = os.path.join(OUT_DIR, f"{name}.png")
    fig_i.tight_layout()
    fig_i.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_i)
    print(f"  Saved: {out_path}")

    # Add to combined figure
    ax = axes[idx]
    # Use non-PML cell dimensions for extent
    inner_x = cx - 2 * dc.dpml if hasattr(dc, 'dpml') else cx
    inner_y = cy - 2 * dc.dpml if hasattr(dc, 'dpml') else cy
    extent_crop = [-inner_x / 2, inner_x / 2, -inner_y / 2, inner_y / 2]
    ax.imshow(eps, origin="lower", cmap="gray_r", extent=extent_crop, aspect="auto")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

# Save combined
combined_path = os.path.join(OUT_DIR, "all_examples.png")
fig.tight_layout()
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined figure saved: {combined_path}")
print(f"All images in: {OUT_DIR}")
