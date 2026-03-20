#!/usr/bin/env python3
"""
Validate all device geometries fit within the target domain and generate
a collage of epsilon maps at max/interesting parameter points.

Active final-dataset config: 160 x 480 pixels at res=20 → 8.0 x 24.0 µm
interior (after PML cropping; dpml = 1.0 µm → pml_px = 20).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FDTD"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from unified_sweep import (
    build_device,
    DEVICE_DOMAIN,
    DOMAIN_RECTANGULAR,
    DOMAIN_SQUARE,
)

# ── Target grid ──────────────────────────────────────────────────────────────
RESOLUTION   = 20
DPML         = 1.0
PML_PX       = round(DPML * RESOLUTION)             # 20 px
DPML_Q       = PML_PX / RESOLUTION                  # quantised
CROP_X_PX    = 480                                  # propagation (X)
CROP_Y_PX    = 160                                  # transverse  (Y)
CELL_X       = (CROP_X_PX + 2 * PML_PX) / RESOLUTION   # 26.0 µm
CELL_Y       = (CROP_Y_PX + 2 * PML_PX) / RESOLUTION   # 10.0 µm
INNER_X      = CROP_X_PX / RESOLUTION               # 24.0 µm
INNER_Y      = CROP_Y_PX / RESOLUTION               # 8.0 µm
SQ_CROP_PX   = 320
SQ_CELL      = (SQ_CROP_PX + 2 * PML_PX) / RESOLUTION  # 18.0 µm
SQ_INNER     = SQ_CROP_PX / RESOLUTION              # 16.0 µm

print(f"Target grid : {CROP_Y_PX}×{CROP_X_PX} px  ({INNER_Y:.1f}×{INNER_X:.1f} µm interior)")
print(f"Cell        : {CELL_Y:.3f}×{CELL_X:.3f} µm  (with {DPML_Q:.4f} µm PML = {PML_PX} px)")
print()

# ── Parameter configs ────────────────────────────────────────────────────────
# For each device we define: min, mid, max, and one "interesting physics" case
DEVICE_CONFIGS = {
    "straight": {
        "params": {
            "min": {"wg_width_um": 0.40, "dev_length_um": 6.0},
            "mid": {"wg_width_um": 0.50, "dev_length_um": 12.0},
            "max": {"wg_width_um": 0.575, "dev_length_um": 18.0},
            "long": {"wg_width_um": 0.575, "dev_length_um": 18.0},
        },
    },
    "taper": {
        "params": {
            "min": {"wg_width_in": 0.40, "wg_width_out": 0.60, "taper_length_um": 3.0},
            "mid": {"wg_width_in": 0.50, "wg_width_out": 1.30, "taper_length_um": 9.0},
            "max": {"wg_width_in": 0.575, "wg_width_out": 2.0, "taper_length_um": 15.0},
            "abrupt": {"wg_width_in": 0.40, "wg_width_out": 2.0, "taper_length_um": 3.0},
        },
    },
    "sbend": {
        "params": {
            "min": {"wg_width_um": 0.40, "lateral_offset_um": 2.0, "R_min_um": 3.0},
            "mid": {"wg_width_um": 0.50, "lateral_offset_um": 3.75, "R_min_um": 5.0},
            "max": {"wg_width_um": 0.575, "lateral_offset_um": 5.5, "R_min_um": 7.0},
            "tight": {"wg_width_um": 0.575, "lateral_offset_um": 5.5, "R_min_um": 3.0},
        },
    },
    "ybranch": {
        "params": {
            "min": {"wg_width_um": 0.40, "l_junction_um": 1.0, "l_bend_um": 4.0, "h_bend_um": 0.575, "l_out_um": 1.0},
            "mid": {"wg_width_um": 0.50, "l_junction_um": 2.0, "l_bend_um": 5.5, "h_bend_um": 1.5, "l_out_um": 2.5},
            "max": {"wg_width_um": 0.575, "l_junction_um": 3.0, "l_bend_um": 7.0, "h_bend_um": 2.5, "l_out_um": 4.0},
            "wide_split": {"wg_width_um": 0.575, "l_junction_um": 1.5, "l_bend_um": 5.0, "h_bend_um": 2.5, "l_out_um": 2.0},
        },
    },
    "directional_coupler": {
        "params": {
            "min": {"wg_width_um": 0.40, "gap_um": 0.10, "wg_length_um": 5.0, "bend_length_um": 4.0, "lead_extra_gap_um": 0.825},
            "mid": {"wg_width_um": 0.50, "gap_um": 0.225, "wg_length_um": 6.5, "bend_length_um": 5.0, "lead_extra_gap_um": 1.4},
            "max": {"wg_width_um": 0.575, "gap_um": 0.35, "wg_length_um": 8.0, "bend_length_um": 6.0, "lead_extra_gap_um": 2.0},
            "strong_coupling": {"wg_width_um": 0.50, "gap_um": 0.10, "wg_length_um": 8.0, "bend_length_um": 5.0, "lead_extra_gap_um": 1.0},
        },
    },
    "mmi": {
        "params": {
            "min": {"wg_width_um": 0.40, "mmi_width_um": 4.5, "mmi_length_um": 8.0, "taper_width_um": 0.575, "taper_length_um": 1.0},
            "mid": {"wg_width_um": 0.50, "mmi_width_um": 5.0, "mmi_length_um": 11.5, "taper_width_um": 1.0, "taper_length_um": 2.0},
            "max": {"wg_width_um": 0.575, "mmi_width_um": 5.5, "mmi_length_um": 15.0, "taper_width_um": 1.5, "taper_length_um": 3.0},
            "3dB": {"wg_width_um": 0.50, "mmi_width_um": 5.0, "mmi_length_um": 10.0, "taper_width_um": 1.0, "taper_length_um": 2.0},
        },
    },
    "euler_bend": {
        "params": {
            "min": {"wg_width": 0.40, "R_min_um": 2.0},
            "mid": {"wg_width": 0.50, "R_min_um": 2.7},
            "max": {"wg_width": 0.575, "R_min_um": 3.4},
            "tight": {"wg_width": 0.575, "R_min_um": 2.0},
        },
    },
    "circular_bend": {
        "params": {
            "min": {"wg_width": 0.40, "bend_radius_um": 2.0},
            "mid": {"wg_width": 0.50, "bend_radius_um": 2.75},
            "max": {"wg_width": 0.575, "bend_radius_um": 3.5},
            "tight": {"wg_width": 0.575, "bend_radius_um": 2.0},
        },
    },
    "crossing": {
        "params": {
            "min": {"wg_width_h": 0.40, "wg_width_v": 0.40},
            "mid": {"wg_width_h": 0.50, "wg_width_v": 0.50},
            "max": {"wg_width_h": 0.575, "wg_width_v": 0.575},
            "asym": {"wg_width_h": 0.575, "wg_width_v": 0.40},
        },
    },
}


def _generic_get_eps(dev):
    """Get epsilon from any device using meep sim.init_sim()."""
    import meep as mp
    sim = mp.Simulation(
        cell_size=dev.cell,
        resolution=int(dev.resolution),
        boundary_layers=[mp.PML(float(dev.dpml))],
        geometry=dev.geometry,
        default_material=dev.clad_medium,
        sources=[],
    )
    sim.init_sim()
    eps = sim.get_epsilon().T.astype(np.float32)  # [ny, nx]
    sim.reset_meep()
    return eps


def build_device_epsilon(device_type, params, wavelength_um=1.55):
    """Build a device and return its epsilon map (interior only, PML cropped)."""
    try:
        domain = DEVICE_DOMAIN[device_type]
        if domain == DOMAIN_RECTANGULAR:
            cell_x, cell_y = CELL_X, CELL_Y
            crop_x_px, crop_y_px = CROP_X_PX, CROP_Y_PX
        elif domain == DOMAIN_SQUARE:
            cell_x, cell_y = SQ_CELL, SQ_CELL
            crop_x_px, crop_y_px = SQ_CROP_PX, SQ_CROP_PX
        else:
            raise ValueError(f"Unknown domain for {device_type}: {domain}")

        dev = build_device(
            device_type=device_type,
            params=params,
            wavelength_um=wavelength_um,
            resolution=RESOLUTION,
            dpml=DPML_Q,
            cell_x=cell_x,
            cell_y=cell_y,
            crop_x_px=crop_x_px,
            crop_y_px=crop_y_px,
        )

        # Use device-specific method if available, else generic meep approach
        if hasattr(dev, "get_eps_and_cell"):
            eps, _ = dev.get_eps_and_cell(crop_pml=True)
        else:
            eps_full = _generic_get_eps(dev)
            # Crop PML
            if PML_PX > 0:
                eps = eps_full[PML_PX:-PML_PX, PML_PX:-PML_PX]
            else:
                eps = eps_full

        return eps, None

    except Exception as e:
        import traceback
        return None, str(e)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # ── Validate all devices ─────────────────────────────────────────────
    print(f"{'Device':<25s} {'Case':<18s} {'Shape':>12s} {'Expected':>12s} {'Status':<8s} {'Notes'}")
    print("─" * 100)
    results = {}  # {device_type: {case_name: (eps, params, error)}}

    for dev_name, cfg in DEVICE_CONFIGS.items():
        results[dev_name] = {}
        for case_name, params in cfg["params"].items():
            eps, err = build_device_epsilon(dev_name, params)
            domain = DEVICE_DOMAIN[dev_name]
            expected_shape = (CROP_Y_PX, CROP_X_PX) if domain == DOMAIN_RECTANGULAR else (SQ_CROP_PX, SQ_CROP_PX)
            shape_str = f"{eps.shape[0]}×{eps.shape[1]}" if eps is not None else "FAILED"
            exp_str = f"{expected_shape[0]}×{expected_shape[1]}"

            if err:
                status = "FAIL"
                notes = err[:60]
            elif eps.shape != expected_shape:
                status = "MISMATCH"
                notes = f"Got {eps.shape}, expected {expected_shape}"
            else:
                status = "OK"
                n_si = np.sum(eps > 5.0)
                fill = n_si / eps.size * 100
                notes = f"Si fill={fill:.1f}%"

            results[dev_name][case_name] = (eps, params, err)
            print(f"{dev_name:<25s} {case_name:<18s} {shape_str:>12s} {exp_str:>12s} {status:<8s} {notes}")

    print()

    # ── Generate collage ─────────────────────────────────────────────────
    device_names = list(DEVICE_CONFIGS.keys())
    n_devices = len(device_names)
    case_names = ["min", "mid", "max", list(DEVICE_CONFIGS[device_names[0]]["params"].keys())[-1]]  # 4 cases

    fig = plt.figure(figsize=(24, 3.5 * n_devices))
    gs = GridSpec(n_devices, 4, figure=fig, hspace=0.3, wspace=0.15)

    for row, dev_name in enumerate(device_names):
        cases = list(DEVICE_CONFIGS[dev_name]["params"].keys())
        for col, case_name in enumerate(cases[:4]):
            ax = fig.add_subplot(gs[row, col])
            eps, params, err = results[dev_name][case_name]

            if eps is not None:
                # Plot epsilon with good contrast
                ax.imshow(eps.T if eps.shape[0] < eps.shape[1] else eps,
                          origin="lower", cmap="gray_r", aspect="auto",
                          extent=[0, eps.shape[1]/RESOLUTION, 0, eps.shape[0]/RESOLUTION])
                ax.set_xlabel("µm", fontsize=7)
                ax.set_ylabel("µm", fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                ax.text(0.5, 0.5, f"FAILED\n{err[:40]}", ha="center", va="center",
                        fontsize=8, color="red", transform=ax.transAxes)

            # Title with key params
            param_str = ", ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in params.items())
            ax.set_title(f"{dev_name} [{case_name}]\n{param_str}", fontsize=7, fontweight="bold")

    fig.suptitle(
        f"Device Geometry Validation — {CROP_Y_PX}×{CROP_X_PX} px @ res={RESOLUTION} "
        f"({INNER_Y:.0f}×{INNER_X:.0f} µm)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    collage_path = os.path.join(out_dir, "device_geometry_validation.png")
    fig.savefig(collage_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved collage: {collage_path}")

    # ── Physics sampling notes ───────────────────────────────────────────
    print()
    print("=== Physics Sampling Notes ===")
    print()
    print("Directional Coupler:")
    print("  Coupling coefficient κ ∝ exp(-gap/λ_decay) · coupling_length")
    print("  At gap=0.1 µm, λ=1.55: beat length L_π ≈ 8-12 µm")
    print("  → wg_length range [5, 8] µm samples from weak to near-full crossover")
    print("  → 5 wavelengths will show strong wavelength dependence of κ")
    print()
    print("MMI 2x2:")
    print("  Self-imaging length L_π ∝ W² · n_eff / λ")
    print("  For W=4 µm: L_π ≈ 20 µm, 3dB at L_π/2 ≈ 10 µm")
    print("  → mmi_length [8, 15] spans from under-length to over-length")
    print("  → 5 wavelengths will show L_π shift with λ")
    print()
    print("Y-Branch:")
    print("  Splitting ratio depends on junction taper and bend angle")
    print("  → h_bend controls effective split angle → excess loss")
    print("  → Multi-wavelength shows modest wavelength dependence")
    print()
    print("S-Bend:")
    print("  Loss = radiation loss from curvature + mode mismatch at transitions")
    print("  → tight case (large offset, small R) has most loss and wavelength dependence")
    print()
    print("=== Wavelength Recommendation ===")
    print("  5 wavelengths: [1.500, 1.525, 1.550, 1.575, 1.600] µm")
    print("  Adds 1.525 and 1.575 → better resolves wavelength-dependent coupling")
    print("  Especially important for directional_coupler (sinusoidal transfer function)")
    print("  and MMI (self-imaging length shifts with λ)")
    print(f"  Dataset size increase: ×{5/3:.2f} (5/3 wavelengths)")


if __name__ == "__main__":
    main()
