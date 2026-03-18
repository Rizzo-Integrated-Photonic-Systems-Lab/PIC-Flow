#!/usr/bin/env python3
"""
Validate all device geometries fit within the target domain and generate
a collage of epsilon maps at max/interesting parameter points.

Target domain: 112 x 336 pixels at res=14 → 8.0 x 24.0 µm interior
(after PML cropping; dpml = 5/7 µm → pml_px = 10)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FDTD"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Target grid ──────────────────────────────────────────────────────────────
RESOLUTION   = 14
DPML         = 5.0 / 7.0                           # ≈0.714 µm
PML_PX       = round(DPML * RESOLUTION)             # 10 px
DPML_Q       = PML_PX / RESOLUTION                  # quantised
CROP_X_PX    = 336                                  # propagation (X)
CROP_Y_PX    = 112                                  # transverse  (Y)
CELL_X       = (CROP_X_PX + 2 * PML_PX) / RESOLUTION   # 25.43 µm
CELL_Y       = (CROP_Y_PX + 2 * PML_PX) / RESOLUTION   # 9.43  µm
INNER_X      = CROP_X_PX / RESOLUTION               # 24.0  µm
INNER_Y      = CROP_Y_PX / RESOLUTION               # 8.0   µm

print(f"Target grid : {CROP_Y_PX}×{CROP_X_PX} px  ({INNER_Y:.1f}×{INNER_X:.1f} µm interior)")
print(f"Cell        : {CELL_Y:.3f}×{CELL_X:.3f} µm  (with {DPML_Q:.4f} µm PML = {PML_PX} px)")
print()

# ── Parameter configs ────────────────────────────────────────────────────────
# For each device we define: min, mid, max, and one "interesting physics" case
DEVICE_CONFIGS = {
    # ── Already in training ──────────────────────────────────────────────────
    "sbend": {
        "params": {
            "min":  {"wg_width_um": 0.4286, "lateral_offset_um": 2.0,  "R_min_um": 3.0},
            "mid":  {"wg_width_um": 0.5000, "lateral_offset_um": 3.5,  "R_min_um": 5.0},
            "max":  {"wg_width_um": 0.5714, "lateral_offset_um": 5.5,  "R_min_um": 7.0},
            "tight": {"wg_width_um": 0.5714, "lateral_offset_um": 5.5, "R_min_um": 3.0},  # max offset, min radius = most loss
        },
        "builder": "sbend",
    },
    "ybranch": {
        "params": {
            "min":  {"wg_width_um": 0.4286, "l_junction_um": 1.1429, "l_bend_um": 4.0, "h_bend_um": 0.5714, "l_out_um": 1.0},
            "mid":  {"wg_width_um": 0.5000, "l_junction_um": 2.0,    "l_bend_um": 5.5, "h_bend_um": 1.5,    "l_out_um": 2.5},
            "max":  {"wg_width_um": 0.5714, "l_junction_um": 3.1429, "l_bend_um": 7.0, "h_bend_um": 2.5,    "l_out_um": 4.0},
            "wide_split": {"wg_width_um": 0.5714, "l_junction_um": 1.5, "l_bend_um": 5.0, "h_bend_um": 2.5, "l_out_um": 2.0},  # large Y, compact X
        },
        "builder": "ybranch",
    },
    "directional_coupler": {
        "params": {
            "min":  {"wg_width_um": 0.4286, "gap_um": 0.1071, "wg_length_um": 5.0, "bend_length_um": 4.0, "lead_extra_gap_um": 0.8214},
            "mid":  {"wg_width_um": 0.5000, "gap_um": 0.2000, "wg_length_um": 6.5, "bend_length_um": 5.0, "lead_extra_gap_um": 1.4},
            "max":  {"wg_width_um": 0.5714, "gap_um": 0.3571, "wg_length_um": 8.0, "bend_length_um": 6.0, "lead_extra_gap_um": 2.0},
            "strong_coupling": {"wg_width_um": 0.5000, "gap_um": 0.1071, "wg_length_um": 8.0, "bend_length_um": 5.0, "lead_extra_gap_um": 1.0},  # small gap, long coupling → near 100% crossover
        },
        "builder": "directional_coupler",
    },
    "mmi": {
        "params": {
            "min":  {"wg_width_um": 0.4286, "mmi_width_um": 2.5, "mmi_length_um": 8.0,  "taper_width_um": 0.5714, "taper_length_um": 1.0},
            "mid":  {"wg_width_um": 0.5000, "mmi_width_um": 4.0, "mmi_length_um": 11.0, "taper_width_um": 1.0,    "taper_length_um": 2.0},
            "max":  {"wg_width_um": 0.5714, "mmi_width_um": 5.5, "mmi_length_um": 15.0, "taper_width_um": 1.5,    "taper_length_um": 3.0},
            "3dB":  {"wg_width_um": 0.5000, "mmi_width_um": 4.0, "mmi_length_um": 10.0, "taper_width_um": 1.0,    "taper_length_um": 2.0},  # ~3dB splitting length
        },
        "builder": "mmi",
    },

    # ── NEW devices to add ───────────────────────────────────────────────────
    "euler_bend": {
        "params": {
            "min":  {"wg_width_um": 0.4286, "R_min_um": 3.0,  "lead_in_um": 2.0, "lead_out_um": 2.0},
            "mid":  {"wg_width_um": 0.5000, "R_min_um": 5.0,  "lead_in_um": 3.0, "lead_out_um": 3.0},
            "max":  {"wg_width_um": 0.5714, "R_min_um": 7.0,  "lead_in_um": 3.0, "lead_out_um": 3.0},
            "tight": {"wg_width_um": 0.5714, "R_min_um": 2.0, "lead_in_um": 2.0, "lead_out_um": 2.0},  # tightest bend, max radiation loss
        },
        "builder": "euler_bend",
    },
    "taper": {
        "params": {
            "min":  {"wg_width_in": 0.4286, "wg_width_out": 0.8,  "taper_length_um": 5.0},
            "mid":  {"wg_width_in": 0.5000, "wg_width_out": 1.5,  "taper_length_um": 10.0},
            "max":  {"wg_width_in": 0.5714, "wg_width_out": 2.5,  "taper_length_um": 18.0},
            "abrupt": {"wg_width_in": 0.4286, "wg_width_out": 2.5, "taper_length_um": 3.0},  # short+wide = strong scattering
        },
        "builder": "taper",
    },
    "crossing": {
        "params": {
            "min":  {"wg_width_h": 0.4286, "wg_width_v": 0.4286},
            "mid":  {"wg_width_h": 0.5000, "wg_width_v": 0.5000},
            "max":  {"wg_width_h": 0.5714, "wg_width_v": 0.5714},
            "asym": {"wg_width_h": 0.5714, "wg_width_v": 0.4286},  # asymmetric crossing
        },
        "builder": "crossing",
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
    from sbend.sbend import EulerSBend2D
    from ybranch.ybranch import YBranch2D
    from directional_coupler.directional_coupler import DirectionalCoupler2D
    from mmi.mmi import MMI2x2
    from euler_bend.euler_bend import EulerBend2D
    from taper.taper import TaperWaveguide2D
    from crossing.crossing import UniformCrossing2D

    common = {"wavelength_um": wavelength_um, "resolution": RESOLUTION}

    try:
        if device_type == "sbend":
            dev = EulerSBend2D(
                wg_width=params["wg_width_um"],
                lateral_offset_um=params["lateral_offset_um"],
                R_min_um=params["R_min_um"],
                dpml=DPML_Q, cell_x=CELL_X, cell_y=CELL_Y,
                **common,
            )
        elif device_type == "ybranch":
            dev = YBranch2D(
                wg_width_um=params["wg_width_um"],
                l_junction_um=params["l_junction_um"],
                l_bend_um=params["l_bend_um"],
                h_bend_um=params["h_bend_um"],
                l_out_um=params["l_out_um"],
                dpml=DPML_Q, cell_x_um=CELL_X, cell_y_um=CELL_Y,
                quantize_grid=False, fit_margin_um=0.5,
                **common,
            )
        elif device_type == "directional_coupler":
            dev = DirectionalCoupler2D(
                wg_width_um=params["wg_width_um"],
                gap_um=params["gap_um"],
                wg_length_um=params["wg_length_um"],
                bend_length_um=params["bend_length_um"],
                lead_extra_gap_um=params["lead_extra_gap_um"],
                dpml=DPML_Q, crop_x_px=CROP_X_PX, crop_y_px=CROP_Y_PX,
                quantize_grid=True, fit_margin_um=0.5,
                **common,
            )
        elif device_type == "mmi":
            dev = MMI2x2(
                wg_width_um=params["wg_width_um"],
                mmi_width_um=params["mmi_width_um"],
                mmi_length_um=params["mmi_length_um"],
                taper_width_um=params["taper_width_um"],
                taper_length_um=params["taper_length_um"],
                dpml=DPML_Q, crop_x_px=CROP_X_PX, crop_y_px=CROP_Y_PX,
                quantize_grid=True, fit_margin_um=0.5,
                **common,
            )
        elif device_type == "euler_bend":
            dev = EulerBend2D(
                wg_width=params["wg_width_um"],
                R_min_um=params["R_min_um"],
                bend_angle_deg=90.0,
                lead_in_um=params.get("lead_in_um", 3.0),
                lead_out_um=params.get("lead_out_um", 3.0),
                dpml=DPML_Q, cell_x=CELL_X, cell_y=CELL_Y,
                **common,
            )
        elif device_type == "taper":
            dev = TaperWaveguide2D(
                wg_width_in=params["wg_width_in"],
                wg_width_out=params["wg_width_out"],
                taper_length_um=params["taper_length_um"],
                dpml=DPML_Q, cell_x=CELL_X, cell_y=CELL_Y,
                **common,
            )
        elif device_type == "crossing":
            dev = UniformCrossing2D(
                wg_width_h=params["wg_width_h"],
                wg_width_v=params["wg_width_v"],
                dpml=DPML_Q, cell_x=CELL_X, cell_y=CELL_Y,
                **common,
            )
        else:
            raise ValueError(f"Unknown device: {device_type}")

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

    # ── Wavelength plan ──────────────────────────────────────────────────
    wavelengths_3 = [1.50, 1.55, 1.60]
    wavelengths_5 = [1.50, 1.525, 1.55, 1.575, 1.60]
    print("=== Wavelength Plan ===")
    print(f"  Current (3): {wavelengths_3}")
    print(f"  Proposed (5): {wavelengths_5}")
    print(f"  Δλ: {(wavelengths_5[1]-wavelengths_5[0])*1000:.1f} nm spacing")
    print()

    # ── Validate all devices ─────────────────────────────────────────────
    print(f"{'Device':<25s} {'Case':<18s} {'Shape':>12s} {'Expected':>12s} {'Status':<8s} {'Notes'}")
    print("─" * 100)

    expected_shape = (CROP_Y_PX, CROP_X_PX)
    results = {}  # {device_type: {case_name: (eps, params, error)}}

    for dev_name, cfg in DEVICE_CONFIGS.items():
        results[dev_name] = {}
        for case_name, params in cfg["params"].items():
            eps, err = build_device_epsilon(dev_name, params)
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
