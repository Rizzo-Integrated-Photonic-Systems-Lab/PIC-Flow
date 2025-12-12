import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver


# --- Geometry constants (µm) ---
WG_HEIGHT_UM   = 0.22   # silicon thickness
SUBSTRATE_UM   = 2.0    # SiO2 below
AIR_TOP_UM     = 2.0    # air above
LATERAL_PAD_UM = 2.0    # lateral padding around core

# Design wavelength for coupler simulations (µm)
TARGET_LAMBDA_UM = 1.55


def compute_neff_for_lambda(lam_um: float, wg_width_um: float) -> float:
    """
    Return n_eff of the fundamental TE-like guided mode at wavelength lam_um (µm)
    for a given waveguide width wg_width_um (in µm).

    Geometry: 220 nm × wg_width_um Si wire on SiO2, air above.
    Propagation direction is y; we solve on an x–z cross-section at y = 0.

    The function asks Tidy3D for several modes, then selects the most
    guided (largest real(n_eff) between approximate n_clad and n_core).
    """

    # Tidy3D units: µm and ps, so C_0 is in µm/ps and freq in 1/ps
    freq = td.C_0 / lam_um  # 1/ps

    # --- simulation box size (µm) ---
    # Make cross-section reasonably large so the mode decays before boundaries.
    Lx_core = wg_width_um + 2 * LATERAL_PAD_UM      # core + padding
    Lx = max(Lx_core, 5.0)                          # ensure at least ~5 µm laterally
    Ly = 3.0                                        # "length" in propagation direction (not used by modes)
    Lz = SUBSTRATE_UM + WG_HEIGHT_UM + AIR_TOP_UM
    sim_size = (Lx, Ly, Lz)

    # --- materials ---
    si   = td.material_library["cSi"]["Palik_Lossless"]
    sio2 = td.material_library["SiO2"]["Palik_Lossless"]
    air  = td.Medium(permittivity=1.0)

    # --- structures ---

    # silicon wire: finite width in x, infinite along y (prop), finite in z,
    # sitting on z = 0 (top of substrate)
    waveguide = td.Structure(
        geometry=td.Box(
            size=(wg_width_um, td.inf, WG_HEIGHT_UM),      # (x, y, z)
            center=(0.0, 0.0, 0.5 * WG_HEIGHT_UM),         # spans z in [0, WG_HEIGHT_UM]
        ),
        medium=si,
    )

    # SiO2 substrate: half-space below z = 0 (approx)
    substrate = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, SUBSTRATE_UM),
            center=(0.0, 0.0, -0.5 * SUBSTRATE_UM),        # spans z in [-SUBSTRATE_UM, 0]
        ),
        medium=sio2,
    )

    # --- grid spec: fixed, fine uniform grid ---
    dl = 0.01  # 10 nm resolution in all directions
    grid_spec = td.GridSpec.uniform(dl=dl)

    # --- Simulation container (used only by ModeSolver) ---
    sim = td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[waveguide, substrate],
        medium=air,
        run_time=1e-12,  # irrelevant for ModeSolver, but required
    )

    # Mode plane: x–z cross-section at y = 0
    plane = td.Box(
        center=(0.0, 0.0, 0.0),
        size=(Lx, 0.0, Lz),
    )

    # Ask for several modes; target_neff just steers toward the Si guided mode
    mode_spec = td.ModeSpec(
        num_modes=4,          # find up to 4 modes
        target_neff=2.4,      # typical for Si wire around 1.55 µm
    )

    ms = ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=mode_spec,
        freqs=[freq],
    )

    # Local solve (no cloud / stub files) for robustness on this cluster.
    # If you later want to use the Tidy3D cloud mode solver, replace this with
    # tidy3d.web.run(ms, ...) and make sure the API key + stub handling work
    # with your installed tidy3d version.
    mode_data = ms.solve()

    # mode_data.n_eff is an xarray DataArray with dims ('f', 'mode_index')
    neffs_all = mode_data.n_eff.values[0, :].real  # shape (num_modes,)

    # Heuristic filter: guided Si mode should lie between n_clad and n_core.
    # Use approximate bounds just for selection.
    n_clad_est = 1.44
    n_core_est = 3.6

    mask = (neffs_all > n_clad_est + 0.05) & (neffs_all < n_core_est - 0.05)

    if np.any(mask):
        # among plausible guided modes, pick the one with the largest n_eff
        neff_val = neffs_all[mask].max()
    else:
        # fallback: pick the mode with the largest real n_eff
        neff_val = neffs_all.max()

    return float(neff_val)


def sweep_widths_and_save_neff_csvs(
    widths_um: np.ndarray,
    out_dir: str = "neff_tables",
    lambda_min_um: float = 1.40,
    lambda_max_um: float = 1.60,
    lambda_step_um: float = 0.01,
) -> None:
    """
    Sweep over a list of waveguide widths and wavelengths and save n_eff(λ) tables.

    For each width, we sweep λ from `lambda_min_um` to `lambda_max_um` (inclusive)
    in steps of `lambda_step_um` and compute the effective index of the
    fundamental TE-like guided mode.

    Each width w produces a CSV like:
        {out_dir}/neff_siwire_w{w_code}_t0220.csv
    where w_code is e.g. 038, 045, 060 for 0.38, 0.45, 0.60 µm.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Construct wavelength grid once and reuse for all widths
    lambda_grid = np.arange(lambda_min_um, lambda_max_um + 1e-9, lambda_step_um)

    for wg_width in widths_um:
        # round to 2 decimal places to avoid 0.379999... artifacts
        wg_width_rounded = float(np.round(wg_width, 2))
        w_code = int(round(wg_width_rounded * 100))  # 0.45 -> 45
        w_str = f"{w_code:03d}"                      # 45 -> "045"

        csv_name = f"neff_siwire_w{w_str}_t0220.csv"
        csv_path = os.path.join(out_dir, csv_name)

        print(
            f"\n=== Computing n_eff over λ ∈ "
            f"[{lambda_min_um:.2f}, {lambda_max_um:.2f}] µm "
            f"in steps of {lambda_step_um:.3f} µm "
            f"for wg_width = {wg_width_rounded:.2f} µm ==="
        )

        neff_vals = []
        for lam in lambda_grid:
            neff = compute_neff_for_lambda(lam_um=float(lam), wg_width_um=wg_width_rounded)
            neff_vals.append(neff)
            print(f"λ = {lam:.4f} µm  ->  n_eff = {neff:.6f}")

        # Save a multi-row CSV per width: one row per wavelength.
        df = pd.DataFrame(
            {
                "lambda_um": lambda_grid,
                "neff": neff_vals,
            }
        )
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    # Sweep wg_width from 0.38 to 0.60 µm in 0.01 µm steps
    widths_um = np.arange(0.38, 0.60 + 1e-9, 0.01)

    # Compute n_eff vs wavelength for each width, sweeping λ from 1.40 to 1.60 µm.
    sweep_widths_and_save_neff_csvs(
        widths_um=widths_um,
        lambda_min_um=1.40,
        lambda_max_um=1.60,
        lambda_step_um=0.01,
    )

    # Optionally, you can uncomment the block below to quickly visualize one
    # particular width (e.g. 0.45 µm) after the sweep by loading its CSV and
    # fitting a simple polynomial, as in the original script.
    #
    # import pandas as pd
    #
    # sample_width_um = 0.45
    # w_code = int(round(sample_width_um * 100))
    # w_str = f"{w_code:03d}"
    # csv_path = os.path.join("neff_tables", f"neff_siwire_w{w_str}_t0220.csv")
    #
    # df = pd.read_csv(csv_path)
    # lam = df["lambda_um"].values
    # neff = df["neff"].values
    #
    # order = np.argsort(lam)
    # lam = lam[order]
    # neff = neff[order]
    #
    # deg = 2
    # coeffs = np.polyfit(lam, neff, deg=deg)
    # lam_fit = np.linspace(lam.min(), lam.max(), 400)
    # neff_fit_vals = np.polyval(coeffs, lam_fit)
    #
    # plt.figure(figsize=(7, 5))
    # plt.scatter(lam, neff, s=20, label="Raw data", alpha=0.7)
    # plt.plot(lam_fit, neff_fit_vals, label=f"Poly fit (deg={deg})", linewidth=2)
    # plt.xlabel("Wavelength λ (µm)")
    # plt.ylabel("n_eff")
    # plt.title(f"Effective index vs wavelength for {sample_width_um:.2f} µm × 220 nm Si wire")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

