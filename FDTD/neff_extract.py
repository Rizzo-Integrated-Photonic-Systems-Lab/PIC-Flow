import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    # Use local solver
    try:
        logger.debug(f"Running local mode solve for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm")
    mode_data = ms.solve()
        logger.debug(f"Local mode solve completed for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm")
    except Exception as e:
        logger.error(f"Local mode solve failed for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm: {e}")
        raise RuntimeError(f"Mode solving failed for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm")

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


def compute_neff_for_width_lambda(args):
    """
    Compute n_eff for a single width-wavelength pair.
    Returns (width_um, lambda_um, neff)
    """
    wg_width_um, lam_um = args
    try:
        logger.debug(f"Computing n_eff for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm")
        neff = compute_neff_for_lambda(lam_um=lam_um, wg_width_um=wg_width_um)
        logger.debug(f"Completed n_eff={neff:.6f} for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm")
        return wg_width_um, lam_um, neff
    except Exception as e:
        logger.error(f"Error computing n_eff for width={wg_width_um:.3f} µm, lambda={lam_um:.4f} µm: {e}")
        raise


def sweep_widths_and_save_neff_csvs(
    widths_um: np.ndarray,
    out_dir: str = "neff_tables",
    lambda_min_um: float = 1.45,
    lambda_max_um: float = 1.65,
    lambda_step_um: float = 0.01,
    num_processes: int = None,
) -> None:
    """
    Sweep over a list of waveguide widths and wavelengths and save n_eff(λ) tables.

    For each width, we sweep λ from `lambda_min_um` to `lambda_max_um` (inclusive)
    in steps of `lambda_step_um` and compute the effective index of the
    fundamental TE-like guided mode.

    Each width w produces a CSV like:
        {out_dir}/neff_siwire_w{w_code}_t0220.csv
    where w_code is e.g. 038, 045, 060 for 0.38, 0.45, 0.60 µm.

    Parameters:
        num_processes: Number of parallel processes. If None, uses CPU count.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Construct wavelength grid once and reuse for all widths
    lambda_grid = np.arange(lambda_min_um, lambda_max_um + 1e-9, lambda_step_um)

    # Create all width-wavelength combinations
    logger.info("Creating width-wavelength combinations...")
    width_lambda_pairs = []
    for wg_width in widths_um:
        wg_width_rounded = float(np.round(wg_width, 2))
        for lam in lambda_grid:
            width_lambda_pairs.append((wg_width_rounded, float(lam)))

    total_computations = len(width_lambda_pairs)
    logger.info(f"Total computations: {total_computations}")
    logger.info(f"Width range: {widths_um[0]:.3f} to {widths_um[-1]:.3f} µm ({len(widths_um)} widths)")
    logger.info(f"Wavelength range: {lambda_grid[0]:.4f} to {lambda_grid[-1]:.4f} µm ({len(lambda_grid)} wavelengths)")

    # Use parallel processing
    if num_processes is None:
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} parallel processes")

    results = {}
    completed = 0
    failed = 0

    logger.info("Starting parallel computation...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all jobs
        logger.info(f"Submitting {total_computations} jobs to process pool...")
        future_to_params = {
            executor.submit(compute_neff_for_width_lambda, params): params
            for params in width_lambda_pairs
        }
        logger.info("All jobs submitted, waiting for completion...")

        # Process completed jobs
        for future in as_completed(future_to_params):
            try:
                wg_width, lam_um, neff = future.result()
                if wg_width not in results:
                    results[wg_width] = {}
                results[wg_width][lam_um] = neff

                completed += 1
                if completed % 50 == 0 or completed == total_computations:
                    logger.info(f"Completed {completed}/{total_computations} computations")
            except Exception as e:
                failed += 1
                params = future_to_params[future]
                logger.error(f"Failed computation for width={params[0]:.3f} µm, lambda={params[1]:.4f} µm: {e}")

    logger.info(f"Parallel computation finished. Completed: {completed}, Failed: {failed}")
    logger.info(f"Results collected for {len(results)} widths")

    if len(results) == 0:
        logger.error("No results were collected! All computations may have failed.")
        logger.error("Check Tidy3D configuration and try running with fewer processes or a single test case.")
        return

    # Save results for each width
    logger.info(f"Saving results to {out_dir}/ directory...")
    saved_count = 0
    for wg_width in sorted(results.keys()):
        w_code = int(round(wg_width * 100))
        w_str = f"{w_code:03d}"
        csv_name = f"neff_siwire_w{w_str}_t0220.csv"
        csv_path = os.path.join(out_dir, csv_name)

        # Sort by wavelength for consistent output
        width_results = results[wg_width]
        sorted_lams = sorted(width_results.keys())
        sorted_neffs = [width_results[lam] for lam in sorted_lams]

        logger.info(f"Saving results for wg_width = {wg_width:.2f} µm ({len(sorted_lams)} data points) to {csv_path}")

        try:
        df = pd.DataFrame(
            {
                    "lambda_um": sorted_lams,
                    "neff": sorted_neffs,
            }
        )
        df.to_csv(csv_path, index=False)
            saved_count += 1
            logger.info(f"Successfully saved {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save {csv_path}: {e}")

    logger.info(f"Saved {saved_count} CSV files to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute effective indices for silicon waveguides")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    parser.add_argument(
        "--width-min",
        type=float,
        default=0.38,
        help="Minimum waveguide width in µm (default: 0.38)"
    )
    parser.add_argument(
        "--width-max",
        type=float,
        default=2.00,
        help="Maximum waveguide width in µm (default: 2.00)"
    )
    parser.add_argument(
        "--width-step",
        type=float,
        default=0.01,
        help="Waveguide width step in µm (default: 0.01)"
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1.45,
        help="Minimum wavelength in µm (default: 1.45)"
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1.65,
        help="Maximum wavelength in µm (default: 1.65)"
    )
    parser.add_argument(
        "--lambda-step",
        type=float,
        default=0.01,
        help="Wavelength step in µm (default: 0.01)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="neff_tables",
        help="Output directory for CSV files (default: neff_tables)"
    )

    args = parser.parse_args()

    logger.info("Starting neff_extract.py")
    logger.info(f"Arguments: num_processes={args.num_processes}, width_min={args.width_min}, "
                f"width_max={args.width_max}, width_step={args.width_step}")
    logger.info(f"Wavelength range: {args.lambda_min} to {args.lambda_max} µm (step: {args.lambda_step})")
    logger.info(f"Output directory: {args.out_dir}")

    # Sweep wg_width from args.width_min to args.width_max in args.width_step µm steps
    widths_um = np.arange(args.width_min, args.width_max + 1e-9, args.width_step)

    logger.info(f"Computing n_eff for {len(widths_um)} widths from {args.width_min:.2f} to {args.width_max:.2f} µm")

    try:
        # Compute n_eff vs wavelength for each width
    sweep_widths_and_save_neff_csvs(
        widths_um=widths_um,
            out_dir=args.out_dir,
            lambda_min_um=args.lambda_min,
            lambda_max_um=args.lambda_max,
            lambda_step_um=args.lambda_step,
            num_processes=args.num_processes,
    )
        logger.info("neff_extract.py completed successfully")
    except Exception as e:
        logger.error(f"neff_extract.py failed with error: {e}")
        sys.exit(1)

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

