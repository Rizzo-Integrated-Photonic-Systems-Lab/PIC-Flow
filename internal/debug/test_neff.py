#!/usr/bin/env python3
"""
Simple test script to verify neff computation works
"""
import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

# Test a single computation
def test_single_computation():
    print("Testing single neff computation...")

    # Simple test case
    lam_um = 1.55
    wg_width_um = 0.5

    print(f"Computing n_eff for width={wg_width_um} µm at lambda={lam_um} µm")

    # Tidy3D units: µm and ps, so C_0 is in µm/ps and freq in 1/ps
    freq = td.C_0 / lam_um  # 1/ps

    # Simple geometry
    si = td.material_library["cSi"]["Palik_Lossless"]
    sio2 = td.material_library["SiO2"]["Palik_Lossless"]
    air = td.Medium(permittivity=1.0)

    # Simple simulation box
    sim_size = (2.0, 1.0, 1.0)
    grid_spec = td.GridSpec.uniform(dl=0.01)

    # Structures
    waveguide = td.Structure(
        geometry=td.Box(size=(wg_width_um, td.inf, 0.22), center=(0.0, 0.0, 0.11)),
        medium=si,
    )

    substrate = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 2.0), center=(0.0, 0.0, -1.0)),
        medium=sio2,
    )

    sim = td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[waveguide, substrate],
        medium=air,
        run_time=1e-12,
    )

    # Mode plane
    plane = td.Box(center=(0.0, 0.0, 0.0), size=(2.0, 0.0, 1.0))

    mode_spec = td.ModeSpec(num_modes=4, target_neff=2.4)

    ms = ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=mode_spec,
        freqs=[freq],
    )

    try:
        print("Running local mode solve...")
        mode_data = ms.solve()
        print("Local solve successful!")
        neffs_all = mode_data.n_eff.values[0, :].real

        # Simple selection
        n_clad_est = 1.44
        n_core_est = 3.6
        mask = (neffs_all > n_clad_est + 0.05) & (neffs_all < n_core_est - 0.05)

        if np.any(mask):
            neff_val = neffs_all[mask].max()
        else:
            neff_val = neffs_all.max()

        print(f"Success! n_eff = {neff_val:.6f}")
        return True

    except Exception as e:
        print(f"Local solve failed: {e}")
        return False

if __name__ == "__main__":
    success = test_single_computation()
    if success:
        print("Basic functionality test passed!")
    else:
        print("Basic functionality test failed!")
