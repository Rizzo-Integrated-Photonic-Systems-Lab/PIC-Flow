#!/usr/bin/env python3
"""
Test script comparing modal decomposition vs simple averaging for S-parameter extraction.

This script loads real simulation data from the unified_sweep dataset and compares:
1. Simple averaging method (current implementation)
2. Modal decomposition method (MEEP-style)
3. Ground truth S-parameters from FDTD simulation

Usage:
    python test_modal_sparams.py [--n-samples 100] [--device-types all]
"""

import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from collections import defaultdict
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modal_sparams import extract_sparams_modal, compare_sparam_methods
from sparams_loss import extract_sparams as extract_sparams_simple


def load_sample_from_shard(shard_path: str, slot: int):
    """Load a single sample from a shard file."""
    data = np.load(shard_path, allow_pickle=True)
    prefix = f"s{slot}/"

    def get(key):
        full_key = prefix + key
        if full_key in data:
            return data[full_key]
        return None

    # Load fields
    Ez_real = get("Ez_real")
    Ez_imag = get("Ez_imag")
    Ez = Ez_real + 1j * Ez_imag

    # Load permittivity
    eps = get("eps")

    # Load port masks
    port_masks = get("port_masks")

    # Load wavelength
    wavelength_um = float(get("wavelength_um"))

    # Load grid spacing
    dx_um = float(get("dx_um"))

    # Load input port
    input_port = int(get("input_port"))

    # Load port IDs
    port_ids = get("port_ids")

    # Load ground truth S-parameters
    # Try to find all S-params (S11, S21, S31, S41, etc.)
    sparams = []
    for i in range(1, 5):  # Up to 4 ports
        key_real = f"sparams/S{i}1_real"
        key_imag = f"sparams/S{i}1_imag"
        sr = get(key_real)
        si = get(key_imag)
        if sr is not None and si is not None:
            sparams.append(complex(float(sr), float(si)))
        else:
            break

    S_true = np.array(sparams, dtype=np.complex128)

    # Load device type
    device_type = str(get("device"))

    return {
        'Ez': Ez,
        'eps': eps,
        'port_masks': port_masks,
        'wavelength_um': wavelength_um,
        'dx_um': dx_um,
        'input_port': input_port,
        'port_ids': port_ids,
        'S_true': S_true,
        'device_type': device_type,
    }


def run_comparison_test(
    data_root: str = "Data/unified_sweep",
    n_samples: int = 100,
    device_types: list = None,
    verbose: bool = True,
):
    """
    Run comparison test on samples from the unified_sweep dataset.

    Args:
        data_root: Path to unified_sweep data
        n_samples: Number of samples to test
        device_types: List of device types to include (None = all)
        verbose: Print detailed results

    Returns:
        Dictionary with aggregate statistics
    """
    index_path = Path(data_root) / "shards" / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Filter by device type if specified
    if device_types:
        index = [e for e in index if e['device'] in device_types]

    # Sample randomly
    np.random.seed(42)
    if len(index) > n_samples:
        indices = np.random.choice(len(index), n_samples, replace=False)
        samples_to_test = [index[i] for i in indices]
    else:
        samples_to_test = index

    print(f"Testing {len(samples_to_test)} samples...")

    # Collect results by device type
    results_by_device = defaultdict(lambda: {
        'modal_mag_errors': [],
        'simple_mag_errors': [],
        'modal_phase_errors': [],
        'simple_phase_errors': [],
        'n_samples': 0,
    })

    device = torch.device('cpu')

    for i, entry in enumerate(samples_to_test):
        shard_path = Path(data_root) / "shards" / entry['shard']
        slot = entry['slot']
        device_type = entry['device']

        try:
            sample = load_sample_from_shard(str(shard_path), slot)
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
            continue

        # Convert to torch tensors
        Ez = torch.from_numpy(sample['Ez']).to(device)
        eps = torch.from_numpy(sample['eps']).float().to(device)
        port_masks = torch.from_numpy(sample['port_masks']).float().to(device)
        S_true = torch.from_numpy(sample['S_true']).to(device)
        wavelength_um = sample['wavelength_um']
        dx_um = sample['dx_um']
        port_ids = torch.from_numpy(sample['port_ids']).to(device) if sample['port_ids'] is not None else None

        # Input port index (convert from 1-based to 0-based)
        in_port_idx = sample['input_port'] - 1
        n_ports = len(sample['S_true'])

        try:
            # Compare methods
            comparison = compare_sparam_methods(
                Ez, port_masks[:n_ports], eps, wavelength_um, dx_um,
                S_true, in_port_idx=in_port_idx, port_ids=port_ids
            )

            # Aggregate errors (only for valid ports)
            for p in range(n_ports):
                mag_true = torch.abs(S_true[p]).item()
                if mag_true > 0.01:  # Only include significant ports
                    results_by_device[device_type]['modal_mag_errors'].append(
                        comparison['error_modal_mag'][p].item()
                    )
                    results_by_device[device_type]['simple_mag_errors'].append(
                        comparison['error_simple_mag'][p].item()
                    )
                    results_by_device[device_type]['modal_phase_errors'].append(
                        abs(comparison['error_modal_phase'][p].item())
                    )
                    results_by_device[device_type]['simple_phase_errors'].append(
                        abs(comparison['error_simple_phase'][p].item())
                    )

            results_by_device[device_type]['n_samples'] += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(samples_to_test)} samples...")

        except Exception as e:
            print(f"  Error processing sample {i} ({device_type}): {e}")
            continue

    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("RESULTS: S-Parameter Extraction Method Comparison")
    print("=" * 70)

    all_modal_mag = []
    all_simple_mag = []
    all_modal_phase = []
    all_simple_phase = []

    for device_type, results in sorted(results_by_device.items()):
        if results['n_samples'] == 0:
            continue

        modal_mag = np.array(results['modal_mag_errors'])
        simple_mag = np.array(results['simple_mag_errors'])
        modal_phase = np.array(results['modal_phase_errors'])
        simple_phase = np.array(results['simple_phase_errors'])

        all_modal_mag.extend(modal_mag)
        all_simple_mag.extend(simple_mag)
        all_modal_phase.extend(modal_phase)
        all_simple_phase.extend(simple_phase)

        print(f"\n{device_type.upper()} ({results['n_samples']} samples, {len(modal_mag)} S-param values)")
        print("-" * 50)
        print(f"  Magnitude Error (|S| difference):")
        print(f"    Modal:  mean={np.mean(modal_mag):.4f}, std={np.std(modal_mag):.4f}, max={np.max(modal_mag):.4f}")
        print(f"    Simple: mean={np.mean(simple_mag):.4f}, std={np.std(simple_mag):.4f}, max={np.max(simple_mag):.4f}")
        print(f"  Phase Error (degrees):")
        print(f"    Modal:  mean={np.mean(modal_phase):.2f}, std={np.std(modal_phase):.2f}, max={np.max(modal_phase):.2f}")
        print(f"    Simple: mean={np.mean(simple_phase):.2f}, std={np.std(simple_phase):.2f}, max={np.max(simple_phase):.2f}")

        # Which is better?
        modal_better_mag = np.mean(modal_mag) < np.mean(simple_mag)
        modal_better_phase = np.mean(modal_phase) < np.mean(simple_phase)
        print(f"  Better method: {'Modal' if modal_better_mag else 'Simple'} (mag), {'Modal' if modal_better_phase else 'Simple'} (phase)")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    all_modal_mag = np.array(all_modal_mag)
    all_simple_mag = np.array(all_simple_mag)
    all_modal_phase = np.array(all_modal_phase)
    all_simple_phase = np.array(all_simple_phase)

    print(f"\nTotal S-parameter values compared: {len(all_modal_mag)}")
    print(f"\nMagnitude Error:")
    print(f"  Modal:  mean={np.mean(all_modal_mag):.4f}, std={np.std(all_modal_mag):.4f}")
    print(f"  Simple: mean={np.mean(all_simple_mag):.4f}, std={np.std(all_simple_mag):.4f}")
    print(f"  Improvement: {(np.mean(all_simple_mag) - np.mean(all_modal_mag)) / np.mean(all_simple_mag) * 100:.1f}% reduction with modal method")

    print(f"\nPhase Error (degrees):")
    print(f"  Modal:  mean={np.mean(all_modal_phase):.2f}, std={np.std(all_modal_phase):.2f}")
    print(f"  Simple: mean={np.mean(all_simple_phase):.2f}, std={np.std(all_simple_phase):.2f}")
    print(f"  Improvement: {(np.mean(all_simple_phase) - np.mean(all_modal_phase)) / np.mean(all_simple_phase) * 100:.1f}% reduction with modal method")

    return {
        'by_device': dict(results_by_device),
        'overall': {
            'modal_mag_mean': np.mean(all_modal_mag),
            'simple_mag_mean': np.mean(all_simple_mag),
            'modal_phase_mean': np.mean(all_modal_phase),
            'simple_phase_mean': np.mean(all_simple_phase),
        }
    }


def run_detailed_example(data_root: str = "Data/unified_sweep"):
    """Run a detailed example on a single sample and print values."""
    index_path = Path(data_root) / "shards" / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Find a directional coupler (4 ports) for a good test case
    dc_entries = [e for e in index if e['device'] == 'directional_coupler']
    if not dc_entries:
        dc_entries = [e for e in index if e['device'] == 'ybranch']
    if not dc_entries:
        dc_entries = index[:1]

    entry = dc_entries[0]
    shard_path = Path(data_root) / "shards" / entry['shard']

    print(f"\nDetailed Example: {entry['device']}")
    print(f"  Shard: {entry['shard']}, Slot: {entry['slot']}")
    print(f"  Wavelength: {entry['wavelength_um']} um")
    print("-" * 50)

    sample = load_sample_from_shard(str(shard_path), entry['slot'])

    device = torch.device('cpu')
    Ez = torch.from_numpy(sample['Ez']).to(device)
    eps = torch.from_numpy(sample['eps']).float().to(device)
    port_masks = torch.from_numpy(sample['port_masks']).float().to(device)
    S_true = torch.from_numpy(sample['S_true']).to(device)
    wavelength_um = sample['wavelength_um']
    dx_um = sample['dx_um']
    port_ids = torch.from_numpy(sample['port_ids']).to(device) if sample['port_ids'] is not None else None
    in_port_idx = sample['input_port'] - 1
    n_ports = len(sample['S_true'])

    print(f"  Number of ports: {n_ports}")
    print(f"  Input port: {sample['input_port']} (index {in_port_idx})")
    print(f"  Port IDs: {sample['port_ids']}")
    print(f"  Grid spacing: {dx_um:.4f} um")
    print(f"  Field shape: {sample['Ez'].shape}")

    # Extract S-parameters with both methods
    S_modal = extract_sparams_modal(
        Ez, port_masks[:n_ports], eps, wavelength_um, dx_um,
        in_port_idx=in_port_idx, port_ids=port_ids
    )

    # Simple method expects batched input [B,H,W]
    Ez_batched = Ez.unsqueeze(0) if Ez.dim() == 2 else Ez
    S_simple = extract_sparams_simple(
        Ez_batched, port_masks[:n_ports],
        in_port_idx=in_port_idx, port_ids=port_ids
    )
    # Squeeze batch dimension
    if S_simple.dim() == 2 and S_simple.shape[0] == 1:
        S_simple = S_simple.squeeze(0)

    print(f"\nS-Parameter Comparison:")
    print(f"{'Port':<6} {'S_true (FDTD)':<25} {'S_modal':<25} {'S_simple':<25}")
    print("-" * 85)

    for p in range(n_ports):
        s_t = S_true[p].item() if hasattr(S_true[p], 'item') else complex(S_true[p])
        s_m = S_modal[p].item() if hasattr(S_modal[p], 'item') else complex(S_modal[p])
        s_s = S_simple[p].item() if hasattr(S_simple[p], 'item') else complex(S_simple[p])

        mag_t, ph_t = abs(s_t), np.angle(s_t) * 180 / np.pi
        mag_m, ph_m = abs(s_m), np.angle(s_m) * 180 / np.pi
        mag_s, ph_s = abs(s_s), np.angle(s_s) * 180 / np.pi

        print(f"S{p+1}1    {mag_t:.4f}∠{ph_t:+6.1f}°          "
              f"{mag_m:.4f}∠{ph_m:+6.1f}°          "
              f"{mag_s:.4f}∠{ph_s:+6.1f}°")

    print("\nErrors:")
    print(f"{'Port':<6} {'Modal |S| err':<15} {'Simple |S| err':<15} {'Modal ∠ err':<15} {'Simple ∠ err':<15}")
    print("-" * 70)

    for p in range(n_ports):
        s_t = S_true[p].item() if hasattr(S_true[p], 'item') else complex(S_true[p])
        s_m = S_modal[p].item() if hasattr(S_modal[p], 'item') else complex(S_modal[p])
        s_s = S_simple[p].item() if hasattr(S_simple[p], 'item') else complex(S_simple[p])

        mag_t = abs(s_t)
        if mag_t < 0.01:
            continue

        err_m_mag = abs(abs(s_m) - mag_t)
        err_s_mag = abs(abs(s_s) - mag_t)

        ph_t = np.angle(s_t)
        ph_m = np.angle(s_m)
        ph_s = np.angle(s_s)

        err_m_ph = abs(np.arctan2(np.sin(ph_m - ph_t), np.cos(ph_m - ph_t))) * 180 / np.pi
        err_s_ph = abs(np.arctan2(np.sin(ph_s - ph_t), np.cos(ph_s - ph_t))) * 180 / np.pi

        print(f"S{p+1}1    {err_m_mag:.4f}          {err_s_mag:.4f}          "
              f"{err_m_ph:.1f}°          {err_s_ph:.1f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare S-parameter extraction methods")
    parser.add_argument("--data-root", type=str, default="Data/unified_sweep",
                        help="Path to unified_sweep data")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to test")
    parser.add_argument("--device-types", type=str, default="",
                        help="Comma-separated device types to test (empty=all)")
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed example on single sample")
    args = parser.parse_args()

    device_types = [d.strip() for d in args.device_types.split(",") if d.strip()] or None

    if args.detailed:
        run_detailed_example(args.data_root)
    else:
        run_comparison_test(
            data_root=args.data_root,
            n_samples=args.n_samples,
            device_types=device_types,
        )
