#!/usr/bin/env python3
"""
Quick script to compute Helmholtz residual on ground truth FDTD data.
This tells us the "floor" for sample_residual_mean.

Works with both:
- Preprocessed .pt format (fast dataset)
- NPZ shard format (original dataset)
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/dartfs-hpc/rc/home/j/f0071mj/rayfield/Model")
from physics_unet import HelmholtzResidual2D


def load_preprocessed_samples(data_root: str, n_samples: int, stats: dict):
    """Load samples from preprocessed .pt format."""
    root = Path(data_root)
    index_path = root / "index.json"

    with open(index_path) as f:
        entries = json.load(f)

    samples = []
    for i, entry in enumerate(entries[:n_samples]):
        pt_path = root / entry["file"]
        sample = torch.load(pt_path, weights_only=False)

        # Denormalize fields (they're stored normalized in .pt files)
        ez_r = sample["ez_real"] * stats["ez_real_std"] + stats["ez_real_mean"]
        ez_i = sample["ez_imag"] * stats["ez_imag_std"] + stats["ez_imag_mean"]
        eps = sample["eps"] * stats["eps_std"] + stats["eps_mean"]
        lambda_um = float(sample["lambda_um"])

        samples.append({
            "ez_real": ez_r,
            "ez_imag": ez_i,
            "eps": eps,
            "lambda_um": lambda_um,
        })

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to preprocessed data dir")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to check")
    parser.add_argument("--dx", type=float, default=1.0/24.0, help="Grid spacing in um")
    args = parser.parse_args()

    root = Path(args.data_root)

    # Load stats
    stats_path = root / "stats.json"
    if not stats_path.is_file():
        print(f"Error: stats.json not found at {stats_path}")
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    print(f"Loading samples from {args.data_root}...")
    samples = load_preprocessed_samples(args.data_root, args.n_samples, stats)

    # Create Helmholtz operator
    dx = args.dx
    dy = args.dx
    # We'll use per-sample k0, so omega here is just a placeholder
    helmholtz_op = HelmholtzResidual2D(dx=dx, dy=dy, omega=1.0, c0=1.0, pml_cells=0)

    print(f"\nComputing Helmholtz residual on {len(samples)} GT samples...")
    print(f"  dx = {dx:.6f} um")
    print()

    residuals = []
    field_mags = []

    for i, sample in enumerate(samples):
        ez_r = sample["ez_real"].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        ez_i = sample["ez_imag"].unsqueeze(0).unsqueeze(0)
        eps = sample["eps"].unsqueeze(0).unsqueeze(0)
        lambda_um = sample["lambda_um"]

        k0 = 2.0 * np.pi / lambda_um

        # Stack fields
        fields = torch.cat([ez_r, ez_i], dim=1).float()  # [1, 2, H, W]
        eps = eps.float()

        # Compute Helmholtz residual
        with torch.no_grad():
            R = helmholtz_op(fields, eps, k0=torch.tensor([k0]))
            res_mag = torch.sqrt(R[:, 0:1]**2 + R[:, 1:2]**2 + 1e-12) * (dx**2)
            residuals.append(float(res_mag.mean().item()))

            # Also compute field magnitude for reference
            field_mag = torch.sqrt(ez_r**2 + ez_i**2 + 1e-12)
            field_mags.append(float(field_mag.mean().item()))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    residuals = np.array(residuals)
    field_mags = np.array(field_mags)

    print("\n" + "="*60)
    print("GROUND TRUTH HELMHOLTZ RESIDUAL STATISTICS")
    print("="*60)
    print(f"  Samples analyzed: {len(samples)}")
    print()
    print(f"  Residual (same metric as sample_residual_mean):")
    print(f"    Mean:   {residuals.mean():.6f}")
    print(f"    Std:    {residuals.std():.6f}")
    print(f"    Min:    {residuals.min():.6f}")
    print(f"    Max:    {residuals.max():.6f}")
    print(f"    Median: {np.median(residuals):.6f}")
    print()
    print(f"  Field magnitude |Ez|:")
    print(f"    Mean:   {field_mags.mean():.4f}")
    print(f"    Std:    {field_mags.std():.4f}")
    print(f"    Min:    {field_mags.min():.4f}")
    print(f"    Max:    {field_mags.max():.4f}")
    print()
    print(f"  Relative residual (residual / field_mag):")
    rel_res = residuals / (field_mags + 1e-12)
    print(f"    Mean:   {rel_res.mean():.6f} ({rel_res.mean()*100:.2f}%)")
    print(f"    Std:    {rel_res.std():.6f}")
    print()
    print("="*60)
    print("INTERPRETATION:")
    print("="*60)
    print(f"  Your model's sample_residual_mean: ~0.5")
    print(f"  Ground truth residual mean:        {residuals.mean():.4f}")
    gap = 0.5 - residuals.mean()
    if gap > 0.3:
        print(f"\n  -> GT residual is MUCH lower than model's 0.5")
        print(f"     Gap of ~{gap:.2f} suggests significant room for improvement")
        print(f"     in physics consistency.")
    elif gap > 0.1:
        print(f"\n  -> GT residual is lower than model's 0.5")
        print(f"     Gap of ~{gap:.2f} - some room for improvement.")
    else:
        print(f"\n  -> GT residual is similar to model's 0.5")
        print(f"     Model may be near the physics floor for this discretization.")
    print("="*60)


if __name__ == "__main__":
    main()
