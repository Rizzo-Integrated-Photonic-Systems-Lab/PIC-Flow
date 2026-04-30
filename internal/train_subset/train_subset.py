#!/usr/bin/env python3
"""
Training script for Rayfield on a filtered 5-device dataset.

Devices: straight, sbend, directional_coupler, ybranch, euler_bend
Losses: Flow Matching + Residual + Endpoint + S-Parameter

Usage:
    python scripts/train_subset.py [--dry-run]
    python scripts/train_subset.py --epochs 500 --batch-size 16
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

DEVICES = ["straight", "sbend", "directional_coupler", "ybranch", "euler_bend"]

# ~5000 total samples, 90/10 train/val split
# Straight waveguides are simplest -> smallest portion
TRAIN_SAMPLES_PER_DEVICE = {
    "straight": 270,
    "sbend": 630,
    "directional_coupler": 1620,
    "ybranch": 1260,
    "euler_bend": 720,
}

VAL_SAMPLES_PER_DEVICE = {
    "straight": 30,
    "sbend": 70,
    "directional_coupler": 180,
    "ybranch": 140,
    "euler_bend": 80,
}

# ==============================================================================
# TARGET SPECIFICATIONS
# ==============================================================================

TARGET_SPECS = {
    "fm_loss": {"target": 0.01, "unit": "", "desc": "Flow matching MSE"},
    "residual_loss": {"target": 1e-4, "unit": "", "desc": "Helmholtz residual"},
    "endpoint_loss": {"target": 0.05, "unit": "", "desc": "Endpoint prediction error"},
    "sparam_mag_error": {"target": 0.02, "unit": "", "desc": "|S| magnitude error"},
    "sparam_phase_error": {"target": 5.0, "unit": "°", "desc": "S-param phase error"},
    "sample_residual": {"target": 1e-3, "unit": "", "desc": "Sampled field residual"},
}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

TRAINING_CONFIG = {
    # Model
    "hidden_size": 128,
    "complex_unet": True,

    # Losses (FM + Residual + Endpoint + S-param, NO phase)
    "lambda_residual": 1.0,
    "lambda_endpoint": 0.5,
    "lambda_sparam": 0.5,
    "lambda_phase": 0.0,        # DISABLED
    "lambda_phase_grad": 0.0,   # DISABLED

    # Curriculum
    "phaseA_epochs": 25,        # FM only
    "phaseB_epochs": 100,       # FM + Residual
    # Phase C: FM + Residual + Endpoint + S-param

    # Warmup
    "residual_warmup_epochs": 25,
    "endpoint_warmup_epochs": 25,
    "sparam_warmup_epochs": 50,

    # S-param settings
    "sparam_mode": "modal",
    "sparam_every": 1,
    "sparam_from_start": False,

    # Unrolling (PBFM temporal integration)
    "unroll_steps": 4,      # integrate ODE t→1 with 4 Heun steps (~8 extra fwd passes)
    "unroll_phase": False,  # use unrolled field for phase losses too

    # Optimization
    "lr": 1e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-6,

    # Logging & Checkpoints
    "log_every": 1,
    "eval_every": 10,
    "ckpt_every": 25,
    "sample_eval_limit": 16,
    "val_batches": 32,
}


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_table(rows: list, headers: list = None, col_widths: list = None):
    """Print a formatted table."""
    if not rows:
        return

    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in ([headers] if headers else []) + rows) + 2
                      for i in range(len(rows[0]))]

    def format_row(row):
        return "│ " + " │ ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " │"

    separator = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    print(top_border)
    if headers:
        print(format_row(headers))
        print(separator)
    for row in rows:
        print(format_row(row))
    print(bottom_border)


def create_filtered_index(
    src_index_path: Path,
    dst_index_path: Path,
    devices: list,
    train_per_device: dict,
    val_per_device: dict,
    seed: int = 42,
) -> tuple:
    """Create a filtered index.json with only specified devices and sample counts."""
    import random
    random.seed(seed)

    with open(src_index_path, "r") as f:
        all_entries = json.load(f)

    # Group by device and split
    by_device_split = {}
    for entry in all_entries:
        device = entry["device"]
        split = entry["split"]
        if device not in devices:
            continue
        key = (device, split)
        by_device_split.setdefault(key, []).append(entry)

    filtered = []
    stats = []

    for device in devices:
        # Training
        train_entries = by_device_split.get((device, "train"), [])
        random.shuffle(train_entries)
        n_train = min(train_per_device.get(device, len(train_entries)), len(train_entries))
        filtered.extend(train_entries[:n_train])

        # Validation
        val_entries = by_device_split.get((device, "val"), [])
        random.shuffle(val_entries)
        n_val = min(val_per_device.get(device, len(val_entries)), len(val_entries))
        filtered.extend(val_entries[:n_val])

        total = n_train + n_val
        pct = 100 * total / 5000
        stats.append([device, n_train, n_val, total, f"{pct:.1f}%"])

    # Sort for reproducibility
    filtered.sort(key=lambda e: (e["shard"], e["slot"]))

    os.makedirs(dst_index_path.parent, exist_ok=True)
    with open(dst_index_path, "w") as f:
        json.dump(filtered, f, indent=2)

    n_train_total = len([e for e in filtered if e["split"] == "train"])
    n_val_total = len([e for e in filtered if e["split"] == "val"])

    return n_train_total, n_val_total, stats


def print_target_specs():
    """Print target specifications table."""
    print_header("TARGET SPECIFICATIONS")
    rows = []
    for metric, spec in TARGET_SPECS.items():
        target_str = f"{spec['target']:.2e}" if spec['target'] < 0.01 else f"{spec['target']}"
        rows.append([metric, target_str + spec['unit'], spec['desc']])
    print_table(rows, headers=["Metric", "Target", "Description"])


def print_loss_config():
    """Print loss configuration."""
    print_header("LOSS CONFIGURATION")

    losses = [
        ["Flow Matching", "1.0", "Always on", "✓"],
        ["Residual (Helmholtz)", f"{TRAINING_CONFIG['lambda_residual']}", f"Phase B+ (warmup {TRAINING_CONFIG['residual_warmup_epochs']}ep)", "✓"],
        ["Endpoint", f"{TRAINING_CONFIG['lambda_endpoint']}", f"Phase C (warmup {TRAINING_CONFIG['endpoint_warmup_epochs']}ep)", "✓"],
        ["S-Parameter", f"{TRAINING_CONFIG['lambda_sparam']}", f"Phase C (warmup {TRAINING_CONFIG['sparam_warmup_epochs']}ep)", "✓"],
        ["Phase", f"{TRAINING_CONFIG['lambda_phase']}", "Disabled", "✗"],
        ["Phase Gradient", f"{TRAINING_CONFIG['lambda_phase_grad']}", "Disabled", "✗"],
    ]
    print_table(losses, headers=["Loss", "λ", "Schedule", ""])


def print_curriculum():
    """Print curriculum schedule."""
    print_header("CURRICULUM SCHEDULE")

    phases = [
        ["Phase A", f"0 → {TRAINING_CONFIG['phaseA_epochs']}", "Flow Matching only"],
        ["Phase B", f"{TRAINING_CONFIG['phaseA_epochs']} → {TRAINING_CONFIG['phaseB_epochs']}", "FM + Residual"],
        ["Phase C", f"{TRAINING_CONFIG['phaseB_epochs']} → end", "FM + Residual + Endpoint + S-param"],
    ]
    print_table(phases, headers=["Phase", "Epochs", "Active Losses"])


def print_unrolling():
    """Print unrolling configuration."""
    print_header("UNROLLING (PBFM)")

    steps = TRAINING_CONFIG['unroll_steps']
    if steps > 0:
        fwd_passes = steps * 2  # Heun's method = 2 passes per step
        print(f"  Status:        ENABLED")
        print(f"  Steps:         {steps} (Heun/RK2 integration)")
        print(f"  Extra passes:  ~{fwd_passes} forward passes per batch")
        print(f"  Phase unroll:  {'Yes' if TRAINING_CONFIG['unroll_phase'] else 'No'}")
        print(f"  Purpose:       Integrate ODE t→1 for physics losses")
    else:
        print(f"  Status:        DISABLED (one-step prediction only)")


def main():
    parser = argparse.ArgumentParser(
        description="Train Rayfield on 5-device subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--version", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-complex-unet", action="store_true", help="Use real-valued UNet")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model channels")
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--source-data", type=str, default=None,
                        help="Source data directory (default: Data/unified_sweep)")
    args = parser.parse_args()

    # Generate version name if not provided
    if args.version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.version = f"subset_5dev_{timestamp}"

    # Paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / "Data"

    # Source data can be overridden (e.g., for local scratch)
    if args.source_data:
        unified_sweep = Path(args.source_data)
    else:
        unified_sweep = data_root / "unified_sweep"

    src_index = unified_sweep / "shards" / "index.json"
    filtered_dir = data_root / "filtered_5devices"
    filtered_shards = filtered_dir / "shards"
    dst_index = filtered_shards / "index.json"

    # ===========================================================================
    # PRINT CONFIGURATION
    # ===========================================================================

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    RAYFIELD - 5-Device Subset Training".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Dataset
    print_header("DATASET")
    n_train, n_val, device_stats = create_filtered_index(
        src_index, dst_index, DEVICES,
        TRAIN_SAMPLES_PER_DEVICE, VAL_SAMPLES_PER_DEVICE,
    )
    print_table(device_stats, headers=["Device", "Train", "Val", "Total", "%"])
    print(f"\n  Total: {n_train + n_val} samples ({n_train} train / {n_val} val)")
    print(f"  Split: {100*n_train/(n_train+n_val):.0f}% / {100*n_val/(n_train+n_val):.0f}%")

    # Symlink shards
    src_shards = unified_sweep / "shards"
    for shard_file in src_shards.glob("shard_*.npz"):
        dst_shard = filtered_shards / shard_file.name
        if not dst_shard.exists():
            os.symlink(shard_file, dst_shard)

    # Model
    print_header("MODEL")
    use_complex = not args.no_complex_unet
    print(f"  Architecture:  {'Complex' if use_complex else 'Real'}-valued Physics UNet")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Attention:     Enabled at ds=8")

    # Losses
    print_loss_config()

    # Curriculum
    print_curriculum()

    # Unrolling
    print_unrolling()

    # Targets
    print_target_specs()

    # Training
    print_header("TRAINING")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  GPUs:          {args.gpus}")
    print(f"  Learning rate: {TRAINING_CONFIG['lr']}")
    print(f"  Warmup:        {TRAINING_CONFIG['warmup_epochs']} epochs")
    print(f"  Min LR:        {TRAINING_CONFIG['min_lr']}")
    print(f"  Experiment:    {args.version}")
    print(f"  W&B:           {'Enabled' if args.use_wandb else 'Disabled'}")
    if args.resume:
        print(f"  Resume from:   {args.resume}")

    if args.dry_run:
        print_header("DRY RUN - No training will be executed")
        return

    # ===========================================================================
    # BUILD AND RUN TRAINING COMMAND
    # ===========================================================================

    print_header("LAUNCHING TRAINING")

    train_script = project_root / "Model" / "train.py"

    use_torchrun = args.gpus > 1
    if use_torchrun:
        cmd = ["torchrun", "--nproc_per_node", str(args.gpus)]

        # Multi-node support when launched via Slurm
        slurm_nnodes = os.environ.get("SLURM_NNODES")
        slurm_nodeid = os.environ.get("SLURM_NODEID")
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")

        if slurm_nnodes and slurm_nodeid:
            cmd.extend(["--nnodes", str(slurm_nnodes), "--node_rank", str(slurm_nodeid)])
            if master_addr and master_port:
                cmd.extend(["--master_addr", master_addr, "--master_port", master_port])
    else:
        cmd = ["python"]

    cmd.extend([
        str(train_script),
        # Data
        "--data-root", str(data_root),
        "--use-shards",
        "--use-index-split",
        "--include-sweeps", filtered_dir.name,
        # Model
        "--hidden-size", str(args.hidden_size),
        # Training
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--version", args.version,
        "--lr", str(TRAINING_CONFIG['lr']),
        "--warmup-epochs", str(TRAINING_CONFIG['warmup_epochs']),
        "--min-lr", str(TRAINING_CONFIG['min_lr']),
        # Losses
        "--lambda-residual", str(TRAINING_CONFIG['lambda_residual']),
        "--lambda-endpoint", str(TRAINING_CONFIG['lambda_endpoint']),
        "--lambda-sparam", str(TRAINING_CONFIG['lambda_sparam']),
        "--lambda-phase", str(TRAINING_CONFIG['lambda_phase']),
        "--lambda-phase-grad", str(TRAINING_CONFIG['lambda_phase_grad']),
        # Curriculum
        "--phaseA-epochs", str(TRAINING_CONFIG['phaseA_epochs']),
        "--phaseB-epochs", str(TRAINING_CONFIG['phaseB_epochs']),
        # Warmup
        "--residual-warmup-epochs", str(TRAINING_CONFIG['residual_warmup_epochs']),
        "--endpoint-warmup-epochs", str(TRAINING_CONFIG['endpoint_warmup_epochs']),
        "--sparam-warmup-epochs", str(TRAINING_CONFIG['sparam_warmup_epochs']),
        # S-param
        "--sparam-mode", TRAINING_CONFIG['sparam_mode'],
        "--sparam-every", str(TRAINING_CONFIG['sparam_every']),
        # Unrolling
        "--unroll-steps", str(TRAINING_CONFIG['unroll_steps']),
        # Logging
        "--log-every", str(TRAINING_CONFIG['log_every']),
        "--eval-every", str(TRAINING_CONFIG['eval_every']),
        "--ckpt-every", str(TRAINING_CONFIG['ckpt_every']),
        "--sample-eval-limit", str(TRAINING_CONFIG['sample_eval_limit']),
        "--val-batches", str(TRAINING_CONFIG['val_batches']),
        # UI
        "--tqdm",
    ])

    if use_complex:
        cmd.append("--complex-unet")

    if TRAINING_CONFIG['unroll_phase']:
        cmd.append("--unroll-phase")

    if args.use_wandb:
        cmd.append("--use-wandb")

    if args.resume:
        cmd.extend(["--resume-from", args.resume])

    print(f"\n  Command: {' '.join(cmd[:5])} ... [{len(cmd)} args]\n")

    # Run training
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
