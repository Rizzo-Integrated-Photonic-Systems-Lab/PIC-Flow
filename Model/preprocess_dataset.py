#!/usr/bin/env python3
"""
Preprocess FDTD dataset for fast training.

This script:
1. Loads samples from NPZ shards
2. Applies phase anchoring (expensive at runtime)
3. Saves as uncompressed .pt files for fast loading

Usage:
    python preprocess_dataset.py \
        --data-root /path/to/Data \
        --output-dir /path/to/Data_preprocessed \
        --include-sweeps unified_sweep \
        --num-workers 16
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Import phase anchoring functions from dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import phase_anchor_mask, phase_anchor_roi, FDTDDataset


def process_single_sample(
    args_tuple: Tuple[int, Path, int, str, Path, int, Dict]
) -> Optional[str]:
    """
    Process a single sample: load from shard, phase anchor, save as .pt

    Returns error message if failed, None if successful.
    """
    idx, shard_path, slot, tag, output_dir, pml_cells, stats = args_tuple

    try:
        # Load from shard
        data = np.load(shard_path, allow_pickle=False)
        prefix = f"s{slot}/"

        def get_arr(key: str, required: bool = True):
            full_key = prefix + key
            if full_key not in data:
                if required:
                    raise KeyError(f"Missing key {full_key}")
                return None
            return data[full_key].astype(np.float32)

        # Load main arrays
        ez_r = get_arr("Ez_real")
        ez_i = get_arr("Ez_imag")
        eps = get_arr("eps")
        src = get_arr("src_mask", required=False)
        if src is None:
            src = np.zeros_like(eps)

        # Wavelength
        lam_arr = get_arr("sparams/wavelength_um", required=False)
        if lam_arr is None:
            lam_arr = get_arr("sparams/lambda_um", required=False)
        if lam_arr is None:
            lam_arr = get_arr("wavelength_um", required=False)
        if lam_arr is None:
            return f"Missing wavelength for {shard_path} slot {slot}"
        lam_um = float(lam_arr)

        # Phase anchoring (the expensive part!)
        ez_r2, ez_i2, _ = phase_anchor_mask(ez_r, ez_i, (src > 0.5), eps_r=eps, thr_eps=3.0)
        if (src is None) or (float((src > 0.5).sum()) < 4.0):
            ez_r2, ez_i2, _ = phase_anchor_roi(
                ez_r, ez_i, eps_r=eps, pml_cells=pml_cells, margin=2
            )
        ez_r, ez_i = ez_r2, ez_i2

        # Normalize fields
        ez_r_norm = (ez_r - stats["ez_real_mean"]) / stats["ez_real_std"]
        ez_i_norm = (ez_i - stats["ez_imag_mean"]) / stats["ez_imag_std"]
        eps_norm = (eps - stats["eps_mean"]) / stats["eps_std"]
        src_binary = (src > 0.5).astype(np.float32)

        # Load auxiliary data
        aux = {}

        # S-parameters
        Sr = get_arr("sparams/S_real", required=False)
        Si = get_arr("sparams/S_imag", required=False)
        if Sr is not None and Si is not None:
            aux["sparams_true"] = (Sr + 1j * Si).astype(np.complex64)
        else:
            # Try individual S-params
            sparam_keys = []
            for key in data.files:
                if key.startswith(prefix + "sparams/S") and key.endswith("_real"):
                    sparam_name = key[len(prefix + "sparams/"):-len("_real")]
                    if len(sparam_name) >= 2 and sparam_name[0] == "S":
                        sparam_keys.append(sparam_name)

            if sparam_keys:
                sparam_keys = sorted(set(sparam_keys), key=lambda s: int(s[1]) if len(s) > 1 and s[1].isdigit() else 999)
                sparams = []
                for sname in sparam_keys:
                    real_key = prefix + f"sparams/{sname}_real"
                    imag_key = prefix + f"sparams/{sname}_imag"
                    if real_key in data and imag_key in data:
                        sr = float(data[real_key])
                        si = float(data[imag_key])
                        sparams.append(complex(sr, si))
                if sparams:
                    aux["sparams_true"] = np.array(sparams, dtype=np.complex64)

        # Port masks
        port_masks = get_arr("ports/masks", required=False)
        if port_masks is None:
            port_masks = get_arr("port_masks", required=False)
        if port_masks is not None:
            aux["port_masks"] = port_masks

        # Port IDs
        port_ids = None
        port_ids_key = prefix + "ports/ids"
        if port_ids_key not in data:
            port_ids_key = prefix + "port_ids"
        if port_ids_key in data:
            port_ids = data[port_ids_key].astype(np.int32)
            aux["port_ids"] = port_ids

        # Input port
        in_port = get_arr("sparams/input_port", required=False)
        if in_port is None:
            in_port = get_arr("input_port", required=False)
        if in_port is not None:
            in_port_val = int(np.array(in_port).item())
            if port_ids is not None:
                hits = np.where(port_ids == in_port_val)[0]
                if len(hits) > 0:
                    aux["in_port_idx"] = int(hits[0])
            else:
                aux["in_port_idx"] = in_port_val - 1

        # Get split info if available
        split_key = prefix + "split"
        if split_key in data:
            try:
                split_val = data[split_key]
                if isinstance(split_val, np.ndarray):
                    split_val = split_val.item()
                if isinstance(split_val, bytes):
                    split_val = split_val.decode()
                aux["split"] = str(split_val)
            except:
                pass

        # Build output tensor dict
        sample = {
            "ez_real": torch.from_numpy(ez_r_norm),
            "ez_imag": torch.from_numpy(ez_i_norm),
            "eps": torch.from_numpy(eps_norm),
            "src": torch.from_numpy(src_binary),
            "eps_phys": torch.from_numpy(eps),  # Keep physical eps for aux
            "lambda_um": torch.tensor(lam_um, dtype=torch.float32),
            "tag": tag,
        }

        # Add aux data
        for k, v in aux.items():
            if isinstance(v, np.ndarray):
                if np.iscomplexobj(v):
                    sample[f"aux_{k}"] = torch.from_numpy(v)
                else:
                    sample[f"aux_{k}"] = torch.from_numpy(v.astype(np.float32))
            else:
                sample[f"aux_{k}"] = v

        # Save as .pt file
        safe_tag = tag.replace("/", "_").replace("\\", "_")
        output_path = output_dir / f"{safe_tag}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, output_path)

        return None  # Success

    except Exception as e:
        return f"Error processing {shard_path} slot {slot}: {e}"


def compute_stats_from_shards(
    shard_refs: List[Tuple[Path, int, str]],
    pml_cells: int = 0,
    max_samples: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute normalization stats from a subset of samples."""

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(shard_refs), min(max_samples, len(shard_refs)), replace=False)

    ez_r_all, ez_i_all, eps_all, lam_all = [], [], [], []

    print(f"Computing stats from {len(indices)} samples...")
    for i in tqdm(indices, desc="Stats"):
        shard_path, slot, tag = shard_refs[i]

        try:
            data = np.load(shard_path, allow_pickle=False)
            prefix = f"s{slot}/"

            ez_r = data[prefix + "Ez_real"].astype(np.float32)
            ez_i = data[prefix + "Ez_imag"].astype(np.float32)
            eps = data[prefix + "eps"].astype(np.float32)

            src_key = prefix + "src_mask"
            src = data[src_key].astype(np.float32) if src_key in data else np.zeros_like(eps)

            # Phase anchor
            ez_r2, ez_i2, _ = phase_anchor_mask(ez_r, ez_i, (src > 0.5), eps_r=eps, thr_eps=3.0)
            if float((src > 0.5).sum()) < 4.0:
                ez_r2, ez_i2, _ = phase_anchor_roi(ez_r, ez_i, eps_r=eps, pml_cells=pml_cells, margin=2)

            ez_r_all.append(ez_r2)
            ez_i_all.append(ez_i2)
            eps_all.append(eps)

            # Wavelength
            for key in ["sparams/wavelength_um", "sparams/lambda_um", "wavelength_um"]:
                if prefix + key in data:
                    lam_all.append(float(data[prefix + key]))
                    break

        except Exception as e:
            print(f"Warning: skipping {shard_path} slot {slot}: {e}")
            continue

    ez_r_cat = np.stack(ez_r_all)
    ez_i_cat = np.stack(ez_i_all)
    eps_cat = np.stack(eps_all)
    lam_arr = np.array(lam_all)

    stats = {
        "ez_real_mean": float(ez_r_cat.mean()),
        "ez_real_std": float(ez_r_cat.std()) + 1e-8,
        "ez_imag_mean": float(ez_i_cat.mean()),
        "ez_imag_std": float(ez_i_cat.std()) + 1e-8,
        "eps_mean": float(eps_cat.mean()),
        "eps_std": float(eps_cat.std()) + 1e-8,
        "lambda_um_mean": float(lam_arr.mean()),
        "lambda_um_std": float(lam_arr.std()) + 1e-8,
    }

    print("Stats:", {k: round(v, 4) for k, v in stats.items()})
    return stats


def collect_shard_refs(data_root: Path, sweeps: List[str], shard_subdir: str = "shards") -> List[Tuple[Path, int, str]]:
    """Collect all shard references."""
    refs = []

    for sweep in sweeps:
        sweep_dir = data_root / sweep
        shard_dir = sweep_dir / shard_subdir
        index_path = shard_dir / "index.json"

        if not index_path.is_file():
            print(f"Warning: No index.json found at {index_path}")
            continue

        with open(index_path) as f:
            entries = json.load(f)

        for e in entries:
            shard_path = shard_dir / e["shard"]
            slot = int(e["slot"])
            tag = e.get("tag", e.get("geometry_id", f"{sweep}_{slot}"))
            refs.append((shard_path, slot, tag))

    print(f"Found {len(refs)} samples across {len(sweeps)} sweeps")
    return refs


def main():
    parser = argparse.ArgumentParser(description="Preprocess FDTD dataset for fast training")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--include-sweeps", type=str, required=True, help="Comma-separated sweep names")
    parser.add_argument("--shard-subdir", type=str, default="shards")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pml-cells", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=500, help="Number of samples for computing stats")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction for train split")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Fraction for val split (rest is test)")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split assignment")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    sweeps = [s.strip() for s in args.include_sweeps.split(",")]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect shard references
    shard_refs = collect_shard_refs(data_root, sweeps, args.shard_subdir)

    if not shard_refs:
        print("No samples found!")
        return

    # Compute stats
    stats = compute_stats_from_shards(shard_refs, args.pml_cells, args.stats_samples)

    # Save stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Create index for preprocessed samples with split assignments
    index = []

    # Assign splits deterministically based on tag hash
    rng = np.random.default_rng(args.split_seed)
    n_samples = len(shard_refs)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * args.train_fraction)
    n_val = int(n_samples * args.val_fraction)

    split_assignments = {}  # tag -> split
    for i, idx in enumerate(indices):
        _, _, tag = shard_refs[idx]
        if i < n_train:
            split_assignments[tag] = "train"
        elif i < n_train + n_val:
            split_assignments[tag] = "val"
        else:
            split_assignments[tag] = "test"

    print(f"Split assignments: train={n_train}, val={n_val}, test={n_samples - n_train - n_val}")

    # Process all samples in parallel
    print(f"\nPreprocessing {len(shard_refs)} samples with {args.num_workers} workers...")

    tasks = []
    for idx, (shard_path, slot, tag) in enumerate(shard_refs):
        tasks.append((idx, shard_path, slot, tag, output_dir, args.pml_cells, stats))

    errors = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_sample, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            task = futures[future]
            idx, shard_path, slot, tag, _, _, _ = task

            try:
                error = future.result()
                if error:
                    errors.append(error)
                else:
                    safe_tag = tag.replace("/", "_").replace("\\", "_")
                    index.append({
                        "file": f"{safe_tag}.pt",
                        "tag": tag,
                        "original_shard": str(shard_path.name),
                        "original_slot": slot,
                        "split": split_assignments.get(tag, "train"),
                    })
            except Exception as e:
                errors.append(f"Exception for {shard_path} slot {slot}: {e}")

    # Save index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved index with {len(index)} samples to {index_path}")

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nDone! Preprocessed data saved to {output_dir}")
    print(f"Total samples: {len(index)}")


if __name__ == "__main__":
    main()
