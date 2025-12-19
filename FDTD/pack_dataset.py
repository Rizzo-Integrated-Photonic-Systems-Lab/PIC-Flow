"""
pack_dataset.py

Utility to pack many per-sample folders (e.g., coupler_sweep or y_branch_sweep)
into fewer shard .npz files to reduce file-open overhead during training.

Usage examples:
  python FDTD/pack_dataset.py --data-root ../Data/coupler_sweep --dataset coupler --shard-size 100
  python FDTD/pack_dataset.py --data-root ../Data/y_branch_sweep --dataset ybranch --shard-size 100

Outputs:
  - Shards written to <data-root>/shards unless --out-root is provided.
  - An index JSON mapping each sample tag to (shard_path, slot_in_shard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def list_samples(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def load_sample(
    sample_dir: Path, dataset: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, float | int]]:
    """
    Return (arrays, meta) where arrays is a dict of numpy arrays to store in shard.
    meta carries small scalars to copy separately.
    """
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, float | int] = {}

    # sparams
    sparams_path = sample_dir / "sparams.npz"
    if not sparams_path.exists():
        raise FileNotFoundError(f"Missing sparams.npz in {sample_dir}")
    sparams = load_npz_dict(sparams_path)

    for k, v in sparams.items():
        if isinstance(v, np.ndarray):
            arrays[f"sparams/{k}"] = v
        else:
            # store scalars in meta
            meta[f"sparams/{k}"] = float(v) if np.issubdtype(type(v), np.floating) else int(v)

    # eps
    eps_path = sample_dir / "eps.npy"
    if eps_path.exists():
        arrays["eps"] = np.load(eps_path)

    # fields
    ezr_path = sample_dir / "Ez_real.npy"
    ezi_path = sample_dir / "Ez_imag.npy"
    if ezr_path.exists():
        arrays["Ez_real"] = np.load(ezr_path)
    if ezi_path.exists():
        arrays["Ez_imag"] = np.load(ezi_path)

    # src mask (coupler only)
    src_mask_path = sample_dir / "src_mask.npy"
    if src_mask_path.exists():
        arrays["src_mask"] = np.load(src_mask_path)

    # grid meta
    grid_meta_path = sample_dir / "grid_meta.npz"
    if grid_meta_path.exists():
        gm = load_npz_dict(grid_meta_path)
        for k, v in gm.items():
            if np.isscalar(v) or v.shape == ():
                meta[f"grid/{k}"] = float(v) if np.issubdtype(v.dtype, np.floating) else int(v)
            else:
                arrays[f"grid/{k}"] = v

    # dataset tag
    meta["tag"] = sample_dir.name
    meta["dataset"] = dataset

    return arrays, meta


def write_shard(out_path: Path, samples: List[Tuple[Dict[str, np.ndarray], Dict[str, float | int]]]):
    """
    Write a shard containing multiple samples. Each sample i gets keys prefixed with f"s{i}/".
    """
    save_dict: Dict[str, np.ndarray] = {}
    for i, (arrays, meta) in enumerate(samples):
        prefix = f"s{i}/"
        for k, v in arrays.items():
            save_dict[prefix + k] = v
        # meta as small arrays to keep everything in npz
        for k, v in meta.items():
            save_dict[prefix + k] = np.array(v)
    np.savez(out_path, **save_dict)


def pack_dataset(
    data_root: Path,
    out_root: Path,
    dataset: str,
    shard_size: int,
    max_samples: int | None = None,
) -> List[Tuple[str, str, int]]:
    """
    Returns index: list of (tag, shard_rel_path, slot_in_shard)
    """
    samples = list_samples(data_root)
    if max_samples is not None:
        samples = samples[:max_samples]

    out_root.mkdir(parents=True, exist_ok=True)
    index: List[Tuple[str, str, int]] = []

    batch: List[Tuple[Dict[str, np.ndarray], Dict[str, float | int]]] = []
    shard_id = 0
    for sample_dir in tqdm(samples, desc="Packing"):
        arrays, meta = load_sample(sample_dir, dataset=dataset)
        batch.append((arrays, meta))
        if len(batch) == shard_size:
            shard_name = f"shard_{shard_id:05d}.npz"
            shard_path = out_root / shard_name
            write_shard(shard_path, batch)
            for slot, (_, m) in enumerate(batch):
                index.append((m["tag"], shard_name, slot))
            shard_id += 1
            batch = []

    # flush remainder
    if batch:
        shard_name = f"shard_{shard_id:05d}.npz"
        shard_path = out_root / shard_name
        write_shard(shard_path, batch)
        for slot, (_, m) in enumerate(batch):
            index.append((m["tag"], shard_name, slot))

    return index


def main():
    ap = argparse.ArgumentParser(description="Pack per-sample folders into shard .npz files.")
    ap.add_argument("--data-root", type=Path, required=True, help="Root folder with sample subfolders.")
    ap.add_argument("--out-root", type=Path, default=None, help="Where to write shards (default: <data-root>/shards)")
    ap.add_argument("--dataset", choices=["coupler", "ybranch"], required=True, help="Dataset type.")
    ap.add_argument("--shard-size", type=int, default=100, help="Samples per shard.")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for testing.")
    args = ap.parse_args()

    data_root = args.data_root
    out_root = args.out_root if args.out_root is not None else data_root / "shards"

    index = pack_dataset(
        data_root=data_root,
        out_root=out_root,
        dataset=args.dataset,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
    )

    index_path = out_root / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            [{"tag": t, "shard": s, "slot": int(slot)} for (t, s, slot) in index],
            f,
            indent=2,
        )
    print(f"Wrote {len(index)} samples into {len(set([s for _, s, _ in index]))} shards.")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()

