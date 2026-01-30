# dataset_fast.py
"""
Fast dataset loader for preprocessed .pt files.

This is 10-20x faster than loading from NPZ shards because:
1. No decompression needed (.pt is uncompressed by default)
2. No phase anchoring (already done during preprocessing)
3. No normalization (already done during preprocessing)
4. Direct tensor loading (no numpy conversion)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

# D4 transforms (same as dataset.py)
D4_TRANSFORMS = [
    (0, False),  # identity
    (1, False),  # 90 CCW
    (2, False),  # 180
    (3, False),  # 270 CCW
    (0, True),   # horizontal flip
    (1, True),   # 90 CCW + flip
    (2, True),   # 180 + flip
    (3, True),   # 270 CCW + flip
]


def apply_d4_transform_torch(x: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """Apply D4 transform to tensor [..., H, W]."""
    rot_k, flip_h = D4_TRANSFORMS[transform_idx % 8]
    if rot_k != 0:
        x = torch.rot90(x, k=rot_k, dims=(-2, -1))
    if flip_h:
        x = torch.flip(x, dims=[-1])
    return x


class FastFDTDDataset(Dataset):
    """
    Fast dataset that loads preprocessed .pt files.

    Expected directory structure:
        preprocessed_dir/
            stats.json
            index.json
            sample1.pt
            sample2.pt
            ...
    """

    def __init__(
        self,
        preprocessed_dir: str,
        split: str = "train",
        train_fraction: float = 0.9,
        stats: Optional[Dict] = None,
        augment: bool = False,
        return_aux: bool = True,
        max_ports: int = 4,
        split_seed: int = 0,
        use_index_split: bool = False,  # Use pre-computed split from preprocessing
    ):
        super().__init__()
        self.root = Path(preprocessed_dir)
        self.split = split
        self.augment = augment
        self.return_aux = return_aux
        self.max_ports = max_ports

        # Load index
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Index not found: {index_path}")

        with open(index_path) as f:
            all_entries = json.load(f)

        # Load stats
        if stats is None:
            stats_path = self.root / "stats.json"
            if stats_path.is_file():
                with open(stats_path) as f:
                    self.stats = json.load(f)
            else:
                raise FileNotFoundError(f"Stats not found: {stats_path}")
        else:
            self.stats = stats

        # Filter by split if using index split
        if use_index_split:
            # Entries may have 'split' field from preprocessing
            self.entries = [e for e in all_entries if e.get("split", "train") == split]
            if not self.entries:
                # Fall back to all entries if no split field
                self.entries = all_entries
        else:
            # Deterministic split
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(len(all_entries))
            all_entries = [all_entries[i] for i in perm]

            n_train = int(len(all_entries) * train_fraction)
            if split == "train":
                self.entries = all_entries[:n_train]
            else:
                self.entries = all_entries[n_train:]

        print(f"[FastFDTDDataset {split}] Loaded {len(self.entries)} samples")

        # Store stats for compatibility
        self.cond_dim = 2
        self.x_channels = 4

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        pt_path = self.root / entry["file"]

        # Load preprocessed sample (very fast!)
        sample = torch.load(pt_path, weights_only=False)

        # Stack into [C, H, W] format
        x = torch.stack([
            sample["ez_real"],
            sample["ez_imag"],
            sample["eps"],
            sample["src"],
        ], dim=0)  # [4, H, W]

        # Build conditioning vector
        lambda_um = float(sample["lambda_um"])
        lam_std = float(self.stats.get("lambda_um_std", 1.0))
        if lam_std <= 0:
            lam_std = 1.0
        lam_norm = (lambda_um - float(self.stats["lambda_um_mean"])) / lam_std

        # Default wg_width to 0 (normalized) if not present
        cond = torch.tensor([lam_norm, 0.0], dtype=x.dtype)

        # D4 augmentation
        d4_idx = 0
        if self.augment:
            d4_idx = np.random.randint(0, 8)
            x = apply_d4_transform_torch(x, d4_idx)

        if not self.return_aux:
            return x, cond

        # Build aux dict
        H, W = x.shape[-2], x.shape[-1]
        aux = {
            "n_ports": torch.tensor(0, dtype=torch.long),
            "in_port_idx": torch.tensor(-1, dtype=torch.long),
            "port_valid": torch.zeros((self.max_ports,), dtype=x.dtype),
            "port_ids": torch.full((self.max_ports,), -1, dtype=torch.long),
            "port_masks": torch.zeros((self.max_ports, H, W), dtype=x.dtype),
            "sparams_true": torch.zeros((self.max_ports,), dtype=torch.complex64),
        }

        # Load aux data from sample
        if "aux_sparams_true" in sample:
            st = sample["aux_sparams_true"]
            P = min(len(st), self.max_ports)
            aux["sparams_true"][:P] = st[:P]
            aux["n_ports"] = torch.tensor(P, dtype=torch.long)
            aux["port_valid"][:P] = 1.0

        if "aux_port_masks" in sample:
            pm = sample["aux_port_masks"]
            if self.augment and d4_idx != 0:
                pm = apply_d4_transform_torch(pm, d4_idx)
            P = min(len(pm), self.max_ports)
            aux["port_masks"][:P] = pm[:P]
            if aux["n_ports"] == 0:
                aux["n_ports"] = torch.tensor(P, dtype=torch.long)
                aux["port_valid"][:P] = 1.0

        if "aux_port_ids" in sample:
            pid = sample["aux_port_ids"]
            P = min(len(pid), self.max_ports)
            aux["port_ids"][:P] = torch.as_tensor(pid[:P], dtype=torch.long)

        if "aux_in_port_idx" in sample:
            aux["in_port_idx"] = torch.tensor(int(sample["aux_in_port_idx"]), dtype=torch.long)

        return x, cond, aux

    def get_stats(self) -> Dict:
        return self.stats
