import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict


class CouplerDataset(Dataset):
    """
    Dataset for FDTD simulations of 2D directional couplers.

    Each sample folder must contain:
      - eps.npy       : [H, W]
      - Ez_real.npy   : [H, W]
      - Ez_imag.npy   : [H, W]
      - grid_meta.npz : metadata (dx, dy, nx, ny, Lx_um, Ly_um)
      - sparams.npz   : includes 'wavelength_um' and device params, S_real, S_imag
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_fraction: float = 0.8,
        stats: Optional[Dict[str, float]] = None,
        normalize_eps: bool = True,
        use_double: bool = False,
    ):
        """
        root_dir: path to coupler_sweep (not its parent).
        split: "train" or "val"
        train_fraction: fraction of samples for train split
        stats: normalization stats; if None and split=="train", compute them.
        normalize_eps: whether to normalize eps channel
        """
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.train_fraction = train_fraction
        self.normalize_eps = normalize_eps
        self.use_double = use_double

        if not self.root.is_dir():
            raise ValueError(f"root_dir {root_dir} does not exist or is not a directory")

        # all device folders under coupler_sweep
        all_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if not all_dirs:
            raise RuntimeError(f"No sample subfolders found under {self.root}")

        # deterministic split
        n_total = len(all_dirs)
        n_train = int(round(train_fraction * n_total))
        if split == "train":
            self.sample_dirs = all_dirs[:n_train]
        elif split == "val":
            self.sample_dirs = all_dirs[n_train:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No samples in {split} split (train_fraction={train_fraction})")

        # compute or reuse stats (fields, eps, wavelength)
        if stats is None:
            if split != "train":
                raise ValueError("stats must be provided for non-train splits")
            print(f"[{split}] computing normalization stats from subset...")
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        print(f"[{split}] dataset size = {len(self.sample_dirs)}")

    def _compute_stats(self) -> Dict[str, float]:
        # use a subset to estimate stats
        subset_size = min(len(self.sample_dirs), 500)
        idxs = np.random.choice(len(self.sample_dirs), subset_size, replace=False)

        ez_r_all = []
        ez_i_all = []
        eps_all = []
        lam_all = []

        for i in idxs:
            d = self.sample_dirs[i]
            ez_r = np.load(d / "Ez_real.npy")
            ez_i = np.load(d / "Ez_imag.npy")
            eps = np.load(d / "eps.npy")

            sp = np.load(d / "sparams.npz")
            lam = float(sp["wavelength_um"])  # scalar

            ez_r_all.append(ez_r)
            ez_i_all.append(ez_i)
            eps_all.append(eps)
            lam_all.append(lam)

        ez_r_cat = np.stack(ez_r_all)
        ez_i_cat = np.stack(ez_i_all)
        eps_cat = np.stack(eps_all)
        lam_arr = np.array(lam_all, dtype=np.float32)

        stats = dict(
            ez_real_mean=float(ez_r_cat.mean()),
            ez_real_std=float(ez_r_cat.std()) + 1e-8,
            ez_imag_mean=float(ez_i_cat.mean()),
            ez_imag_std=float(ez_i_cat.std()) + 1e-8,
            eps_mean=float(eps_cat.mean()),
            eps_std=float(eps_cat.std()) + 1e-8,
            lambda_um_mean=float(lam_arr.mean()),
            lambda_um_std=float(lam_arr.std()) + 1e-8,
        )

        print(
            "Stats:",
            {k: round(v, 4) for k, v in stats.items()},
        )
        return stats

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        d = self.sample_dirs[idx]
        dtype_np = np.float64 if self.use_double else np.float32

        ez_r = np.load(d / "Ez_real.npy").astype(dtype_np)
        ez_i = np.load(d / "Ez_imag.npy").astype(dtype_np)
        eps = np.load(d / "eps.npy").astype(dtype_np)

        sp = np.load(d / "sparams.npz")
        lam_um = float(sp["wavelength_um"])  # scalar

        # normalize fields
        ez_r = (ez_r - self.stats["ez_real_mean"]) / self.stats["ez_real_std"]
        ez_i = (ez_i - self.stats["ez_imag_mean"]) / self.stats["ez_imag_std"]
        if self.normalize_eps:
            eps = (eps - self.stats["eps_mean"]) / self.stats["eps_std"]

        # normalize wavelength for conditioning
        lam_norm = (lam_um - self.stats["lambda_um_mean"]) / self.stats["lambda_um_std"]

        sample = np.stack([ez_r, ez_i, eps], axis=0)  # [3,H,W]
        x = torch.from_numpy(sample)
        cond = torch.tensor([lam_norm], dtype=x.dtype)  # [1]

        return x, cond

    def get_stats(self) -> Dict[str, float]:
        return self.stats
