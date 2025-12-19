import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Iterable, List

def phase_anchor_roi(ez_r, ez_i, eps_r=None, pml_cells=30, margin=2, roi_x=(40, 140), thr_eps=3.0, eps=1e-8):
    """
    Anchor global phase using a stable ROI near the input waveguide.
    Optionally uses eps_r to restrict to core (eps_r > thr_eps).
    """
    H, W = ez_r.shape
    p2 = min(pml_cells + margin, max(0, H//2 - 1), max(0, W//2 - 1))

    x0, x1 = roi_x
    x0 = max(x0, p2); x1 = min(x1, W - p2)
    y0, y1 = p2, H - p2

    Ez = ez_r + 1j * ez_i
    roi = Ez[y0:y1, x0:x1]

    if eps_r is not None:
        roi_eps = eps_r[y0:y1, x0:x1]
        m = (roi_eps > thr_eps).astype(np.float32)
    else:
        m = np.ones_like(roi.real, dtype=np.float32)

    A = np.abs(roi)
    # weight by A^2 but average unit phasors => stable phase estimate
    w = (A**2) * m
    u = roi / np.maximum(A, eps)

    c = np.sum(w * u)
    phi = np.angle(c + 1j*0.0)

    rot = np.exp(-1j * phi)
    Ez2 = Ez * rot
    return Ez2.real.astype(ez_r.dtype), Ez2.imag.astype(ez_i.dtype), float(phi)


class FDTDDataset(Dataset):
    """
    Unified dataset that pulls samples from all sweep subdirectories under a root
    (e.g., Data/ containing coupler_sweep/, y_branch_sweep/, ...).

    Each sample folder must contain:
      - eps.npy
      - Ez_real.npy
      - Ez_imag.npy
      - grid_meta.npz
      - sparams.npz (with 'wavelength_um')
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_fraction: float = 0.8,
        stats: Optional[Dict[str, float]] = None,
        normalize_eps: bool = True,
        use_double: bool = False,
        include_sweeps: Optional[Iterable[str]] = None,
    ):
        """
        root_dir: path to Data (parent containing sweep subfolders).
        include_sweeps: optional iterable of subfolder names to include (e.g., ["coupler_sweep", "y_branch_sweep"]);
                        if None, include all immediate subdirectories.
        """
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.train_fraction = train_fraction
        self.normalize_eps = normalize_eps
        self.use_double = use_double

        if not self.root.is_dir():
            raise ValueError(f"root_dir {root_dir} does not exist or is not a directory")

        sweep_dirs: List[Path] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            if include_sweeps is not None and d.name not in include_sweeps:
                continue
            sweep_dirs.append(d)

        if not sweep_dirs:
            raise RuntimeError(f"No sweep subfolders found under {self.root}")

        all_dirs: List[Path] = []
        for sdir in sweep_dirs:
            all_dirs += [d for d in sorted(sdir.iterdir()) if d.is_dir()]

        if not all_dirs:
            raise RuntimeError(f"No sample subfolders found in sweeps {sweep_dirs}")

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

        if stats is None:
            if split != "train":
                raise ValueError("stats must be provided for non-train splits")
            print(f"[{split}] computing normalization stats from subset...")
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        print(f"[{split}] dataset size = {len(self.sample_dirs)}")

    def _compute_stats(self) -> Dict[str, float]:
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
            ez_r, ez_i, _ = phase_anchor_roi(ez_r, ez_i, eps_r=eps, pml_cells=30, margin=2)

            sp = np.load(d / "sparams.npz")
            lam = float(sp["wavelength_um"])

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

        print("Stats:", {k: round(v, 4) for k, v in stats.items()})
        return stats

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        d = self.sample_dirs[idx]
        dtype_np = np.float64 if self.use_double else np.float32

        ez_r = np.load(d / "Ez_real.npy").astype(dtype_np)
        ez_i = np.load(d / "Ez_imag.npy").astype(dtype_np)
        eps = np.load(d / "eps.npy").astype(dtype_np)


        ### Phase anchor ###
        ez_r, ez_i, phi = phase_anchor_roi(
            ez_r, ez_i,
            eps_r=eps,
            pml_cells=30,
            margin=2,
            roi_x=(40, 140),
            thr_eps=3.0,
        )

        sp = np.load(d / "sparams.npz")
        lam_um = float(sp["wavelength_um"])

        ### Normalize ###
        ez_r = (ez_r - self.stats["ez_real_mean"]) / self.stats["ez_real_std"]
        ez_i = (ez_i - self.stats["ez_imag_mean"]) / self.stats["ez_imag_std"]
        if self.normalize_eps:
            eps = (eps - self.stats["eps_mean"]) / self.stats["eps_std"]

        lam_norm = (lam_um - self.stats["lambda_um_mean"]) / self.stats["lambda_um_std"]

        sample = np.stack([ez_r, ez_i, eps], axis=0)
        x = torch.from_numpy(sample)
        cond = torch.tensor([lam_norm], dtype=x.dtype)

        return x, cond

    def get_stats(self) -> Dict[str, float]:
        return self.stats
