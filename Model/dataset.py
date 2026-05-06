# dataset.py
"""
FDTD dataset loader with:
- optional shard (.npz) support
- deterministic splitting (seeded) to avoid sweep-order bias
- robust phase anchoring (mask-first, ROI fallback)
- optional SDF features (raw/clip/exp/clip_exp)
- optional PML cropping with safety checks
- return_aux that is DataLoader-collate-safe (no None values)
- port_masks are cropped consistently when crop_pml=True
- subset selection is reproducible (stable hash; no Python hash randomization)
- D4 symmetry augmentation (8 unique transforms, no duplicates)

Drop-in replacement for your current file.
"""

import json
import os
import zlib
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # optional (fast path)
    from scipy import ndimage as _scipy_ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _scipy_ndimage = None

_SDF_WARNED = False

ShardRef = Tuple[Path, int, str]  # (shard_path, slot, tag)
SampleRef = Union[Path, ShardRef]


# ---------------------------------------------------------------------------
# D4 symmetry group augmentation (8 unique transformations, no duplicates)
# ---------------------------------------------------------------------------
# The dihedral group D4 has exactly 8 elements:
#   - 4 rotations: 0°, 90°, 180°, 270° (CCW)
#   - 4 reflections: horizontal flip composed with each rotation
#
# We represent each as (rot_k, flip_h) where:
#   rot_k in {0,1,2,3} = number of 90° CCW rotations
#   flip_h in {False, True} = whether to flip horizontally AFTER rotation
#
# This gives 4 × 2 = 8 unique transformations with no overlap.

D4_TRANSFORMS = [
    (0, False),  # identity
    (1, False),  # 90° CCW
    (2, False),  # 180°
    (3, False),  # 270° CCW
    (0, True),   # horizontal flip
    (1, True),   # 90° CCW + flip = anti-diagonal reflection
    (2, True),   # 180° + flip = vertical flip
    (3, True),   # 270° CCW + flip = diagonal reflection
]


def apply_d4_transform_np(arr: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Apply one of the 8 unique D4 transforms to a numpy array.

    Args:
        arr: Array of shape [..., H, W] (last two dims are spatial)
        transform_idx: Integer 0-7 selecting which D4 element to apply

    Returns:
        Transformed array with same shape (H,W may swap for 90°/270° rotations)
    """
    rot_k, flip_h = D4_TRANSFORMS[transform_idx % 8]

    # Rotate by k*90° CCW (axes=(-2,-1) for last two spatial dims)
    if rot_k != 0:
        arr = np.rot90(arr, k=rot_k, axes=(-2, -1))

    # Flip horizontally (along last axis = W)
    if flip_h:
        arr = np.flip(arr, axis=-1)

    return np.ascontiguousarray(arr)


def apply_d4_transform_torch(x: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """
    Apply one of the 8 unique D4 transforms to a torch tensor.

    Args:
        x: Tensor of shape [..., H, W] (last two dims are spatial)
        transform_idx: Integer 0-7 selecting which D4 element to apply

    Returns:
        Transformed tensor with same shape (H,W may swap for 90°/270° rotations)
    """
    rot_k, flip_h = D4_TRANSFORMS[transform_idx % 8]

    # Rotate by k*90° CCW (dims=(-2,-1) for last two spatial dims)
    if rot_k != 0:
        x = torch.rot90(x, k=rot_k, dims=(-2, -1))

    # Flip horizontally (along last dim = W)
    if flip_h:
        x = torch.flip(x, dims=[-1])

    return x


def phase_anchor_roi(
    ez_r,
    ez_i,
    eps_r=None,
    pml_cells=30,
    margin=2,
    roi_x=(40, 140),
    thr_eps=3.0,
    eps=1e-8,
):
    """
    Anchor global phase using a stable ROI near the input waveguide.
    Optionally uses eps_r to restrict to core (eps_r > thr_eps).
    """
    H, W = ez_r.shape
    p2 = min(pml_cells + margin, max(0, H // 2 - 1), max(0, W // 2 - 1))

    x0, x1 = roi_x
    x0 = max(x0, p2)
    x1 = min(x1, W - p2)
    y0, y1 = p2, H - p2

    # If ROI collapses (e.g. small cropped grids), fall back to a central window.
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        x0 = max(p2, int(0.25 * W))
        x1 = min(W - p2, int(0.55 * W))
        y0 = max(p2, int(0.25 * H))
        y1 = min(H - p2, int(0.75 * H))

    Ez = ez_r + 1j * ez_i
    roi = Ez[y0:y1, x0:x1]

    if eps_r is not None:
        roi_eps = eps_r[y0:y1, x0:x1]
        m = (roi_eps > thr_eps).astype(np.float32)
    else:
        m = np.ones_like(roi.real, dtype=np.float32)

    A = np.abs(roi)
    w = (A**2) * m
    u = roi / np.maximum(A, eps)

    c = np.sum(w * u)
    phi = np.angle(c + 1j * 0.0)

    rot = np.exp(-1j * phi)
    Ez2 = Ez * rot
    return Ez2.real.astype(ez_r.dtype), Ez2.imag.astype(ez_i.dtype), float(phi)


def phase_anchor_mask(
    ez_r,
    ez_i,
    mask,
    eps_r=None,
    thr_eps=3.0,
    eps=1e-8,
):
    """
    Anchor global phase using a provided pixel mask (e.g. src_mask / input port mask).
    Rotation-invariant (works if inputs rotate).

    Strategy:
      - compute unit phasors u = E / |E|
      - weight by |E|^2 and by provided mask (and optionally core mask eps_r > thr_eps)
      - choose phi = arg(sum(w * u)), rotate field by exp(-j phi)
    """
    Ez = ez_r + 1j * ez_i

    if mask is None:
        return Ez.real.astype(ez_r.dtype), Ez.imag.astype(ez_i.dtype), 0.0

    m = np.asarray(mask)
    if m.ndim != 2:
        m = np.squeeze(m)
    if m.ndim != 2 or m.shape != ez_r.shape:
        return Ez.real.astype(ez_r.dtype), Ez.imag.astype(ez_i.dtype), 0.0

    w = (m > 0.0).astype(np.float32)
    if eps_r is not None:
        w = w * (np.asarray(eps_r) > thr_eps).astype(np.float32)

    if float(w.sum()) < 4.0:
        return Ez.real.astype(ez_r.dtype), Ez.imag.astype(ez_i.dtype), 0.0

    A = np.abs(Ez)
    u = Ez / np.maximum(A, eps)
    ww = (A**2) * w
    c = np.sum(ww * u)

    if np.abs(c) < eps:
        phi = 0.0
    else:
        phi = float(np.angle(c + 1j * 0.0))

    rot = np.exp(-1j * phi)
    Ez2 = Ez * rot
    return Ez2.real.astype(ez_r.dtype), Ez2.imag.astype(ez_i.dtype), float(phi)


class FDTDDataset(Dataset):
    """
    Unified dataset that pulls samples from all sweep subdirectories under a root.
    Supports:
      - per-sample folders
      - packed shard .npz files created by FDTD/pack_dataset.py
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_fraction: float = 0.9,
        stats: Optional[Dict[str, float]] = None,
        normalize_eps: bool = True,
        use_double: bool = False,
        include_sdf: bool = False,
        normalize_sdf: bool = True,
        sdf_thr_eps: float = 3.0,
        sdf_dx_um: float = 1.0 / 30.0,
        sdf_dy_um: Optional[float] = None,
        sdf_feature: str = "raw",          # raw|exp|clip|clip_exp
        sdf_sigma_nm: float = 100.0,
        include_sweeps: Optional[Iterable[str]] = None,
        exclude_devices: Optional[Iterable[str]] = None,
        include_wavelengths: Optional[Iterable[float]] = None,
        use_shards: bool = False,
        shard_subdir: str = "shards",
        shard_index_name: str = "index.json",
        return_aux: bool = False,
        # --- Optional deterministic subset selection (applied before split) ---
        subset_train_per_sweep: Optional[Dict[str, int]] = None,
        subset_val_per_sweep: Optional[Dict[str, int]] = None,
        subset_seed: int = 0,
        # --- PML handling ---
        crop_pml: bool = False,
        pml_cells: Optional[int] = None,
        # --- Determinism knobs ---
        split_seed: int = 0,               # deterministic train/val split shuffle
        stats_seed: Optional[int] = None,  # deterministic stats subset; defaults to split_seed if None
        # --- Augmentation ---
        augment: bool = False,             # D4 symmetry augmentation (8 unique transforms)
        # --- Pre-computed splits (unified_sweep format) ---
        use_index_split: bool = False,     # Use split field from index.json instead of train_fraction
        # --- Corrupt/Bad sample handling ---
        skip_bad_samples: bool = True,     # Skip corrupted samples instead of crashing
        bad_sample_max_retries: int = 10,  # Max retries before failing
        skip_missing_shards: bool = False, # Drop index entries whose shard file is absent on disk
        # --- Joint training: S-param conditioning ---
        include_sparams_cond: bool = False, # Append S-params to conditioning vector
        # --- Mixed-domain center padding ---
        canvas_hw: Optional[Tuple[int, int]] = None,  # (H, W) target size; center-pad smaller samples
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.train_fraction = float(train_fraction)
        self.use_index_split = bool(use_index_split)
        self.normalize_eps = bool(normalize_eps)
        self.use_double = bool(use_double)
        self.use_shards = bool(use_shards)
        self.shard_subdir = str(shard_subdir)
        self.shard_index_name = str(shard_index_name)
        self.return_aux = bool(return_aux)
        self.exclude_devices: Optional[set] = set(exclude_devices) if exclude_devices else None
        self.include_wavelengths: Optional[List[float]] = list(include_wavelengths) if include_wavelengths else None

        self.include_sdf = bool(include_sdf)
        self.normalize_sdf = bool(normalize_sdf)
        self.sdf_thr_eps = float(sdf_thr_eps)
        self.sdf_dx_um = float(sdf_dx_um)
        self.sdf_dy_um = float(sdf_dx_um if sdf_dy_um is None else sdf_dy_um)

        self.sdf_feature = str(sdf_feature).strip().lower()
        if self.sdf_feature not in ("raw", "exp", "clip", "clip_exp"):
            raise ValueError(f"sdf_feature must be one of ['raw','exp','clip','clip_exp'], got: {sdf_feature}")
        self.sdf_sigma_nm = float(sdf_sigma_nm)
        if self.sdf_sigma_nm <= 0:
            raise ValueError(f"sdf_sigma_nm must be > 0, got: {sdf_sigma_nm}")

        self.subset_train_per_sweep = subset_train_per_sweep
        self.subset_val_per_sweep = subset_val_per_sweep
        self.subset_seed = int(subset_seed)

        self.crop_pml = bool(crop_pml)
        self.pml_cells = 0 if pml_cells is None else int(pml_cells)

        self.split_seed = int(split_seed)
        self.stats_seed = int(split_seed if stats_seed is None else stats_seed)

        self.augment = bool(augment)

        # Number of SDF channels appended after [Ezr, Ezi, eps, src]
        self.sdf_n_channels: int = 0
        self.sdf_feature_names: List[str] = []
        if self.include_sdf:
            if self.sdf_feature == "clip_exp":
                self.sdf_n_channels = 2
                self.sdf_feature_names = ["sdf_clip", "sdf_exp"]
            elif self.sdf_feature == "raw":
                self.sdf_n_channels = 1
                self.sdf_feature_names = ["sdf_raw_nm"]
            elif self.sdf_feature == "clip":
                self.sdf_n_channels = 1
                self.sdf_feature_names = ["sdf_clip"]
            else:
                self.sdf_n_channels = 1
                self.sdf_feature_names = ["sdf_exp"]

        self.x_channels: int = 4 + (self.sdf_n_channels if self.include_sdf else 0)

        self._shard_cache: Dict[Path, np.lib.npyio.NpzFile] = {}
        self._cache_owner_pid: int = os.getpid()
        self._bad_shards: Set[Path] = set()
        self._bad_sample_count = 0
        self._retry_rng = np.random.default_rng(self.split_seed + 12345)

        self.skip_bad_samples = bool(skip_bad_samples)
        self.bad_sample_max_retries = max(1, int(bad_sample_max_retries))
        self.skip_missing_shards = bool(skip_missing_shards)

        # Mixed-domain center padding: pad all samples to a common canvas size.
        # E.g., canvas_hw=(320, 480) pads 160×480 rect and 320×320 sq to 320×480.
        self.canvas_hw: Optional[Tuple[int, int]] = (
            (int(canvas_hw[0]), int(canvas_hw[1])) if canvas_hw is not None else None
        )

        # Conditioning: wavelength only (geometry is in the spatial eps map)
        self.cond_param_names: List[str] = []
        self.device_type_names: List[str] = []  # not used
        self.include_sparams_cond = bool(include_sparams_cond)
        # Base cond_dim: wavelength only
        base_cond_dim = 1  # wavelength
        if self.include_sparams_cond:
            # +8 for Re/Im of 4 S-params, +4 for port_valid flags = 12 extra
            self.cond_dim: int = base_cond_dim + 12  # 13
        else:
            self.cond_dim: int = base_cond_dim  # 1

        self.max_ports: int = 4

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

        # Collect refs - optionally filter by pre-computed split from index
        if self.use_shards:
            if self.use_index_split:
                # Use pre-computed splits from index.json (unified_sweep format)
                all_refs: List[SampleRef] = self._collect_shard_refs(sweep_dirs, target_split=split, exclude_devices=self.exclude_devices, include_wavelengths=self.include_wavelengths)
            else:
                all_refs = self._collect_shard_refs(sweep_dirs, exclude_devices=self.exclude_devices, include_wavelengths=self.include_wavelengths)
        else:
            all_refs = self._collect_folder_refs(sweep_dirs)

        # Deterministic shuffle before split to avoid sweep-order bias.
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_refs))
        all_refs = [all_refs[i] for i in perm]

        # If using index splits, we already filtered - just use all refs
        if self.use_index_split and self.use_shards:
            self.sample_refs = all_refs
        # Optional deterministic subset selection (applied before split).
        elif (self.subset_train_per_sweep is not None) or (self.subset_val_per_sweep is not None):
            self.sample_refs = self._subset_refs_by_sweep(
                all_refs,
                split=split,
                train_per_sweep=self.subset_train_per_sweep or {},
                val_per_sweep=self.subset_val_per_sweep or {},
                seed=self.subset_seed,
            )
        else:
            n_total = len(all_refs)
            n_train = int(round(self.train_fraction * n_total))
            if split == "train":
                self.sample_refs = all_refs[:n_train]
            elif split == "val":
                self.sample_refs = all_refs[n_train:]
            else:
                raise ValueError("split must be 'train' or 'val'")

        if len(self.sample_refs) == 0:
            raise RuntimeError(f"No samples in {split} split (train_fraction={train_fraction})")

        # Infer canonical spatial orientation once so mixed HxW / WxH samples can be normalized.
        self.target_hw, self._shape_hist = self._infer_target_hw(max_probe=64)

        if stats is None:
            if split != "train":
                raise ValueError("stats must be provided for non-train splits")
            print(f"[{split}] computing normalization stats from subset...")
            self.stats = self._compute_stats(seed=self.stats_seed)
            # Important for multi-worker DataLoader: avoid inheriting open NPZ handles after fork.
            self._clear_shard_cache()
        else:
            self.stats = stats

        # Allow stats (from older checkpoints) to override if present
        try:
            self.cond_dim = int(self.stats.get("cond_dim", self.cond_dim))
        except Exception:
            pass

        print(f"[{split}] dataset size = {len(self.sample_refs)}")
        if self.target_hw is not None and len(self._shape_hist) > 1:
            print(
                f"[{split}] detected mixed sample shapes {dict(self._shape_hist)}; "
                f"canonicalizing to HxW={self.target_hw}"
            )

    # -----------------------
    # Helpers: refs + splits
    # -----------------------

    def _infer_sweep_name_from_ref(self, ref: SampleRef) -> str:
        if isinstance(ref, tuple):
            shard_path, _, _ = ref
            try:
                return shard_path.parent.parent.name
            except Exception:
                return ""
        try:
            return Path(ref).parent.name
        except Exception:
            return ""

    def _tag_from_ref(self, ref: SampleRef) -> str:
        if isinstance(ref, tuple):
            _, _, tag = ref
            return str(tag)
        return Path(ref).name

    def _geom_key_from_tag(self, tag: str) -> str:
        t = str(tag)
        if "__inPort" in t:
            return t.split("__inPort", 1)[0]
        return t

    def _stable_hash_u32(self, s: str) -> int:
        return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)

    def _infer_target_hw(self, max_probe: int = 64) -> Tuple[Optional[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """
        Probe a subset of samples to infer canonical (H, W).
        This makes loading robust when a dataset accidentally mixes transposed samples.

        When canvas_hw is set, returns the canvas size as the target (mixed domains
        are handled by center-padding, not by transposing).
        """
        hist: Dict[Tuple[int, int], int] = {}
        n_probe = min(int(max_probe), len(self.sample_refs))
        for i in range(n_probe):
            ref = self.sample_refs[i]
            try:
                ez_r, _, _, _, _ = self._load_raw(ref, dtype_np=np.float32)
                pml_px = self._pml_px_from_ref(ref)
                (ez_r_c,) = self._maybe_crop_pml_arrays(pml_px, ez_r)
                hw = (int(ez_r_c.shape[0]), int(ez_r_c.shape[1]))
                hist[hw] = hist.get(hw, 0) + 1
            except Exception:
                continue
        if not hist:
            return None, {}
        # When canvas_hw is set, use it as the canonical target (padding handles size mismatch)
        if self.canvas_hw is not None:
            return self.canvas_hw, hist
        target = sorted(hist.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1], kv[0][0], kv[0][1]), reverse=True)[0][0]
        return target, hist

    def _canonicalize_hw(
        self,
        ez_r: np.ndarray,
        ez_i: np.ndarray,
        eps: np.ndarray,
        src: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensure arrays match canonical (H, W). If exactly transposed, swap axes.
        """
        if self.target_hw is None:
            return ez_r, ez_i, eps, src
        h, w = int(ez_r.shape[0]), int(ez_r.shape[1])
        th, tw = self.target_hw
        if (h, w) == (th, tw):
            return ez_r, ez_i, eps, src
        if (h, w) == (tw, th):
            return ez_r.T.copy(), ez_i.T.copy(), eps.T.copy(), src.T.copy()
        # With canvas_hw, samples may have different native sizes — skip strict check
        if self.canvas_hw is not None:
            return ez_r, ez_i, eps, src
        raise ValueError(f"Unexpected sample shape {(h, w)}; expected {(th, tw)} or {(tw, th)}")

    @staticmethod
    def _center_pad_2d(arr: np.ndarray, target_h: int, target_w: int,
                       pad_value: float = 0.0) -> np.ndarray:
        """Center-pad a 2D array [H, W] to (target_h, target_w)."""
        h, w = arr.shape
        if h == target_h and w == target_w:
            return arr
        if h > target_h or w > target_w:
            raise ValueError(f"Array {(h, w)} exceeds canvas {(target_h, target_w)}")
        out = np.full((target_h, target_w), pad_value, dtype=arr.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        out[y0:y0 + h, x0:x0 + w] = arr
        return out

    @staticmethod
    def _center_pad_3d(arr: np.ndarray, target_h: int, target_w: int,
                       pad_value: float = 0.0) -> np.ndarray:
        """Center-pad a 3D array [N, H, W] to [N, target_h, target_w]."""
        n, h, w = arr.shape
        if h == target_h and w == target_w:
            return arr
        if h > target_h or w > target_w:
            raise ValueError(f"Array {(n, h, w)} exceeds canvas {(target_h, target_w)}")
        out = np.full((n, target_h, target_w), pad_value, dtype=arr.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        out[:, y0:y0 + h, x0:x0 + w] = arr
        return out

    def _subset_refs_by_sweep(
        self,
        all_refs: List[SampleRef],
        *,
        split: str,
        train_per_sweep: Dict[str, int],
        val_per_sweep: Dict[str, int],
        seed: int,
    ) -> List[SampleRef]:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # Group refs by sweep -> geom_key -> [refs...]
        by_sweep: Dict[str, Dict[str, List[SampleRef]]] = {}
        for r in all_refs:
            sweep = self._infer_sweep_name_from_ref(r)
            tag = self._tag_from_ref(r)
            gk = self._geom_key_from_tag(tag)
            by_sweep.setdefault(sweep, {}).setdefault(gk, []).append(r)

        requested_sweeps = set(train_per_sweep.keys()) | set(val_per_sweep.keys())
        if not requested_sweeps:
            raise ValueError("subset_train_per_sweep/subset_val_per_sweep provided but empty; specify at least one sweep.")

        out_refs: List[SampleRef] = []

        for sweep in sorted(requested_sweeps):
            geom_map = by_sweep.get(sweep, {})
            if not geom_map:
                raise RuntimeError(
                    f"Requested subset sweep '{sweep}' not found under {self.root}. Available: {sorted(by_sweep.keys())}"
                )

            # Stable per-sweep RNG seed (no Python hash randomization).
            sweep_seed = int(seed) + self._stable_hash_u32(sweep) % 1_000_000
            rng = np.random.default_rng(sweep_seed)

            geom_keys = list(geom_map.keys())
            rng.shuffle(geom_keys)

            n_train_target = int(train_per_sweep.get(sweep, 0))
            n_val_target = int(val_per_sweep.get(sweep, 0))
            if n_train_target < 0 or n_val_target < 0:
                raise ValueError("subset train/val counts must be >= 0")

            # Select train geoms until we hit the sample target.
            train_geoms: List[str] = []
            train_count = 0
            i = 0
            while i < len(geom_keys) and train_count < n_train_target:
                gk = geom_keys[i]
                k = len(geom_map[gk])
                if train_count + k <= n_train_target:
                    train_geoms.append(gk)
                    train_count += k
                i += 1

            if train_count != n_train_target:
                raise RuntimeError(
                    f"Could not satisfy train subset for sweep '{sweep}': requested {n_train_target} samples, "
                    f"but only reached {train_count}. (Try a multiple of 2 for couplers, or increase dataset size.)"
                )

            # Continue selecting disjoint val geoms.
            val_geoms: List[str] = []
            val_count = 0
            while i < len(geom_keys) and val_count < n_val_target:
                gk = geom_keys[i]
                k = len(geom_map[gk])
                if val_count + k <= n_val_target:
                    val_geoms.append(gk)
                    val_count += k
                i += 1

            if val_count != n_val_target:
                raise RuntimeError(
                    f"Could not satisfy val subset for sweep '{sweep}': requested {n_val_target} samples, "
                    f"but only reached {val_count}. (Try a multiple of 2 for couplers, or increase dataset size.)"
                )

            chosen = train_geoms if split == "train" else val_geoms
            for gk in chosen:
                out_refs.extend(geom_map[gk])

        out_refs.sort(key=lambda r: self._tag_from_ref(r))
        return out_refs

    def _collect_folder_refs(self, sweep_dirs: List[Path]) -> List[Path]:
        all_dirs: List[Path] = []
        for sdir in sweep_dirs:
            for d in sorted(sdir.iterdir()):
                if not d.is_dir():
                    continue
                # Do not treat the shard folder as a sample when using folder-mode.
                if d.name == self.shard_subdir:
                    continue
                # Basic sanity: prefer directories that look like samples.
                # (If you have other non-sample dirs, this avoids accidental inclusion.)
                if (d / "Ez_real.npy").is_file() or (d / "Ez_imag.npy").is_file() or (d / "eps.npy").is_file():
                    all_dirs.append(d)
        if not all_dirs:
            raise RuntimeError(f"No sample subfolders found in sweeps {sweep_dirs}")
        return all_dirs

    def _collect_shard_refs(self, sweep_dirs: List[Path], target_split: Optional[str] = None,
                            exclude_devices: Optional[set] = None,
                            include_wavelengths: Optional[List[float]] = None) -> List[ShardRef]:
        """
        Collect shard references from all sweep directories.

        Args:
            sweep_dirs: List of sweep directory paths to scan
            target_split: If provided, only include entries where index["split"] matches this value.
                         Supports "train", "val", "test". For "val", also accepts "val" entries.
                         (unified_sweep format uses pre-computed splits)
            exclude_devices: If provided, skip entries whose "device" field is in this set.
            include_wavelengths: If provided, only include entries whose wavelength_um is
                                within 0.01 µm of one of these values.
        """
        refs: List[ShardRef] = []
        for sdir in sweep_dirs:
            shard_dir = sdir / self.shard_subdir
            index_path = shard_dir / self.shard_index_name
            if not index_path.is_file():
                raise RuntimeError(f"Missing shard index {index_path}")
            with open(index_path, "r") as f:
                entries = json.load(f)
            shard_present_cache: Dict[str, bool] = {}
            skipped_missing = 0
            for e in entries:
                # Filter by split if target_split is specified
                if target_split is not None:
                    entry_split = e.get("split", "")
                    if entry_split != target_split:
                        continue

                # Filter by device type
                if exclude_devices is not None:
                    dev = e.get("device", "")
                    if dev in exclude_devices:
                        continue

                # Filter by wavelength
                if include_wavelengths is not None:
                    wl = e.get("wavelength_um", None)
                    if wl is None or not any(abs(wl - target) < 0.01 for target in include_wavelengths):
                        continue

                shard_name = e["shard"]
                shard_path = shard_dir / shard_name
                if self.skip_missing_shards:
                    present = shard_present_cache.get(shard_name)
                    if present is None:
                        present = shard_path.is_file()
                        shard_present_cache[shard_name] = present
                    if not present:
                        skipped_missing += 1
                        continue
                slot = int(e["slot"])
                # Use tag if present, otherwise geometry_id (unified_sweep format)
                tag = e.get("tag", e.get("geometry_id", ""))
                refs.append((shard_path, slot, tag))
            if self.skip_missing_shards and skipped_missing > 0:
                missing = sorted(s for s, ok in shard_present_cache.items() if not ok)
                print(
                    f"[dataset.py] skip_missing_shards: dropped {skipped_missing} entries "
                    f"across {len(missing)} missing shard(s) under {shard_dir} "
                    f"(e.g. {missing[:3]})"
                )
        if not refs:
            if target_split is not None:
                raise RuntimeError(f"No shard entries found for split='{target_split}' under sweeps {sweep_dirs}")
            raise RuntimeError(f"No shard entries found under sweeps {sweep_dirs}")
        refs.sort(key=lambda r: (r[2], r[0].name, r[1]))
        return refs

    # -----------------------
    # Shard IO
    # -----------------------

    def _mark_bad_shard(self, shard_path: Path, err: Exception) -> None:
        self._bad_shards.add(shard_path)
        if shard_path in self._shard_cache:
            try:
                del self._shard_cache[shard_path]
            except Exception:
                pass
        if self.skip_bad_samples:
            print(f"[dataset.py] WARNING: marking shard as bad: {shard_path} ({type(err).__name__}: {err})")

    def _clear_shard_cache(self) -> None:
        for _p, fh in list(self._shard_cache.items()):
            try:
                fh.close()
            except Exception:
                pass
        self._shard_cache.clear()

    def _ensure_cache_pid(self) -> None:
        """
        Make shard cache fork-safe.
        If DataLoader workers fork after cache is populated, inherited file handles can cause
        random BadZipFile errors. Reopen cache per-process.
        """
        pid = os.getpid()
        if pid != self._cache_owner_pid:
            self._clear_shard_cache()
            self._cache_owner_pid = pid

    def _get_shard_file(self, shard_path: Path, max_retries: int = 3) -> np.lib.npyio.NpzFile:
        self._ensure_cache_pid()
        if shard_path in self._bad_shards:
            raise zipfile.BadZipFile(f"Skipping previously marked bad shard: {shard_path}")
        if shard_path not in self._shard_cache:
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            # Retry logic for transient filesystem errors on HPC shared storage
            last_error = None
            for attempt in range(max_retries):
                try:
                    self._shard_cache[shard_path] = np.load(shard_path, allow_pickle=False)
                    break
                except Exception as e:
                    if isinstance(e, zipfile.BadZipFile):
                        self._mark_bad_shard(shard_path, e)
                        raise
                    last_error = e
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1 * (attempt + 1))  # exponential backoff
                        # Clear any cached bad state
                        if shard_path in self._shard_cache:
                            del self._shard_cache[shard_path]
            else:
                raise last_error  # re-raise if all retries failed
        return self._shard_cache[shard_path]

    def _load_raw_from_shard(self, ref: ShardRef, dtype_np):
        shard_path, slot, _ = ref
        data = self._get_shard_file(shard_path)
        prefix = f"s{slot}/"

        def get_arr(key: str, required: bool = True):
            full_key = prefix + key
            if full_key not in data:
                if required:
                    raise KeyError(f"Missing key {full_key} in shard {shard_path}")
                return None
            return data[full_key].astype(dtype_np)

        ez_r = get_arr("Ez_real")
        ez_i = get_arr("Ez_imag")
        eps = get_arr("eps")
        src = get_arr("src_mask", required=False)
        if src is None:
            src = np.zeros_like(eps)

        # Wavelength: try multiple key paths for compatibility
        lam_arr = get_arr("sparams/wavelength_um", required=False)
        if lam_arr is None:
            lam_arr = get_arr("sparams/lambda_um", required=False)
        if lam_arr is None:
            lam_arr = get_arr("wavelength_um", required=False)  # unified_sweep format
        if lam_arr is None:
            raise KeyError(f"Missing wavelength key in shard {shard_path} slot {slot}")
        lam_um = float(lam_arr)

        return ez_r, ez_i, eps, src, lam_um

    def _load_aux_from_shard(self, ref: ShardRef, dtype_np):
        shard_path, slot, _ = ref
        data = self._get_shard_file(shard_path)
        prefix = f"s{slot}/"

        def get_arr(key: str):
            full_key = prefix + key
            if full_key not in data:
                return None
            return data[full_key]

        # S-parameters: try combined format first, then individual S-params (unified_sweep format)
        Sr = get_arr("sparams/S_real")
        Si = get_arr("sparams/S_imag")
        if Sr is not None and Si is not None:
            Sr = Sr.astype(dtype_np)
            Si = Si.astype(dtype_np)
            sparams_true = (Sr + 1j * Si).astype(np.complex128 if dtype_np == np.float64 else np.complex64)
        else:
            # Try unified_sweep format: individual S-params like sparams/S11_real, sparams/S21_real, etc.
            sparams_true = self._load_individual_sparams(data, prefix, dtype_np)

        # Port masks: try ports/masks first, then port_masks (unified_sweep format)
        port_masks = get_arr("ports/masks")
        if port_masks is None:
            port_masks = get_arr("port_masks")  # unified_sweep format
        if port_masks is not None:
            port_masks = port_masks.astype(dtype_np)

        # Port IDs: try ports/ids first, then port_ids (unified_sweep format)
        port_ids = get_arr("ports/ids")
        if port_ids is None:
            port_ids = get_arr("port_ids")  # unified_sweep format
        if port_ids is not None:
            port_ids = port_ids.astype(np.int32)

        # Input port: try sparams/input_port first, then input_port (unified_sweep format)
        in_port_idx = None
        in_port = get_arr("sparams/input_port")
        if in_port is None:
            in_port = get_arr("input_port")  # unified_sweep format
        if in_port is not None:
            try:
                in_port_val = int(np.array(in_port).item())
            except Exception:
                in_port_val = None
            if in_port_val is not None:
                if port_ids is not None:
                    hits = np.where(port_ids == in_port_val)[0]
                    if len(hits) > 0:
                        in_port_idx = int(hits[0])
                else:
                    in_port_idx = int(in_port_val - 1)

        return {
            "sparams_true": sparams_true,
            "in_port_idx": in_port_idx,
            "port_masks": port_masks,
            "port_ids": port_ids,
        }

    def _load_individual_sparams(self, data, prefix: str, dtype_np):
        """
        Load individual S-parameters from unified_sweep format.
        Looks for keys like sparams/S11_real, sparams/S21_real, etc.
        Returns a 1D complex array ordered by port number.
        """
        # Find all S-param keys in this slot
        sparam_keys = []
        for key in data.files:
            if key.startswith(prefix + "sparams/S") and key.endswith("_real"):
                # Extract the S-param name (e.g., "S11", "S21")
                sparam_name = key[len(prefix + "sparams/"):-len("_real")]
                if len(sparam_name) >= 2 and sparam_name[0] == "S":
                    sparam_keys.append(sparam_name)

        if not sparam_keys:
            return None

        # Sort by output port number (first digit after S)
        def sort_key(s):
            try:
                return int(s[1])  # S11 -> 1, S21 -> 2, etc.
            except (IndexError, ValueError):
                return 999
        sparam_keys = sorted(set(sparam_keys), key=sort_key)

        # Build complex array
        sparams = []
        for sname in sparam_keys:
            real_key = prefix + f"sparams/{sname}_real"
            imag_key = prefix + f"sparams/{sname}_imag"
            if real_key in data and imag_key in data:
                sr = float(data[real_key])
                si = float(data[imag_key])
                sparams.append(complex(sr, si))

        if not sparams:
            return None

        ctype = np.complex128 if dtype_np == np.float64 else np.complex64
        return np.array(sparams, dtype=ctype)

    # -----------------------
    # Folder IO
    # -----------------------

    def _load_raw_from_dir(self, d: Path, dtype_np):
        ez_r = np.load(d / "Ez_real.npy").astype(dtype_np)
        ez_i = np.load(d / "Ez_imag.npy").astype(dtype_np)
        eps = np.load(d / "eps.npy").astype(dtype_np)

        src_path = d / "src_mask.npy"
        if src_path.is_file():
            src = np.load(src_path).astype(dtype_np)
        else:
            src = np.zeros_like(eps)

        sp = np.load(d / "sparams.npz")

        if "wavelength_um" in sp:
            lam_um = float(sp["wavelength_um"])
        elif "lambda_um" in sp:
            lam_um = float(sp["lambda_um"])
        else:
            raise KeyError("sparams.npz missing wavelength key (expected 'wavelength_um' or 'lambda_um')")

        return ez_r, ez_i, eps, src, lam_um

    def _load_aux_from_dir(self, d: Path, dtype_np):
        aux = {"sparams_true": None, "in_port_idx": None, "port_masks": None, "port_ids": None}

        sp_path = d / "sparams.npz"
        if sp_path.is_file():
            sp = np.load(sp_path)
            if ("S_real" in sp) and ("S_imag" in sp):
                Sr = sp["S_real"].astype(dtype_np)
                Si = sp["S_imag"].astype(dtype_np)
                aux["sparams_true"] = (Sr + 1j * Si).astype(np.complex128 if dtype_np == np.float64 else np.complex64)
            elif ("s11" in sp) and ("s21" in sp):
                s11 = sp["s11"].item()
                s21 = sp["s21"].item()
                sparams_vec = np.array([s11, s21], dtype=np.complex64)
                aux["sparams_true"] = sparams_vec.astype(np.complex128 if dtype_np == np.float64 else np.complex64)

            if "input_port" in sp:
                try:
                    aux["in_port_idx"] = int(sp["input_port"]) - 1
                except Exception:
                    aux["in_port_idx"] = None

        pm_path = d / "port_masks.npy"
        if pm_path.is_file():
            aux["port_masks"] = np.load(pm_path).astype(dtype_np)

        pid_path = d / "port_ids.npy"
        if pid_path.is_file():
            aux["port_ids"] = np.load(pid_path).astype(np.int32)

        return aux

    def _load_raw(self, ref: SampleRef, dtype_np):
        if isinstance(ref, tuple):
            return self._load_raw_from_shard(ref, dtype_np)
        return self._load_raw_from_dir(ref, dtype_np)

    def _load_aux(self, ref: SampleRef, dtype_np):
        if isinstance(ref, tuple):
            return self._load_aux_from_shard(ref, dtype_np)
        return self._load_aux_from_dir(ref, dtype_np)

    # -----------------------
    # Device type + params
    # -----------------------

    def _infer_device_type_from_ref(self, ref: SampleRef) -> str:
        if isinstance(ref, tuple):
            shard_path, slot, _ = ref
            data = self._get_shard_file(shard_path)
            # Try the 'device' key first (unified_sweep format), then 'dataset'
            for key_suffix in ("device", "dataset"):
                key = f"s{slot}/{key_suffix}"
                if key in data:
                    v = data[key]
                    try:
                        item = np.array(v).item()
                    except Exception:
                        item = v
                    if isinstance(item, bytes):
                        item = item.decode("utf-8", errors="ignore")
                    return str(item)
            return ""
        try:
            parent = Path(ref).parent.name.lower()
        except Exception:
            return ""
        if "coupler" in parent or "directional_coupler" in parent:
            return "directional_coupler"
        if "y_branch" in parent or "ybranch" in parent:
            return "ybranch"
        if "sbend" in parent or "s_bend" in parent:
            return "sbend"
        if "taper" in parent:
            return "taper"
        if "straight" in parent:
            return "straight"
        if "euler" in parent or "zig" in parent:
            return "euler_bend"
        if "circular" in parent:
            return "circular_bend"
        if "crossing" in parent:
            return "crossing"
        return ""

    def _get_param_value_from_ref(self, ref: SampleRef, name: str) -> Optional[float]:
        alias_map = {
            "wg_length_um": ["Lc_um"],
        }
        names_to_try = [name] + alias_map.get(name, [])

        if isinstance(ref, tuple):
            shard_path, slot, _ = ref
            data = self._get_shard_file(shard_path)
            prefix = f"s{slot}/"

            # Try multiple key paths: sparams/{name}, params/{name}, {name}
            for n in names_to_try:
                for key_path in [f"sparams/{n}", f"params/{n}", n]:
                    full_key = prefix + key_path
                    if full_key in data:
                        try:
                            return float(np.array(data[full_key]).item())
                        except Exception:
                            try:
                                return float(data[full_key])
                            except Exception:
                                continue
            return None

        d = Path(ref)
        sp_path = d / "sparams.npz"
        if not sp_path.is_file():
            return None
        try:
            sp = np.load(sp_path)
        except Exception:
            return None
        for n in names_to_try:
            if n not in sp:
                continue
            try:
                return float(np.array(sp[n]).item())
            except Exception:
                try:
                    return float(sp[n])
                except Exception:
                    continue
        return None

    def _build_cond_vector(self, ref: SampleRef, lam_um: float, dtype_np) -> np.ndarray:
        # wavelength (always present)
        lam_std = float(self.stats.get("lambda_um_std", 1.0))
        if lam_std <= 0:
            lam_std = 1.0
        lam_norm = (float(lam_um) - float(self.stats["lambda_um_mean"])) / lam_std

        cond_vals = [lam_norm]

        # Append S-param conditioning placeholders (filled later in _build_sample)
        if self.include_sparams_cond:
            # 8 zeros for Re/Im of 4 S-params + 4 zeros for port_valid flags
            cond_vals.extend([0.0] * 12)

        cond = np.array(cond_vals, dtype=dtype_np)
        return cond

    # -----------------------
    # PML cropping utilities
    # -----------------------

    def _pml_px_from_ref(self, ref: SampleRef) -> int:
        if not self.crop_pml:
            return 0
        try:
            if isinstance(ref, tuple):
                shard_path, slot, _ = ref
                data = self._get_shard_file(shard_path)
                key = f"s{slot}/grid/pml_px"
                if key in data:
                    return int(np.array(data[key]).item())
            else:
                gm_path = Path(ref) / "grid_meta.npz"
                if gm_path.is_file():
                    gm = np.load(gm_path, allow_pickle=True)
                    if "pml_px" in gm:
                        return int(np.array(gm["pml_px"]).item())
        except Exception:
            pass
        return int(self.pml_cells)

    def _safe_crop_slice(self, H: int, W: int, pml_px: int) -> Optional[Tuple[slice, slice]]:
        if pml_px <= 0:
            return None
        # Need at least 1 pixel after cropping.
        if (H <= 2 * pml_px + 1) or (W <= 2 * pml_px + 1):
            return None
        return (slice(pml_px, H - pml_px), slice(pml_px, W - pml_px))

    def _maybe_crop_pml_arrays(self, pml_px: int, *arrays: Optional[np.ndarray]):
        if (not self.crop_pml) or pml_px <= 0:
            return arrays
        out = []
        for a in arrays:
            if a is None:
                out.append(None)
                continue
            if a.ndim == 2:
                sl = self._safe_crop_slice(a.shape[0], a.shape[1], pml_px)
                out.append(a if sl is None else a[sl[0], sl[1]])
            elif a.ndim == 3:
                sl = self._safe_crop_slice(a.shape[1], a.shape[2], pml_px)
                out.append(a if sl is None else a[:, sl[0], sl[1]])
            else:
                out.append(a)
        return tuple(out)

    # -----------------------
    # Grid spacing and SDF
    # -----------------------

    def _grid_spacing_um_from_ref(self, ref: SampleRef) -> Tuple[float, float]:
        dx_um = None
        dy_um = None

        if isinstance(ref, tuple):
            shard_path, slot, _ = ref
            data = self._get_shard_file(shard_path)
            prefix = f"s{slot}/"

            def get_scalar(key: str):
                full_key = prefix + key
                if full_key not in data:
                    return None
                try:
                    return float(np.array(data[full_key]).item())
                except Exception:
                    try:
                        return float(data[full_key])
                    except Exception:
                        return None

            # Try grid/dx first, then dx_um (unified_sweep format)
            dx_um = get_scalar("grid/dx")
            dy_um = get_scalar("grid/dy")
            if dx_um is None:
                dx_um = get_scalar("dx_um")  # unified_sweep format
            if dy_um is None:
                dy_um = get_scalar("dy_um")  # unified_sweep format

            if dx_um is None or dy_um is None:
                # Try computing from Lx/Ly and nx/ny
                Lx_um = get_scalar("grid/Lx_um")
                Ly_um = get_scalar("grid/Ly_um")
                nx = get_scalar("grid/nx")
                ny = get_scalar("grid/ny")
                # Also try unified_sweep format at top level
                if Lx_um is None:
                    Lx_um = get_scalar("Lx_um")
                if Ly_um is None:
                    Ly_um = get_scalar("Ly_um")
                if nx is None:
                    nx = get_scalar("nx")
                if ny is None:
                    ny = get_scalar("ny")
                if (Lx_um is not None) and (Ly_um is not None) and (nx is not None) and (ny is not None):
                    try:
                        dx_um = float(Lx_um) / float(nx)
                        dy_um = float(Ly_um) / float(ny)
                    except Exception:
                        dx_um = None
                        dy_um = None
        else:
            d = Path(ref)
            gm_path = d / "grid_meta.npz"
            if gm_path.is_file():
                try:
                    gm = np.load(gm_path, allow_pickle=True)
                    if "dx" in gm and "dy" in gm:
                        dx_um = float(np.array(gm["dx"]).item())
                        dy_um = float(np.array(gm["dy"]).item())
                    elif ("Lx_um" in gm and "Ly_um" in gm and "nx" in gm and "ny" in gm):
                        Lx_um = float(np.array(gm["Lx_um"]).item())
                        Ly_um = float(np.array(gm["Ly_um"]).item())
                        nx = float(np.array(gm["nx"]).item())
                        ny = float(np.array(gm["ny"]).item())
                        dx_um = float(Lx_um) / float(nx)
                        dy_um = float(Ly_um) / float(ny)
                except Exception:
                    dx_um = None
                    dy_um = None

        if (
            dx_um is None
            or dy_um is None
            or (not np.isfinite(dx_um))
            or (not np.isfinite(dy_um))
            or dx_um <= 0
            or dy_um <= 0
        ):
            return self.sdf_dx_um, self.sdf_dy_um
        return float(dx_um), float(dy_um)

    def _distance_to_feature_nm(self, feature: np.ndarray, *, dx_nm: float, dy_nm: float) -> np.ndarray:
        global _SDF_WARNED
        if _scipy_ndimage is not None:
            return _scipy_ndimage.distance_transform_edt(~feature, sampling=(dy_nm, dx_nm)).astype(np.float32)

        if not _SDF_WARNED:
            print("[dataset.py] WARNING: SciPy not available; using chamfer DT fallback for SDF (slower, approximate).")
            _SDF_WARNED = True

        H, W = feature.shape
        inf = np.float32(1e30)
        dist = np.full((H, W), inf, dtype=np.float32)
        dist[feature.astype(bool, copy=False)] = 0.0

        w_x = np.float32(dx_nm)
        w_y = np.float32(dy_nm)
        w_d = np.float32(np.sqrt(dx_nm * dx_nm + dy_nm * dy_nm))

        for i in range(H):
            for j in range(W):
                v = dist[i, j]
                if i > 0:
                    v = min(v, dist[i - 1, j] + w_y)
                    if j > 0:
                        v = min(v, dist[i - 1, j - 1] + w_d)
                    if j + 1 < W:
                        v = min(v, dist[i - 1, j + 1] + w_d)
                if j > 0:
                    v = min(v, dist[i, j - 1] + w_x)
                dist[i, j] = v

        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                v = dist[i, j]
                if i + 1 < H:
                    v = min(v, dist[i + 1, j] + w_y)
                    if j > 0:
                        v = min(v, dist[i + 1, j - 1] + w_d)
                    if j + 1 < W:
                        v = min(v, dist[i + 1, j + 1] + w_d)
                if j + 1 < W:
                    v = min(v, dist[i, j + 1] + w_x)
                dist[i, j] = v

        return dist

    def _signed_distance_nm_from_eps(self, eps_phys: np.ndarray, *, ref: SampleRef) -> np.ndarray:
        dx_um, dy_um = self._grid_spacing_um_from_ref(ref)
        dx_nm = 1000.0 * float(dx_um)
        dy_nm = 1000.0 * float(dy_um)

        inside = (eps_phys > self.sdf_thr_eps)
        d_out = self._distance_to_feature_nm(inside, dx_nm=dx_nm, dy_nm=dy_nm)
        d_in = self._distance_to_feature_nm(~inside, dx_nm=dx_nm, dy_nm=dy_nm)
        phi = d_out - d_in  # negative inside, positive outside
        return phi.astype(np.float32, copy=False)

    def _sdf_features_from_phi_nm(self, phi_nm: np.ndarray) -> np.ndarray:
        phi_nm = phi_nm.astype(np.float32, copy=False)
        s = float(self.sdf_sigma_nm)
        if s <= 0:
            s = 1.0

        clip = np.clip(phi_nm / s, -1.0, 1.0).astype(np.float32, copy=False)
        exp = (np.sign(phi_nm) * np.exp(-np.abs(phi_nm) / s)).astype(np.float32, copy=False)

        if self.sdf_feature == "raw":
            return phi_nm[None, ...]
        if self.sdf_feature == "clip":
            return clip[None, ...]
        if self.sdf_feature == "exp":
            return exp[None, ...]
        if self.sdf_feature == "clip_exp":
            return np.stack([clip, exp], axis=0).astype(np.float32, copy=False)
        raise ValueError(f"Unknown sdf_feature={self.sdf_feature}")

    # -----------------------
    # Stats
    # -----------------------

    def _compute_stats(self, seed: int = 0) -> Dict[str, float]:
        subset_size = min(len(self.sample_refs), 500)

        rng = np.random.default_rng(int(seed))
        idxs = rng.choice(len(self.sample_refs), subset_size, replace=False)

        ez_r_all = []
        ez_i_all = []
        eps_all = []
        sdf_raw_all = []
        sdf_feat_ch: List[List[np.ndarray]] = [[] for _ in range(int(self.sdf_n_channels))]
        lam_all = []
        param_vals: Dict[str, List[float]] = {k: [] for k in self.cond_param_names}

        for i in idxs:
            ref = self.sample_refs[int(i)]
            ez_r, ez_i, eps, src, lam = self._load_raw(ref, dtype_np=np.float32)

            # Phase anchoring: mask-first, ROI fallback.
            ez_r2, ez_i2, _ = phase_anchor_mask(ez_r, ez_i, (src > 0.5), eps_r=eps, thr_eps=3.0)
            if (
                (src is None)
                or float((src > 0.5).sum()) < 4.0
                or (np.allclose(ez_r2, ez_r) and np.allclose(ez_i2, ez_i))
            ):
                ez_r2, ez_i2, _ = phase_anchor_roi(ez_r, ez_i, eps_r=eps, pml_cells=int(self.pml_cells), margin=2)
            ez_r, ez_i = ez_r2, ez_i2

            ez_r_all.append(ez_r)
            ez_i_all.append(ez_i)
            eps_all.append(eps)
            lam_all.append(lam)

            if self.include_sdf:
                try:
                    phi_nm = self._signed_distance_nm_from_eps(eps, ref=ref)
                    sdf_raw_all.append(phi_nm)
                    feats = self._sdf_features_from_phi_nm(phi_nm)  # [C,H,W]
                    if feats.ndim == 3 and feats.shape[0] == int(self.sdf_n_channels):
                        for c in range(int(self.sdf_n_channels)):
                            sdf_feat_ch[c].append(feats[c])
                except Exception:
                    pass

            for name in self.cond_param_names:
                v = self._get_param_value_from_ref(ref, name)
                if v is None or (not np.isfinite(v)):
                    continue
                param_vals[name].append(float(v))

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

        if self.include_sdf and len(sdf_raw_all) > 0:
            sdf_nm_cat = np.stack(sdf_raw_all)
            stats["sdf_nm_mean"] = float(sdf_nm_cat.mean())
            stats["sdf_nm_std"] = float(sdf_nm_cat.std()) + 1e-8
        else:
            stats["sdf_nm_mean"] = 0.0
            stats["sdf_nm_std"] = 1.0

        if self.include_sdf and int(self.sdf_n_channels) > 0:
            for c in range(int(self.sdf_n_channels)):
                if len(sdf_feat_ch[c]) > 0:
                    cat = np.stack(sdf_feat_ch[c])
                    stats[f"sdf_feat{c}_mean"] = float(cat.mean())
                    stats[f"sdf_feat{c}_std"] = float(cat.std()) + 1e-8
                else:
                    stats[f"sdf_feat{c}_mean"] = 0.0
                    stats[f"sdf_feat{c}_std"] = 1.0

        for name in self.cond_param_names:
            xs = np.array(param_vals[name], dtype=np.float64)
            if xs.size == 0:
                stats[f"cond_param_{name}_mean"] = 0.0
                stats[f"cond_param_{name}_std"] = 1.0
                stats[f"cond_param_{name}_present_frac"] = 0.0
            else:
                stats[f"cond_param_{name}_mean"] = float(xs.mean())
                stats[f"cond_param_{name}_std"] = float(xs.std()) + 1e-8
                stats[f"cond_param_{name}_present_frac"] = float(xs.size / subset_size)

        stats["cond_dim"] = int(self.cond_dim)
        stats["cond_param_names"] = list(self.cond_param_names)
        stats["cond_device_type_names"] = list(self.device_type_names)
        stats["x_channels"] = int(self.x_channels)
        stats["include_sdf"] = bool(self.include_sdf)
        stats["sdf_thr_eps"] = float(self.sdf_thr_eps)
        stats["sdf_feature"] = str(self.sdf_feature)
        stats["sdf_sigma_nm"] = float(self.sdf_sigma_nm)
        stats["sdf_n_channels"] = int(self.sdf_n_channels)
        stats["sdf_feature_names"] = list(self.sdf_feature_names)

        pretty = {}
        for k, v in stats.items():
            if isinstance(v, (float, np.floating)):
                pretty[k] = round(float(v), 4)
            else:
                pretty[k] = v
        print("Stats:", pretty)
        return stats

    # -----------------------
    # Dataset API
    # -----------------------

    def __len__(self) -> int:
        return len(self.sample_refs)

    def _is_bad_ref(self, ref: SampleRef) -> bool:
        if not isinstance(ref, tuple):
            return False
        shard_path, _, _ = ref
        return shard_path in self._bad_shards

    def _random_ref(self) -> SampleRef:
        n = len(self.sample_refs)
        if n == 0:
            raise RuntimeError("No samples available")
        for _ in range(10):
            j = int(self._retry_rng.integers(0, n))
            ref = self.sample_refs[j]
            if not self._is_bad_ref(ref):
                return ref
        # Fall back even if bad; __getitem__ will handle retries.
        j = int(self._retry_rng.integers(0, n))
        return self.sample_refs[j]

    def _build_sample(self, ref: SampleRef, dtype_np):
        ez_r, ez_i, eps, src, lam_um = self._load_raw(ref, dtype_np=dtype_np)

        # Optional: crop PML out of arrays before downstream processing.
        pml_px = self._pml_px_from_ref(ref)
        ez_r, ez_i, eps, src = self._maybe_crop_pml_arrays(pml_px, ez_r, ez_i, eps, src)
        ez_r, ez_i, eps, src = self._canonicalize_hw(ez_r, ez_i, eps, src)

        # Center-pad to common canvas size (for mixed-domain datasets).
        # Pad BEFORE normalization: fields with 0, eps with cladding value, src with 0.
        if self.canvas_hw is not None:
            th, tw = self.canvas_hw
            h, w = ez_r.shape
            if h != th or w != tw:
                eps_clad = float(eps[0, 0])  # corner pixel is cladding
                ez_r = self._center_pad_2d(ez_r, th, tw, pad_value=0.0)
                ez_i = self._center_pad_2d(ez_i, th, tw, pad_value=0.0)
                eps = self._center_pad_2d(eps, th, tw, pad_value=eps_clad)
                src = self._center_pad_2d(src, th, tw, pad_value=0.0)

        # Phase anchor (mask-first, ROI fallback).
        ez_r2, ez_i2, _ = phase_anchor_mask(ez_r, ez_i, (src > 0.5), eps_r=eps, thr_eps=3.0)
        if (src is None) or (float((src > 0.5).sum()) < 4.0):
            ez_r2, ez_i2, _ = phase_anchor_roi(
                ez_r,
                ez_i,
                eps_r=eps,
                pml_cells=int(self.pml_cells),
                margin=2,
                roi_x=(40, 140),
                thr_eps=3.0,
            )
        ez_r, ez_i = ez_r2, ez_i2

        # Normalize
        ez_r = (ez_r - self.stats["ez_real_mean"]) / self.stats["ez_real_std"]
        ez_i = (ez_i - self.stats["ez_imag_mean"]) / self.stats["ez_imag_std"]

        eps_phys = eps
        if self.normalize_eps:
            eps = (eps_phys - self.stats["eps_mean"]) / self.stats["eps_std"]

        src = (src > 0.5).astype(dtype_np)

        # Conditioning (Option A)
        cond_np = self._build_cond_vector(ref, lam_um=lam_um, dtype_np=dtype_np)

        # Optional SDF channels
        if self.include_sdf:
            phi_nm = self._signed_distance_nm_from_eps(
                eps_phys.astype(np.float32, copy=False),
                ref=ref,
            ).astype(np.float32, copy=False)
            phi_feat = self._sdf_features_from_phi_nm(phi_nm).astype(dtype_np, copy=False)  # [C,H,W]

            if self.normalize_sdf:
                C = int(phi_feat.shape[0])
                for c in range(C):
                    if self.sdf_feature == "raw":
                        mu = float(self.stats.get("sdf_nm_mean", 0.0))
                        sig = float(self.stats.get("sdf_nm_std", 1.0))
                    else:
                        mu = float(self.stats.get(f"sdf_feat{c}_mean", 0.0))
                        sig = float(self.stats.get(f"sdf_feat{c}_std", 1.0))
                    sig = sig if sig > 0 else 1.0
                    phi_feat[c] = (phi_feat[c] - mu) / sig

            sample = np.concatenate([np.stack([ez_r, ez_i, eps, src], axis=0), phi_feat], axis=0)
        else:
            sample = np.stack([ez_r, ez_i, eps, src], axis=0)

        x = torch.from_numpy(sample)
        cond = torch.from_numpy(cond_np).to(dtype=x.dtype)

        # D4 augmentation: sample one of 8 unique transforms
        d4_idx = 0
        if self.augment:
            # For rectangular grids, avoid 90/270 rotations that swap H/W and break batching.
            if x.shape[-2] != x.shape[-1]:
                d4_idx = int(np.random.choice([0, 2, 4, 6]))  # identity, rot180, hflip, vflip
            else:
                d4_idx = np.random.randint(0, 8)
            x = apply_d4_transform_torch(x, d4_idx)

        if not self.return_aux:
            return x, cond

        # -----------------------
        # Aux (collate-safe)
        # -----------------------
        aux_np = self._load_aux(ref, dtype_np=dtype_np)

        # Fill in S-param conditioning if enabled
        if self.include_sparams_cond:
            sparams_np = aux_np.get("sparams_true", None)
            base_dim = 1 + len(self.cond_param_names)  # 4
            if sparams_np is not None:
                s_arr = np.asarray(sparams_np).reshape(-1)
                n_s = min(len(s_arr), self.max_ports)
                for i in range(n_s):
                    cond_np[base_dim + 2 * i] = float(np.real(s_arr[i]))
                    cond_np[base_dim + 2 * i + 1] = float(np.imag(s_arr[i]))
                # port_valid flags (last 4 entries of the 12 appended)
                port_valid_offset = base_dim + 2 * self.max_ports  # base_dim + 8
                for i in range(n_s):
                    cond_np[port_valid_offset + i] = 1.0
                # Update cond tensor
                cond = torch.from_numpy(cond_np).to(dtype=x.dtype)
        port_ids_np = aux_np.get("port_ids", None)
        port_masks_np = aux_np.get("port_masks", None)
        sparams_true_np = aux_np.get("sparams_true", None)
        in_port_idx_np = aux_np.get("in_port_idx", None)

        # Crop port masks consistently if crop_pml=True.
        if port_masks_np is not None:
            (port_masks_np,) = self._maybe_crop_pml_arrays(pml_px, port_masks_np)

        # Center-pad port masks to match canvas (same as field arrays).
        if self.canvas_hw is not None and port_masks_np is not None:
            th, tw = self.canvas_hw
            if port_masks_np.ndim == 3 and (port_masks_np.shape[1] != th or port_masks_np.shape[2] != tw):
                port_masks_np = self._center_pad_3d(port_masks_np, th, tw, pad_value=0.0)

        # Infer P (#ports)
        P = None
        if port_ids_np is not None:
            P = int(np.asarray(port_ids_np).shape[0])
        elif port_masks_np is not None:
            P = int(np.asarray(port_masks_np).shape[0])
        elif sparams_true_np is not None:
            P = int(np.asarray(sparams_true_np).shape[0])

        if P is None:
            P = 0
        P = min(int(P), int(self.max_ports))
        grid_dx_um, grid_dy_um = self._grid_spacing_um_from_ref(ref)

        # Build collate-safe tensors with sentinel defaults.
        aux = {
            "n_ports": torch.tensor(P, dtype=torch.long),
            "in_port_idx": torch.tensor(-1 if in_port_idx_np is None else int(in_port_idx_np), dtype=torch.long),
            "port_valid": torch.zeros((self.max_ports,), dtype=x.dtype),
            "port_ids": torch.full((self.max_ports,), -1, dtype=torch.long),
            "port_masks": torch.zeros((self.max_ports, x.shape[1], x.shape[2]), dtype=x.dtype),
            "sparams_true": torch.zeros((self.max_ports,), dtype=(torch.complex128 if self.use_double else torch.complex64)),
            "grid_dx_um": torch.tensor(float(grid_dx_um), dtype=x.dtype),
            "grid_dy_um": torch.tensor(float(grid_dy_um), dtype=x.dtype),
            "device_type": self._infer_device_type_from_ref(ref),
        }

        if P > 0:
            aux["port_valid"][:P] = 1.0

        if port_ids_np is not None and P > 0:
            pid = np.asarray(port_ids_np).astype(np.int64).reshape(-1)
            pid_pad = np.full((self.max_ports,), -1, dtype=np.int64)
            pid_pad[:P] = pid[:P]
            aux["port_ids"] = torch.from_numpy(pid_pad)

        if port_masks_np is not None and P > 0:
            pm = np.asarray(port_masks_np).astype(dtype_np)  # [P,H,W]
            # If masks are transposed relative to canonical field orientation, fix them.
            if self.target_hw is not None and pm.ndim == 3:
                th, tw = self.target_hw
                if pm.shape[1] == tw and pm.shape[2] == th:
                    pm = np.swapaxes(pm, 1, 2).copy()
            # Apply same D4 transform as x
            if self.augment and d4_idx != 0:
                pm = apply_d4_transform_np(pm, d4_idx)
            # If pm spatial dims do not match x (should not happen after cropping), fall back to zeros.
            if pm.ndim == 3 and pm.shape[1] == x.shape[1] and pm.shape[2] == x.shape[2]:
                pm_pad = np.zeros((self.max_ports, pm.shape[1], pm.shape[2]), dtype=dtype_np)
                pm_pad[:P] = pm[:P]
                aux["port_masks"] = torch.from_numpy(pm_pad).to(x.dtype)

        if sparams_true_np is not None and P > 0:
            s = np.asarray(sparams_true_np).reshape(-1)
            s_pad = np.zeros((self.max_ports,), dtype=(np.complex128 if dtype_np == np.float64 else np.complex64))
            s_pad[:P] = s[:P].astype(s_pad.dtype, copy=False)
            aux["sparams_true"] = torch.from_numpy(s_pad).to(aux["sparams_true"].dtype)

        return x, cond, aux

    def __getitem__(self, idx: int):
        dtype_np = np.float64 if self.use_double else np.float32

        for attempt in range(self.bad_sample_max_retries):
            ref = self.sample_refs[idx] if attempt == 0 else self._random_ref()
            if self._is_bad_ref(ref):
                if not self.skip_bad_samples:
                    shard_path, _, _ = ref  # type: ignore[misc]
                    raise zipfile.BadZipFile(f"Shard marked bad: {shard_path}")
                continue
            try:
                return self._build_sample(ref, dtype_np)
            except (zipfile.BadZipFile, KeyError, OSError, EOFError, ValueError) as e:
                self._bad_sample_count += 1
                if isinstance(ref, tuple):
                    shard_path, _, _ = ref
                    if isinstance(e, zipfile.BadZipFile):
                        self._mark_bad_shard(shard_path, e)
                if not self.skip_bad_samples:
                    raise
                if (self._bad_sample_count == 1) or (self._bad_sample_count % 50 == 0):
                    print(f"[dataset.py] WARNING: skipping bad sample (count={self._bad_sample_count}): {e}")
                continue

        raise RuntimeError(
            f"Exceeded bad sample retries (max={self.bad_sample_max_retries}). "
            "Dataset likely contains too many corrupt shards."
        )

    def get_stats(self) -> Dict[str, float]:
        return self.stats
