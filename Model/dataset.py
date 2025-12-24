import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

ShardRef = Tuple[Path, int, str]  # (shard_path, slot, tag)
SampleRef = Union[Path, ShardRef]

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
    (e.g., Data/ containing coupler_sweep/, y_branch_sweep/, ...), either as
    per-sample folders or packed shard .npz files created by FDTD/pack_dataset.py.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_fraction: float = 0.9,
        stats: Optional[Dict[str, float]] = None,
        normalize_eps: bool = True,
        use_double: bool = False,
        include_sweeps: Optional[Iterable[str]] = None,
        use_shards: bool = False,
        shard_subdir: str = "shards",
        shard_index_name: str = "index.json",
        return_aux: bool = False,
        # --- Optional deterministic subset selection (applied before train/val split) ---
        # Specify desired *sample counts* per sweep for train/val, while selecting whole geometries
        # (so coupler inPort1/inPort2 stay together).
        subset_train_per_sweep: Optional[Dict[str, int]] = None,
        subset_val_per_sweep: Optional[Dict[str, int]] = None,
        subset_seed: int = 0,
    ):
        """
        root_dir: path to Data (parent containing sweep subfolders).
        include_sweeps: optional iterable of subfolder names to include (e.g., ["coupler_sweep", "y_branch_sweep"]);
                        if None, include all immediate subdirectories.
        use_shards: set True to read shard .npz files (default False uses per-sample folders).
        shard_subdir: name of the shard directory inside each sweep (default: 'shards').
        shard_index_name: filename of the shard index (default: 'index.json').
        return_aux: if True, __getitem__ returns (x, cond, aux) where aux includes
                    any available sparams / port mask supervision from the dataset.
        """
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.train_fraction = train_fraction
        self.normalize_eps = normalize_eps
        self.use_double = use_double
        self.use_shards = use_shards
        self.shard_subdir = shard_subdir
        self.shard_index_name = shard_index_name
        self.return_aux = bool(return_aux)
        self._shard_cache: Dict[Path, np.lib.npyio.NpzFile] = {}
        self.subset_train_per_sweep = subset_train_per_sweep
        self.subset_val_per_sweep = subset_val_per_sweep
        self.subset_seed = int(subset_seed)

        # ------------------------------------------------------------------
        # Conditioning (Option A): [lambda_norm] + [params_norm...] + [masks...] + [device_type_onehot...]
        # ------------------------------------------------------------------
        # Note: we intentionally keep wavelength separate as the first cond entry
        # because parts of the training code derive physical lambda_um from cond[:,0].
        self.cond_param_names: List[str] = [
            # shared-ish
            "wg_width_um",
            # coupler-specific
            "gap_um",
            "Lc_um",
            "bend_length_um",
            "lead_extra_gap_um",
            # y-branch-specific
            "l_junction_um",
            "l_bend_um",
            "h_bend_um",
            "l_out_um",
        ]
        # device types present in existing shard writers
        self.device_type_names: List[str] = ["coupler", "y_branch"]
        self.cond_dim: int = 1 + 2 * len(self.cond_param_names) + len(self.device_type_names)

        # Max number of ports across device families. Used to pad aux tensors so
        # mixed-device batches collate cleanly (e.g., coupler has 4 ports, y-branch has 3).
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

        if self.use_shards:
            all_refs: List[SampleRef] = self._collect_shard_refs(sweep_dirs)
        else:
            all_refs = self._collect_folder_refs(sweep_dirs)

        # Optional deterministic subset selection (applied before split).
        if (self.subset_train_per_sweep is not None) or (self.subset_val_per_sweep is not None):
            self.sample_refs = self._subset_refs_by_sweep(
                all_refs,
                split=split,
                train_per_sweep=self.subset_train_per_sweep or {},
                val_per_sweep=self.subset_val_per_sweep or {},
                seed=self.subset_seed,
            )
        else:
            n_total = len(all_refs)
            n_train = int(round(train_fraction * n_total))
            if split == "train":
                self.sample_refs = all_refs[:n_train]
            elif split == "val":
                self.sample_refs = all_refs[n_train:]
            else:
                raise ValueError("split must be 'train' or 'val'")

        if len(self.sample_refs) == 0:
            raise RuntimeError(f"No samples in {split} split (train_fraction={train_fraction})")

        if stats is None:
            if split != "train":
                raise ValueError("stats must be provided for non-train splits")
            print(f"[{split}] computing normalization stats from subset...")
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        # allow stats (from older checkpoints) to override if present
        try:
            self.cond_dim = int(self.stats.get("cond_dim", self.cond_dim))
        except Exception:
            pass

        print(f"[{split}] dataset size = {len(self.sample_refs)}")

    def _infer_sweep_name_from_ref(self, ref: SampleRef) -> str:
        """
        Return the sweep folder name (e.g., 'coupler_sweep', 'y_branch_sweep') for a sample ref.
        """
        if isinstance(ref, tuple):
            shard_path, _, _ = ref
            # .../Data/<sweep>/shards/<shard>.npz
            try:
                return shard_path.parent.parent.name
            except Exception:
                return ""
        try:
            # folder samples live under .../Data/<sweep>/<tag>/
            return Path(ref).parent.name
        except Exception:
            return ""

    def _tag_from_ref(self, ref: SampleRef) -> str:
        if isinstance(ref, tuple):
            _, _, tag = ref
            return str(tag)
        return Path(ref).name

    def _geom_key_from_tag(self, tag: str) -> str:
        """
        Geometry identifier used to keep related samples together.
        - coupler_sweep: '<base>__inPort1'/'__inPort2' -> '<base>'
        - y_branch_sweep: '<base>__inPort1' -> '<base>' (only one anyway)
        Fallback: full tag.
        """
        t = str(tag)
        if "__inPort" in t:
            return t.split("__inPort", 1)[0]
        return t

    def _subset_refs_by_sweep(
        self,
        all_refs: List[SampleRef],
        *,
        split: str,
        train_per_sweep: Dict[str, int],
        val_per_sweep: Dict[str, int],
        seed: int,
    ) -> List[SampleRef]:
        """
        Deterministically pick a subset per sweep with *sample count targets* for train/val,
        while selecting whole geometries (so inPort pairs stay together).
        """
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # Group refs by sweep -> geom_key -> [refs...]
        by_sweep: Dict[str, Dict[str, List[SampleRef]]] = {}
        for r in all_refs:
            sweep = self._infer_sweep_name_from_ref(r)
            tag = self._tag_from_ref(r)
            gk = self._geom_key_from_tag(tag)
            by_sweep.setdefault(sweep, {}).setdefault(gk, []).append(r)

        # If user didn't specify a sweep in train/val dicts, exclude it (explicit subset behavior).
        requested_sweeps = set(train_per_sweep.keys()) | set(val_per_sweep.keys())
        if not requested_sweeps:
            raise ValueError("subset_train_per_sweep/subset_val_per_sweep provided but empty; specify at least one sweep.")

        out_refs: List[SampleRef] = []

        for sweep in sorted(requested_sweeps):
            geom_map = by_sweep.get(sweep, {})
            if not geom_map:
                raise RuntimeError(f"Requested subset sweep '{sweep}' not found under {self.root}. Available: {sorted(by_sweep.keys())}")

            rng = np.random.default_rng(int(seed) + (abs(hash(sweep)) % 1_000_000))
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

        # deterministic ordering for reproducibility
        out_refs.sort(key=lambda r: self._tag_from_ref(r))
        return out_refs

    def _collect_folder_refs(self, sweep_dirs: List[Path]) -> List[Path]:
        all_dirs: List[Path] = []
        for sdir in sweep_dirs:
            all_dirs += [d for d in sorted(sdir.iterdir()) if d.is_dir()]
        if not all_dirs:
            raise RuntimeError(f"No sample subfolders found in sweeps {sweep_dirs}")
        return all_dirs

    def _collect_shard_refs(self, sweep_dirs: List[Path]) -> List[ShardRef]:
        refs: List[ShardRef] = []
        for sdir in sweep_dirs:
            shard_dir = sdir / self.shard_subdir
            index_path = shard_dir / self.shard_index_name
            if not index_path.is_file():
                raise RuntimeError(f"Missing shard index {index_path}")
            with open(index_path, "r") as f:
                entries = json.load(f)
            for e in entries:
                shard_path = shard_dir / e["shard"]
                slot = int(e["slot"])
                tag = e.get("tag", "")
                refs.append((shard_path, slot, tag))
        if not refs:
            raise RuntimeError(f"No shard entries found under sweeps {sweep_dirs}")
        # deterministic ordering
        refs.sort(key=lambda r: (r[2], r[0].name, r[1]))
        return refs

    def _get_shard_file(self, shard_path: Path) -> np.lib.npyio.NpzFile:
        if shard_path not in self._shard_cache:
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            self._shard_cache[shard_path] = np.load(shard_path, allow_pickle=False)
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

        lam_arr = get_arr("sparams/wavelength_um")
        lam_um = float(lam_arr)
        return ez_r, ez_i, eps, src, lam_um

    def _infer_device_type_from_ref(self, ref: SampleRef) -> str:
        """
        Best-effort device type inference.
        - shard: prefer stored scalar 'dataset' key
        - folder: infer from parent directory name (e.g., Data/coupler_sweep/...).
        """
        if isinstance(ref, tuple):
            shard_path, slot, _ = ref
            data = self._get_shard_file(shard_path)
            key = f"s{slot}/dataset"
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
        # folder-based
        try:
            parent = Path(ref).parent.name.lower()
        except Exception:
            return ""
        if "coupler" in parent:
            return "coupler"
        if "y_branch" in parent or "ybranch" in parent:
            return "y_branch"
        return ""

    def _get_param_value_from_ref(self, ref: SampleRef, name: str) -> Optional[float]:
        """
        Load a geometry scalar parameter by name from either shard or folder sample.
        Returns None if missing.
        """
        key = f"sparams/{name}"
        if isinstance(ref, tuple):
            shard_path, slot, _ = ref
            data = self._get_shard_file(shard_path)
            full_key = f"s{slot}/" + key
            if full_key not in data:
                return None
            try:
                return float(np.array(data[full_key]).item())
            except Exception:
                try:
                    return float(data[full_key])
                except Exception:
                    return None
        # folder-based: try sparams.npz
        d = Path(ref)
        sp_path = d / "sparams.npz"
        if not sp_path.is_file():
            return None
        try:
            sp = np.load(sp_path)
        except Exception:
            return None
        if name not in sp:
            return None
        try:
            return float(np.array(sp[name]).item())
        except Exception:
            try:
                return float(sp[name])
            except Exception:
                return None

    def _build_cond_vector(self, ref: SampleRef, lam_um: float, dtype_np) -> np.ndarray:
        """
        Option A conditioning vector:
          [lambda_norm] + [param_norm...] + [param_mask...] + [device_type_onehot...]
        """
        # wavelength (always expected)
        lam_norm = (lam_um - self.stats["lambda_um_mean"]) / self.stats["lambda_um_std"]

        # params + masks
        p_vals = []
        p_masks = []
        for name in self.cond_param_names:
            v = self._get_param_value_from_ref(ref, name)
            if v is None or not np.isfinite(v):
                p_vals.append(0.0)
                p_masks.append(0.0)
                continue
            mean = float(self.stats.get(f"cond_param_{name}_mean", 0.0))
            std = float(self.stats.get(f"cond_param_{name}_std", 1.0))
            std = std if std > 0 else 1.0
            p_vals.append((float(v) - mean) / std)
            p_masks.append(1.0)

        # device type one-hot
        dtype = self._infer_device_type_from_ref(ref)
        onehot = [0.0] * len(self.device_type_names)
        for i, tname in enumerate(self.device_type_names):
            if dtype == tname:
                onehot[i] = 1.0

        cond = np.array([lam_norm] + p_vals + p_masks + onehot, dtype=dtype_np)
        return cond

    def _load_aux_from_shard(self, ref: ShardRef, dtype_np):
        """
        Optional aux loader for shard datasets.

        Returns dict with keys (when present):
          - sparams_true: np.complex64/128 array [P]
          - in_port_idx: int (0-based index into ports)
          - port_masks: np.float32/64 array [P, H, W]
          - port_ids: np.int32 array [P]
        Missing keys are returned as None.
        """
        shard_path, slot, _ = ref
        data = self._get_shard_file(shard_path)
        prefix = f"s{slot}/"

        def get_arr(key: str):
            full_key = prefix + key
            if full_key not in data:
                return None
            return data[full_key]

        # S-params (stored as real/imag vectors)
        Sr = get_arr("sparams/S_real")
        Si = get_arr("sparams/S_imag")
        if Sr is not None and Si is not None:
            Sr = Sr.astype(dtype_np)
            Si = Si.astype(dtype_np)
            sparams_true = (Sr + 1j * Si).astype(np.complex128 if dtype_np == np.float64 else np.complex64)
        else:
            sparams_true = None

        # Ports (optional; present for coupler shards, not necessarily for others)
        port_masks = get_arr("ports/masks")
        if port_masks is not None:
            port_masks = port_masks.astype(dtype_np)
        port_ids = get_arr("ports/ids")
        if port_ids is not None:
            port_ids = port_ids.astype(np.int32)

        # Input port index (0-based). Prefer explicit input_port scalar if present.
        in_port_idx = None
        in_port = get_arr("sparams/input_port")
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
                    # Common convention: ports are 1..P in order => idx = port-1
                    in_port_idx = int(in_port_val - 1)

        return {
            "sparams_true": sparams_true,
            "in_port_idx": in_port_idx,
            "port_masks": port_masks,
            "port_ids": port_ids,
        }

    def _load_raw_from_dir(self, d: Path, dtype_np):
        ez_r = np.load(d / "Ez_real.npy").astype(dtype_np)
        ez_i = np.load(d / "Ez_imag.npy").astype(dtype_np)
        eps = np.load(d / "eps.npy").astype(dtype_np)
        src = np.load(d / "src_mask.npy").astype(dtype_np)
        sp = np.load(d / "sparams.npz")
        lam_um = float(sp["wavelength_um"])
        return ez_r, ez_i, eps, src, lam_um

    def _load_aux_from_dir(self, d: Path, dtype_np):
        """
        Optional aux loader for per-sample folder datasets.

        Returns dict with keys (when present):
          - sparams_true: np.complex64/128 array [P]
          - in_port_idx: int (0-based)
          - port_masks: np array [P,H,W] (only if present on disk)
          - port_ids: np.int32 array [P] (only if present on disk)
        """
        aux = {"sparams_true": None, "in_port_idx": None, "port_masks": None, "port_ids": None}

        sp_path = d / "sparams.npz"
        if sp_path.is_file():
            sp = np.load(sp_path)
            if ("S_real" in sp) and ("S_imag" in sp):
                Sr = sp["S_real"].astype(dtype_np)
                Si = sp["S_imag"].astype(dtype_np)
                aux["sparams_true"] = (Sr + 1j * Si).astype(np.complex128 if dtype_np == np.float64 else np.complex64)
            if "input_port" in sp:
                try:
                    aux["in_port_idx"] = int(sp["input_port"]) - 1
                except Exception:
                    aux["in_port_idx"] = None

        # Optional: if you ever save these alongside folders (not currently in older sweeps)
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

    def _compute_stats(self) -> Dict[str, float]:
        subset_size = min(len(self.sample_refs), 500)
        idxs = np.random.choice(len(self.sample_refs), subset_size, replace=False)

        ez_r_all = []
        ez_i_all = []
        eps_all = []
        lam_all = []
        # conditioning param stats (presence-aware)
        param_vals: Dict[str, List[float]] = {k: [] for k in self.cond_param_names}

        for i in idxs:
            ref = self.sample_refs[i]
            ez_r, ez_i, eps, _, lam = self._load_raw(ref, dtype_np=np.float32)
            ez_r, ez_i, _ = phase_anchor_roi(ez_r, ez_i, eps_r=eps, pml_cells=30, margin=2)

            ez_r_all.append(ez_r)
            ez_i_all.append(ez_i)
            eps_all.append(eps)
            lam_all.append(lam)

            # collect geometry params if present (exclude NaNs/missing)
            for name in self.cond_param_names:
                v = self._get_param_value_from_ref(ref, name)
                if v is None:
                    continue
                if not np.isfinite(v):
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

        # param normalization stats (mean/std over present values)
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

        # record conditioning layout for downstream consumers
        stats["cond_dim"] = int(self.cond_dim)
        stats["cond_param_names"] = list(self.cond_param_names)
        stats["cond_device_type_names"] = list(self.device_type_names)

        # Pretty-print: round floats, keep other metadata as-is
        pretty = {}
        for k, v in stats.items():
            if isinstance(v, (float, np.floating)):
                pretty[k] = round(float(v), 4)
            else:
                pretty[k] = v
        print("Stats:", pretty)
        return stats

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, idx: int):
        dtype_np = np.float64 if self.use_double else np.float32
        ref = self.sample_refs[idx]
        ez_r, ez_i, eps, src, lam_um = self._load_raw(ref, dtype_np=dtype_np)


        ### Phase anchor ###
        ez_r, ez_i, phi = phase_anchor_roi(
            ez_r, ez_i,
            eps_r=eps,
            pml_cells=30,
            margin=2,
            roi_x=(40, 140),
            thr_eps=3.0,
        )

        ### Normalize ###
        ez_r = (ez_r - self.stats["ez_real_mean"]) / self.stats["ez_real_std"]
        ez_i = (ez_i - self.stats["ez_imag_mean"]) / self.stats["ez_imag_std"]
        if self.normalize_eps:
            eps = (eps - self.stats["eps_mean"]) / self.stats["eps_std"]
        
        src = (src > 0.5).astype(dtype_np)

        # conditioning (Option A)
        cond_np = self._build_cond_vector(ref, lam_um=lam_um, dtype_np=dtype_np)

        sample = np.stack([ez_r, ez_i, eps, src], axis=0)
        x = torch.from_numpy(sample)
        cond = torch.from_numpy(cond_np).to(dtype=x.dtype)

        if not self.return_aux:
            return x, cond

        aux_np = self._load_aux(ref, dtype_np=dtype_np)
        aux = {
            "in_port_idx": aux_np.get("in_port_idx", None),
            "port_ids": None,
            "port_masks": None,
            "sparams_true": None,
            "n_ports": None,
            "port_valid": None,
        }

        # ---- Pad aux port-related tensors to self.max_ports so DataLoader can stack mixed devices ----
        port_ids_np = aux_np.get("port_ids", None)
        port_masks_np = aux_np.get("port_masks", None)
        sparams_true_np = aux_np.get("sparams_true", None)

        # infer P (number of valid ports in this sample)
        P = None
        if port_ids_np is not None:
            P = int(np.asarray(port_ids_np).shape[0])
        elif port_masks_np is not None:
            P = int(np.asarray(port_masks_np).shape[0])
        elif sparams_true_np is not None:
            P = int(np.asarray(sparams_true_np).shape[0])

        if P is not None:
            P = min(P, int(self.max_ports))
            aux["n_ports"] = int(P)
            pv = np.zeros((self.max_ports,), dtype=np.float32)
            pv[:P] = 1.0
            aux["port_valid"] = torch.from_numpy(pv).to(x.dtype)

        if port_ids_np is not None and P is not None:
            pid = np.asarray(port_ids_np).astype(np.int64).reshape(-1)
            pid_pad = np.full((self.max_ports,), -1, dtype=np.int64)
            pid_pad[:P] = pid[:P]
            aux["port_ids"] = torch.from_numpy(pid_pad)

        if port_masks_np is not None and P is not None:
            pm = np.asarray(port_masks_np).astype(dtype_np)  # [P,H,W]
            pm_pad = np.zeros((self.max_ports, pm.shape[1], pm.shape[2]), dtype=dtype_np)
            pm_pad[:P] = pm[:P]
            aux["port_masks"] = torch.from_numpy(pm_pad).to(x.dtype)

        if sparams_true_np is not None and P is not None:
            s = np.asarray(sparams_true_np).reshape(-1)  # [P]
            # pad complex vector to max_ports; keep padded entries at 0
            s_pad = np.zeros((self.max_ports,), dtype=np.complex128 if dtype_np == np.float64 else np.complex64)
            s_pad[:P] = s[:P].astype(s_pad.dtype, copy=False)
            s_t = torch.from_numpy(s_pad)
            aux["sparams_true"] = s_t.to(torch.complex128 if self.use_double else torch.complex64)

        return x, cond, aux

    def get_stats(self) -> Dict[str, float]:
        return self.stats
