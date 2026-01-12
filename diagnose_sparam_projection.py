#!/usr/bin/env python3
"""
diagnose_sparam_projection.py

Diagnostic: compare the stored "true" S-parameters (from Meep S_dict, stored in shards)
against the S-parameters you would get by *projecting the ground-truth field* Ez onto the
stored port masks and forming S_p = a_p / a_in (the same structure used in Model/sparams_loss.py).

If these disagree significantly, then the training-time sparam loss has an irreducible floor:
you're optimizing a proxy measurement (stripe projection) against a different target (mode S-params).

Usage:
  python diagnose_sparam_projection.py \
    --data-root /dartfs-hpc/rc/home/j/f0071mj/rayfield/Data \
    --include-sweeps coupler_sweep,y_branch_sweep \
    --num-samples 200 \
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_index(index_path: Path) -> List[dict]:
    with open(index_path, "r") as f:
        return json.load(f)


def _safe_item(x):
    try:
        return np.array(x).item()
    except Exception:
        return x


def _get(data: np.lib.npyio.NpzFile, key: str):
    return data[key] if key in data else None


def _resolve_in_idx(input_port: Optional[int], port_ids: Optional[np.ndarray]) -> int:
    if input_port is None:
        return 0
    try:
        ip = int(input_port)
    except Exception:
        return 0
    if port_ids is not None:
        try:
            hits = np.where(np.asarray(port_ids).reshape(-1) == ip)[0]
            if hits.size > 0:
                return int(hits[0])
        except Exception:
            pass
    # fall back: assume ports are [1..P] in order
    return max(0, ip - 1)


def extract_sparams_np(
    E: np.ndarray,              # complex [H,W]
    port_masks: np.ndarray,     # float [P,H,W]
    input_port: Optional[int],
    port_ids: Optional[np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    P = int(port_masks.shape[0])
    amps = np.zeros((P,), dtype=np.complex128)
    for p in range(P):
        m = port_masks[p].astype(np.float64)
        den = float(np.sum(m))
        if den < 1.0:
            amps[p] = 0.0 + 0.0j
        else:
            amps[p] = np.sum(m * E) / den

    in_idx = _resolve_in_idx(input_port, port_ids)
    in_idx = int(np.clip(in_idx, 0, P - 1))
    a_in = amps[in_idx]
    mag = np.abs(a_in)
    if mag < eps:
        a_in = 1.0 + 0.0j
    else:
        a_in = a_in / mag * max(mag, eps)
    return amps / a_in


@dataclass
class SampleMetrics:
    sweep: str
    tag: str
    P: int
    mse_complex: float
    mse_mag: float
    phase_cos_mean: float


def _phase_cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = np.abs(a).clip(min=eps)
    bb = np.abs(b).clip(min=eps)
    ua = a / aa
    ub = b / bb
    return float(np.mean(np.real(ua * np.conj(ub))))


def analyze_one_sample(
    sweep: str,
    shard_path: Path,
    slot: int,
    tag: str,
) -> Optional[SampleMetrics]:
    with np.load(shard_path, allow_pickle=False) as z:
        prefix = f"s{int(slot)}/"

        Ezr = _get(z, prefix + "Ez_real")
        Ezi = _get(z, prefix + "Ez_imag")
        pm = _get(z, prefix + "ports/masks")
        pid = _get(z, prefix + "ports/ids")
        Sr = _get(z, prefix + "sparams/S_real")
        Si = _get(z, prefix + "sparams/S_imag")
        ip = _get(z, prefix + "sparams/input_port")

        if Ezr is None or Ezi is None or pm is None or Sr is None or Si is None:
            return None

        E = Ezr.astype(np.float64) + 1j * Ezi.astype(np.float64)  # [H,W]
        port_masks = pm.astype(np.float64)                         # [P,H,W]
        port_ids = pid.astype(np.int32) if pid is not None else None
        input_port = int(_safe_item(ip)) if ip is not None else None

        S_true = Sr.astype(np.float64) + 1j * Si.astype(np.float64)  # [P_true]
        P_true = int(S_true.shape[0])
        P_mask = int(port_masks.shape[0])
        P = int(min(P_true, P_mask))
        if P <= 0:
            return None

        S_proj = extract_sparams_np(
            E=E,
            port_masks=port_masks[:P],
            input_port=input_port,
            port_ids=port_ids[:P] if port_ids is not None else None,
        )
        S_t = S_true[:P]

        mse_c = float(np.mean(np.abs(S_proj - S_t) ** 2))
        mse_m = float(np.mean((np.abs(S_proj) - np.abs(S_t)) ** 2))
        ph_cos = _phase_cos(S_proj, S_t)

        return SampleMetrics(
            sweep=sweep,
            tag=tag,
            P=P,
            mse_complex=mse_c,
            mse_mag=mse_m,
            phase_cos_mean=ph_cos,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--include-sweeps", type=str, default="")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    rng = np.random.default_rng(int(args.seed))

    if args.include_sweeps.strip():
        sweeps = [s.strip() for s in args.include_sweeps.split(",") if s.strip()]
    else:
        sweeps = [p.name for p in root.iterdir() if p.is_dir()]

    entries: List[Tuple[str, Path, int, str]] = []
    for sweep in sweeps:
        idx_path = root / sweep / "shards" / "index.json"
        if not idx_path.is_file():
            continue
        idx = _load_index(idx_path)
        for row in idx:
            shard = row["shard"]
            slot = int(row["slot"])
            tag = str(row.get("tag", ""))
            shard_path = root / sweep / "shards" / shard
            entries.append((sweep, shard_path, slot, tag))

    if not entries:
        raise SystemExit(f"No shard entries found under {root} for sweeps={sweeps}")

    n = min(int(args.num_samples), len(entries))
    sel = rng.choice(len(entries), size=n, replace=False)

    metrics: List[SampleMetrics] = []
    skipped = 0
    for i in sel:
        sweep, shard_path, slot, tag = entries[int(i)]
        m = analyze_one_sample(sweep, shard_path, slot, tag)
        if m is None:
            skipped += 1
            continue
        metrics.append(m)

    if not metrics:
        raise SystemExit("All selected samples were missing Ez/ports/sparams keys; nothing to analyze.")

    # Aggregate
    mse_c = np.array([m.mse_complex for m in metrics], dtype=np.float64)
    mse_m = np.array([m.mse_mag for m in metrics], dtype=np.float64)
    ph_c = np.array([m.phase_cos_mean for m in metrics], dtype=np.float64)
    by_sweep: Dict[str, List[int]] = {}
    for j, m in enumerate(metrics):
        by_sweep.setdefault(m.sweep, []).append(j)

    def _summ(name, x):
        return (
            f"{name}: mean={x.mean():.3e}  median={np.median(x):.3e}  "
            f"p90={np.quantile(x, 0.9):.3e}  p99={np.quantile(x, 0.99):.3e}"
        )

    print("---- Sparam projection diagnostic ----")
    print(f"data_root={root}")
    print(f"sweeps={sweeps}")
    print(f"requested={args.num_samples}  analyzed={len(metrics)}  skipped_missing={skipped}")
    print("")
    print(_summ("MSE(|Sproj - Strue|^2)", mse_c))
    print(_summ("MSE(|Sproj|-|Strue|)^2", mse_m))
    print(_summ("phase_cos_mean", ph_c))
    print("")
    for sweep, idxs in sorted(by_sweep.items()):
        x1 = mse_c[idxs]
        x2 = ph_c[idxs]
        print(f"[{sweep}] n={len(idxs)}  mse_c_mean={x1.mean():.3e}  phase_cos_mean={x2.mean():.3f}")

    # Show worst offenders
    worst = np.argsort(-mse_c)[:10]
    print("\nWorst 10 by complex MSE:")
    for k in worst:
        m = metrics[int(k)]
        print(f"  mse={m.mse_complex:.3e}  phase_cos={m.phase_cos_mean:+.3f}  P={m.P}  {m.sweep}  {m.tag}")


if __name__ == "__main__":
    main()


