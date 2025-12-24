from __future__ import annotations

import os
import json
import uuid
import math
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from directional_coupler import DirectionalCoupler2D
from conditioning_masks import make_source_mask


# -----------------------------
# Config defaults
# -----------------------------
RESOLUTION_DEFAULT = 32
N_GEO_DEFAULT = 7500
N_PROCS_DEFAULT = 24

# Geometry and wavelength ranges (um)
gap_min, gap_max = 0.10, 0.35
wg_length_min, wg_length_max = 5.0, 15.0
bend_length_min, bend_length_max = 5.0, 7.0
wg_width_min, wg_width_max = 0.38, 0.60
lambda_min, lambda_max = 1.40, 1.60
lead_gap_min, lead_gap_max = 1.0, 3.0

# Stratify gap (oversample small gap)
GAP_STRATA = [
    (0.10, 0.16, 0.45),
    (0.16, 0.22, 0.30),
    (0.22, 0.35, 0.25),
]


# -----------------------------
# Naming utilities (collision-resistant)
# -----------------------------
def _as_code_um(x_um: float, scale: int = 1000) -> int:
    return int(np.round(float(x_um) * scale))


def geom_tag(
    wg_width: float,
    gap: float,
    wg_length: float,
    bend_length: float,
    lead_extra_gap: float,
    lam: float,
) -> str:
    # integer-coded fields for uniqueness + readable floats for sanity
    w_i = _as_code_um(wg_width, 1000)
    g_i = _as_code_um(gap, 1000)
    L_i = _as_code_um(wg_length, 1000)
    b_i = _as_code_um(bend_length, 1000)
    lead_i = _as_code_um(lead_extra_gap, 1000)
    lam_i = _as_code_um(lam, 1000)

    return (
        f"w{w_i:04d}nm_gap{g_i:04d}nm_Lc{L_i:05d}nm_b{b_i:05d}nm_lead{lead_i:05d}nm_lam{lam_i:04d}nm"
        f"__w{wg_width:.3f}_g{gap:.3f}_Lc{wg_length:.2f}_b{bend_length:.2f}_lead{lead_extra_gap:.2f}_lam{lam:.2f}"
    )


def flip_y(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=0).copy()


def synthesize_s_for_port2_from_port1(S_port1: np.ndarray) -> np.ndarray:
    # S_port1 = [S11, S21, S31, S41] for input=1
    # By y-mirror symmetry:
    # ports map: 1<->2 and 3<->4
    # => [S(1,2),S(2,2),S(3,2),S(4,2)] = [S(2,1),S(1,1),S(4,1),S(3,1)]
    return np.array([S_port1[1], S_port1[0], S_port1[3], S_port1[2]], dtype=np.complex128)


# -----------------------------
# Sampling: stratified LHS over gap
# -----------------------------
def _allocate_counts(total: int, fracs: list[float]) -> list[int]:
    counts = [int(round(total * f)) for f in fracs]
    drift = total - sum(counts)
    if drift != 0:
        step = 1 if drift > 0 else -1
        for i in range(abs(drift)):
            counts[i % len(counts)] += step
    return counts


def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


def build_param_list(N_GEO: int, seed_base: int = 42) -> list[tuple[float, float, float, float, float, float]]:
    strata_counts = _allocate_counts(N_GEO, [f for (_, _, f) in GAP_STRATA])

    u_list = []
    for si, (gap_lo, gap_hi, _frac) in enumerate(GAP_STRATA):
        n_i = int(strata_counts[si])
        if n_i <= 0:
            continue
        sampler = qmc.LatinHypercube(d=6, seed=seed_base + si)
        u_i = sampler.random(n_i)

        gaps_i = gap_lo + u_i[:, 0] * (gap_hi - gap_lo)
        wg_lengths_i = wg_length_min + u_i[:, 1] * (wg_length_max - wg_length_min)
        bend_lengths_i = bend_length_min + u_i[:, 2] * (bend_length_max - bend_length_min)
        wg_widths_i = wg_width_min + u_i[:, 3] * (wg_width_max - wg_width_min)
        wavelengths_i = lambda_min + u_i[:, 4] * (lambda_max - lambda_min)
        lead_gaps_i = lead_gap_min + u_i[:, 5] * (lead_gap_max - lead_gap_min)

        u_list.append((gaps_i, wg_lengths_i, bend_lengths_i, wg_widths_i, wavelengths_i, lead_gaps_i))

    gaps = np.concatenate([t[0] for t in u_list], axis=0)
    wg_lengths = np.concatenate([t[1] for t in u_list], axis=0)
    bend_lengths = np.concatenate([t[2] for t in u_list], axis=0)
    wg_widths = np.concatenate([t[3] for t in u_list], axis=0)
    wavelengths = np.concatenate([t[4] for t in u_list], axis=0)
    lead_gaps = np.concatenate([t[5] for t in u_list], axis=0)

    assert len(gaps) == N_GEO, f"expected N_GEO={N_GEO}, got {len(gaps)}"

    # quantize only width and wavelength
    wg_widths = _quantize_01(wg_widths, wg_width_min, wg_width_max)
    wavelengths = _quantize_01(wavelengths, lambda_min, lambda_max)

    return [
        (float(w), float(g), float(L), float(b), float(lead), float(lam))
        for w, g, L, b, lead, lam in zip(wg_widths, gaps, wg_lengths, bend_lengths, lead_gaps, wavelengths)
    ]


# -----------------------------
# Worker: run port1 sim, write ONE temp npz, return path
# -----------------------------
def run_fdtd_sim_port1(
    wg_width: float,
    gap: float,
    wg_length: float,
    bend_length: float,
    lead_extra_gap: float,
    wl: float,
    RESOLUTION: int,
):
    dc = DirectionalCoupler2D(
        wg_width_um=wg_width,
        gap_um=gap,
        wg_length_um=wg_length,
        wavelength_um=wl,
        resolution=RESOLUTION,
        dpml=1,
        pad_y_um=1.0,
        lead_extra_gap_um=lead_extra_gap,
        bend_length_um=bend_length,
        bend_n_segments=64,
    )

    # convention: eps, Ez, Hx, Hy, S_dict, cell
    eps, Ez, Hx, Hy, S_dict, cell = dc.run_sim(input_port=1, decay_tol=1e-5)
    return dc, S_dict, Ez, eps, cell


def worker(task):
    """
    Returns:
      ("OK", temp_npz_path_str) or ("ERR", err_string)
    """
    (wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, RESOLUTION, tmp_dir_str) = task
    tmp_dir = Path(tmp_dir_str)

    base = geom_tag(wg_width, gap, wg_length, bend_length, lead_extra_gap, lam)
    tmp_name = f"geom_{base}__{uuid.uuid4().hex}.npz"
    tmp_path = tmp_dir / tmp_name

    try:
        dc, S_dict, Ez1, eps, cell = run_fdtd_sim_port1(
            wg_width, gap, wg_length, bend_length, lead_extra_gap, lam, RESOLUTION
        )
        if S_dict is None:
            raise RuntimeError("S_dict is None")
        if Ez1 is None or eps is None or cell is None:
            raise RuntimeError("Missing Ez/eps/cell")

        eps = np.asarray(eps, dtype=np.float32)
        Ez1 = np.asarray(Ez1, dtype=np.complex64)

        ny, nx = eps.shape
        Lx_um, Ly_um = cell

        # Port masks for ports 1..4 (for S-parameter projection / auxiliary losses)
        port_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        port_centers = dc.get_port_centers_um()
        y_span = dc.get_port_y_span_um()
        port_masks = np.stack(
            [
                make_source_mask(
                    input_port=p,
                    port_centers_um=port_centers,
                    y_span_um=y_span,
                    Lx_um=Lx_um,
                    Ly_um=Ly_um,
                    ny=ny,
                    nx=nx,
                ).astype(np.float32)
                for p in port_ids.tolist()
            ],
            axis=0,
        )  # [P, ny, nx]

        # Source mask for input_port=1
        src_mask1 = make_source_mask(
            input_port=1,
            port_centers_um=port_centers,
            y_span_um=y_span,
            Lx_um=Lx_um,
            Ly_um=Ly_um,
            ny=ny,
            nx=nx,
        ).astype(np.float32)

        # S vector for input=1
        S1 = np.array([S_dict[(1, 1)], S_dict[(2, 1)], S_dict[(3, 1)], S_dict[(4, 1)]], dtype=np.complex128)

        # grid meta
        dx = 1.0 / RESOLUTION
        dy = 1.0 / RESOLUTION

        np.savez_compressed(
            tmp_path,
            # geometry scalars
            wg_width_um=np.float32(wg_width),
            gap_um=np.float32(gap),
            Lc_um=np.float32(wg_length),
            bend_length_um=np.float32(bend_length),
            lead_extra_gap_um=np.float32(lead_extra_gap),
            wavelength_um=np.float32(lam),
            resolution=np.int32(RESOLUTION),
            # grid scalars
            nx=np.int32(nx),
            ny=np.int32(ny),
            dx=np.float32(dx),
            dy=np.float32(dy),
            Lx_um=np.float32(Lx_um),
            Ly_um=np.float32(Ly_um),
            # arrays
            eps=eps,
            Ez_real=Ez1.real.astype(np.float32),
            Ez_imag=Ez1.imag.astype(np.float32),
            src_mask=src_mask1,
            port_ids=port_ids,
            port_masks=port_masks,
            S1_real=S1.real.astype(np.float32),
            S1_imag=S1.imag.astype(np.float32),
            base_tag=np.array(base),  # scalar string array
        )

        return ("OK", str(tmp_path))

    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return ("ERR", f"{base} :: {repr(e)}")


# -----------------------------
# Shard writer process (streaming)
# -----------------------------
def _write_shard_npz(out_path: Path, samples: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]], compress: bool):
    save_dict: Dict[str, np.ndarray] = {}
    for i, (arrays, meta) in enumerate(samples):
        prefix = f"s{i}/"
        for k, v in arrays.items():
            save_dict[prefix + k] = v
        for k, v in meta.items():
            save_dict[prefix + k] = np.array(v)

    if compress:
        np.savez_compressed(out_path, **save_dict)
    else:
        np.savez(out_path, **save_dict)


def _atomic_write_index(index_path: Path, index_list: List[Dict[str, object]]):
    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(index_list, f, indent=2)
    os.replace(tmp_path, index_path)  # atomic on POSIX


def shard_writer(
    q: mp.Queue,
    shards_root_str: str,
    shard_size: int,
    compress: bool,
    dataset_name: str = "coupler",
    index_name: str = "index.json",
    save_index_every_shard: bool = True,
):
    """
    Consumes temp geometry npz paths.
    For each geometry, produces TWO samples (inPort1 + inPort2 via symmetry).
    Writes shards incrementally. Deletes temp npz immediately after reading.

    Also writes index.json (atomically). If save_index_every_shard=True, index is
    updated after each shard write (preemption-friendly).
    """
    shards_root = Path(shards_root_str)
    shards_root.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, object]] = []
    index_path = shards_root / index_name

    buffer: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]] = []
    shard_id = 0

    def write_one_shard(batch: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]]):
        nonlocal shard_id, index
        shard_name = f"shard_{shard_id:05d}.npz"
        shard_path = shards_root / shard_name
        _write_shard_npz(shard_path, batch, compress=compress)
        for slot, (_, meta) in enumerate(batch):
            index.append({"tag": meta["tag"], "shard": shard_name, "slot": int(slot)})
        shard_id += 1
        if save_index_every_shard:
            _atomic_write_index(index_path, index)

    while True:
        msg = q.get()
        if msg is None:
            break

        status, payload = msg
        if status != "OK":
            continue

        tmp_path = Path(payload)
        try:
            with np.load(tmp_path, allow_pickle=True) as f:
                # robust scalar string extraction
                base_tag = f["base_tag"].item()
                if isinstance(base_tag, bytes):
                    base_tag = base_tag.decode("utf-8")
                base_tag = str(base_tag)

                eps = f["eps"]
                Ezr = f["Ez_real"]
                Ezi = f["Ez_imag"]
                src1 = f["src_mask"]
                port_ids = f["port_ids"].astype(np.int32) if "port_ids" in f else None
                port_masks = f["port_masks"].astype(np.float32) if "port_masks" in f else None
                S1 = f["S1_real"] + 1j * f["S1_imag"]

                # scalars
                wg_width_um = float(f["wg_width_um"])
                gap_um = float(f["gap_um"])
                Lc_um = float(f["Lc_um"])
                bend_length_um = float(f["bend_length_um"])
                lead_extra_gap_um = float(f["lead_extra_gap_um"])
                wavelength_um = float(f["wavelength_um"])
                resolution = int(f["resolution"])

                nx = int(f["nx"])
                ny = int(f["ny"])
                dx = float(f["dx"])
                dy = float(f["dy"])
                Lx_um = float(f["Lx_um"])
                Ly_um = float(f["Ly_um"])

            # sample 1 (inPort1)
            tag1 = f"{base_tag}__inPort1"
            arrays1 = {
                "eps": eps.astype(np.float32),
                "Ez_real": Ezr.astype(np.float32),
                "Ez_imag": Ezi.astype(np.float32),
                "src_mask": src1.astype(np.float32),
                **(
                    {
                "ports/ids": port_ids.astype(np.int32),
                "ports/masks": port_masks.astype(np.float32),
                    }
                    if (port_ids is not None and port_masks is not None)
                    else {}
                ),

                "sparams/input_port": np.array(1, dtype=np.int32),
                "sparams/source_off": np.array(0, dtype=np.int32),
                "sparams/gap_um": np.array(gap_um, dtype=np.float32),
                "sparams/Lc_um": np.array(Lc_um, dtype=np.float32),
                "sparams/bend_length_um": np.array(bend_length_um, dtype=np.float32),
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/lead_extra_gap_um": np.array(lead_extra_gap_um, dtype=np.float32),
                "sparams/wavelength_um": np.array(wavelength_um, dtype=np.float32),
                "sparams/resolution": np.array(resolution, dtype=np.int32),
                "sparams/S_real": S1.real.astype(np.float32),
                "sparams/S_imag": S1.imag.astype(np.float32),

                "grid/dx": np.array(dx, dtype=np.float32),
                "grid/dy": np.array(dy, dtype=np.float32),
                "grid/nx": np.array(nx, dtype=np.int32),
                "grid/ny": np.array(ny, dtype=np.int32),
                "grid/Lx_um": np.array(Lx_um, dtype=np.float32),
                "grid/Ly_um": np.array(Ly_um, dtype=np.float32),
            }
            meta1 = {"tag": tag1, "dataset": dataset_name}

            # sample 2 (inPort2 via symmetry)
            tag2 = f"{base_tag}__inPort2"
            Ezr2 = flip_y(Ezr)
            Ezi2 = flip_y(Ezi)
            eps2 = flip_y(eps)
            src2 = flip_y(src1)
            S2 = synthesize_s_for_port2_from_port1(S1)

            # For the y-flipped sample, port numbering maps 1<->2 and 3<->4.
            # Build the mirrored port masks in canonical port order [1,2,3,4].
            if port_masks is not None and port_ids is not None and port_ids.shape[0] == 4:
                m1, m2, m3, m4 = port_masks[0], port_masks[1], port_masks[2], port_masks[3]
                port_masks2 = np.stack([flip_y(m2), flip_y(m1), flip_y(m4), flip_y(m3)], axis=0).astype(np.float32)
                port_ids2 = port_ids.copy()
            elif port_masks is not None and port_ids is not None:
                port_masks2 = flip_y(port_masks).astype(np.float32)
                port_ids2 = port_ids.copy()
            else:
                port_masks2 = None
                port_ids2 = None

            arrays2 = {
                "eps": eps2.astype(np.float32),
                "Ez_real": Ezr2.astype(np.float32),
                "Ez_imag": Ezi2.astype(np.float32),
                "src_mask": src2.astype(np.float32),
                **(
                    {
                "ports/ids": port_ids2.astype(np.int32),
                "ports/masks": port_masks2.astype(np.float32),
                    }
                    if (port_ids2 is not None and port_masks2 is not None)
                    else {}
                ),

                "sparams/input_port": np.array(2, dtype=np.int32),
                "sparams/source_off": np.array(0, dtype=np.int32),
                "sparams/gap_um": np.array(gap_um, dtype=np.float32),
                "sparams/Lc_um": np.array(Lc_um, dtype=np.float32),
                "sparams/bend_length_um": np.array(bend_length_um, dtype=np.float32),
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/lead_extra_gap_um": np.array(lead_extra_gap_um, dtype=np.float32),
                "sparams/wavelength_um": np.array(wavelength_um, dtype=np.float32),
                "sparams/resolution": np.array(resolution, dtype=np.int32),
                "sparams/S_real": S2.real.astype(np.float32),
                "sparams/S_imag": S2.imag.astype(np.float32),

                "grid/dx": np.array(dx, dtype=np.float32),
                "grid/dy": np.array(dy, dtype=np.float32),
                "grid/nx": np.array(nx, dtype=np.int32),
                "grid/ny": np.array(ny, dtype=np.int32),
                "grid/Lx_um": np.array(Lx_um, dtype=np.float32),
                "grid/Ly_um": np.array(Ly_um, dtype=np.float32),
            }
            meta2 = {"tag": tag2, "dataset": dataset_name}

            buffer.append((arrays1, meta1))
            buffer.append((arrays2, meta2))

            # write full shards
            while len(buffer) >= shard_size:
                batch = buffer[:shard_size]
                buffer = buffer[shard_size:]
                write_one_shard(batch)

        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    # final partial shard
    if buffer:
        write_one_shard(buffer)

    # ensure final index exists even if save_index_every_shard=False
    if not save_index_every_shard:
        _atomic_write_index(index_path, index)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None, help="Output root (default: <repo>/Data/coupler_sweep)")
    ap.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--n-procs", type=int, default=N_PROCS_DEFAULT)

    ap.add_argument("--shard-size", type=int, default=100, help="Samples per shard (NOTE: each geometry yields 2 samples).")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed for shards (smaller, slower).")
    ap.add_argument("--queue-max", type=int, default=64, help="Max queued temp files (bounds disk usage).")
    ap.add_argument("--index-every-shard", action="store_true", help="Write index.json after every shard (recommended).")

    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR = Path(args.out_dir) if args.out_dir is not None else (repo_root / "Data" / "coupler_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = OUT_DIR / "tmp_geom"
    shards_dir = OUT_DIR / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    params = build_param_list(args.n_geo, seed_base=42)

    print(f"Unique geometries: {len(params)}")
    print(f"Total samples produced (with symmetry): {2 * len(params)}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"TMP_DIR:   {tmp_dir}  (bounded by queue_max ~ {args.queue_max})")
    print(f"SHARDS:    {shards_dir}")
    print(f"n_procs:   {args.n_procs}")
    print(f"shard_size(samples): {args.shard_size}  => ~{math.ceil((2*len(params))/args.shard_size)} shards")
    print(f"compress:  {bool(args.compress)}")
    print(f"index-every-shard: {bool(args.index_every_shard)}")

    q: mp.Queue = mp.Queue(maxsize=args.queue_max)

    writer_p = mp.Process(
        target=shard_writer,
        args=(
            q,
            str(shards_dir),
            int(args.shard_size),
            bool(args.compress),
            "coupler",
            "index.json",
            bool(args.index_every_shard),
        ),
        daemon=True,
    )
    writer_p.start()

    tasks = [(w, g, L, b, lead, lam, args.resolution, str(tmp_dir)) for (w, g, L, b, lead, lam) in params]

    successes, failures = 0, 0

    with mp.Pool(processes=args.n_procs) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="FDTD (port1) -> tmp"):
            if status == "OK":
                q.put(("OK", payload))
                successes += 1
            else:
                failures += 1
                q.put(("ERR", payload))

    q.put(None)
    writer_p.join()

    # Cleanup tmp dir if empty
    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    print(f"Done. Geom successes={successes}, failures={failures}")
    print(f"Shards written to: {shards_dir / 'shard_*.npz'}")
    print(f"Index written to:  {shards_dir / 'index.json'}")


if __name__ == "__main__":
    main()
