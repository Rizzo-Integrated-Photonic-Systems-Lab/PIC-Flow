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
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x

from ybranch import YBranch2D
from conditioning_masks import make_source_mask


# -----------------------------
# Config defaults
# -----------------------------
RESOLUTION_DEFAULT = 32
N_GEO_DEFAULT = 5000
N_PROCS_DEFAULT = 24

# Sweep ranges (um)
wg_width_min, wg_width_max = 0.38, 0.60
lambda_min, lambda_max = 1.40, 1.60
l_j_min, l_j_max = 1.5, 3.5
l_bend_min, l_bend_max = 5.0, 8.0
h_bend_min, h_bend_max = 0.5, 1.2
l_out_min, l_out_max = 1.0, 15.0


# -----------------------------
# Naming utilities (collision-resistant)
# -----------------------------
def _as_code_um(x_um: float, scale: int = 1000) -> int:
    return int(np.round(float(x_um) * scale))


def geom_tag(
    wg_width: float,
    lam: float,
    l_j: float,
    l_bend: float,
    h_bend: float,
    l_out: float,
) -> str:
    # integer-coded fields for uniqueness + readable floats for sanity
    w_i = _as_code_um(wg_width, 1000)
    lam_i = _as_code_um(lam, 1000)
    lj_i = _as_code_um(l_j, 1000)
    lb_i = _as_code_um(l_bend, 1000)
    hb_i = _as_code_um(h_bend, 1000)
    lo_i = _as_code_um(l_out, 1000)

    return (
        f"w{w_i:04d}nm_lam{lam_i:04d}nm_lj{lj_i:05d}nm_lb{lb_i:05d}nm_hb{hb_i:05d}nm_lout{lo_i:05d}nm"
        f"__w{wg_width:.3f}_lam{lam:.2f}_lj{l_j:.2f}_lb{l_bend:.2f}_hb{h_bend:.2f}_lout{l_out:.2f}"
    )


# -----------------------------
# Sampling: LHS over 6 params
# -----------------------------
def _quantize_01(x, x_min, x_max):
    xq = np.round(x * 100.0) / 100.0
    return np.clip(xq, x_min, x_max)


def _latin_hypercube(n: int, d: int, seed: int = 42) -> np.ndarray:
    """
    Pure-numpy Latin hypercube sampling in [0,1)^d.

    For each dimension, we stratify into n equal bins and draw one sample
    uniformly from each bin, then independently permute bins per dimension.
    """
    rng = np.random.default_rng(seed)
    # base strata edges [0, 1/n, 2/n, ..., (n-1)/n]
    cut = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    a = cut[:-1]
    b = cut[1:]
    # Avoid relying on newer numpy Generator.random(dtype=...) signatures
    u = rng.random((n, d)).astype(np.float64, copy=False)
    # sample within each stratum for each dim
    H = u * (b - a)[:, None] + a[:, None]  # (n, d) with broadcasting
    # permute strata independently per dimension
    for j in range(d):
        rng.shuffle(H[:, j])
    return H.astype(np.float32, copy=False)


def build_param_list(
    N_GEO: int, seed_base: int = 42
) -> list[tuple[float, float, float, float, float, float]]:
    # [wg_width, lambda, l_j, l_bend, h_bend, l_out]
    u = _latin_hypercube(N_GEO, d=6, seed=seed_base)

    wg_widths = wg_width_min + u[:, 0] * (wg_width_max - wg_width_min)
    wavelengths = lambda_min + u[:, 1] * (lambda_max - lambda_min)
    l_js = l_j_min + u[:, 2] * (l_j_max - l_j_min)
    l_bends = l_bend_min + u[:, 3] * (l_bend_max - l_bend_min)
    h_bends = h_bend_min + u[:, 4] * (h_bend_max - h_bend_min)
    l_outs = l_out_min + u[:, 5] * (l_out_max - l_out_min)

    # Quantize only width and wavelength (matches old script behavior)
    wg_widths = _quantize_01(wg_widths, wg_width_min, wg_width_max)
    wavelengths = _quantize_01(wavelengths, lambda_min, lambda_max)

    return [
        (float(w), float(lam), float(lj), float(lb), float(hb), float(lo))
        for w, lam, lj, lb, hb, lo in zip(wg_widths, wavelengths, l_js, l_bends, h_bends, l_outs)
    ]


# -----------------------------
# Worker: run sim, write ONE temp npz, return path
# -----------------------------
def run_fdtd_sim(
    wg_width: float,
    wavelength: float,
    l_j: float,
    l_bend: float,
    h_bend: float,
    l_out: float,
    RESOLUTION: int,
):
    yb = YBranch2D(
        wg_width_um=wg_width,
        wavelength_um=wavelength,
        resolution=RESOLUTION,
        l_junction_um=l_j,
        l_bend_um=l_bend,
        h_bend_um=h_bend,
        l_out_um=l_out,
        bend_n_segments=96,
    )

    # convention: eps, Ez, Hx, Hy, S_dict, cell
    eps, Ez, Hx, Hy, S_dict, cell = yb.run_sim(input_port=1, decay_tol=1e-5)
    return yb, S_dict, Ez, eps, cell


def worker(task):
    """
    Returns:
      ("OK", temp_npz_path_str) or ("ERR", err_string)
    """
    (wg_width, lam, l_j, l_bend, h_bend, l_out, RESOLUTION, tmp_dir_str) = task
    tmp_dir = Path(tmp_dir_str)

    base = geom_tag(wg_width, lam, l_j, l_bend, h_bend, l_out)
    tmp_name = f"geom_{base}__{uuid.uuid4().hex}.npz"
    tmp_path = tmp_dir / tmp_name

    try:
        yb, S_dict, Ez1, eps, cell = run_fdtd_sim(
            wg_width, lam, l_j, l_bend, h_bend, l_out, RESOLUTION
        )
        if S_dict is None:
            raise RuntimeError("S_dict is None")
        if Ez1 is None or eps is None or cell is None:
            raise RuntimeError("Missing Ez/eps/cell")

        eps = np.asarray(eps, dtype=np.float32)
        Ez1 = np.asarray(Ez1, dtype=np.complex64)

        ny, nx = eps.shape
        Lx_um, Ly_um = cell

        # Port masks for ports 1..3 (for S-parameter projection / auxiliary losses)
        port_ids = np.array([1, 2, 3], dtype=np.int32)
        port_centers = yb.get_port_centers_um()
        y_span = yb.get_port_y_span_um()
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

        # S vector for input=1 (ports: 1,2,3)
        S1 = np.array([S_dict[(1, 1)], S_dict[(2, 1)], S_dict[(3, 1)]], dtype=np.complex128)

        # grid meta
        dx = 1.0 / RESOLUTION
        dy = 1.0 / RESOLUTION

        np.savez_compressed(
            tmp_path,
            # geometry scalars
            wg_width_um=np.float32(wg_width),
            wavelength_um=np.float32(lam),
            l_junction_um=np.float32(l_j),
            l_bend_um=np.float32(l_bend),
            h_bend_um=np.float32(h_bend),
            l_out_um=np.float32(l_out),
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
def _write_shard_npz(
    out_path: Path,
    samples: List[Tuple[Dict[str, np.ndarray], Dict[str, object]]],
    compress: bool,
):
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
    dataset_name: str = "y_branch",
    index_name: str = "index.json",
    save_index_every_shard: bool = True,
):
    """
    Consumes temp geometry npz paths.
    For each geometry, produces ONE sample (inPort1).
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

                wg_width_um = float(f["wg_width_um"])
                wavelength_um = float(f["wavelength_um"])
                l_junction_um = float(f["l_junction_um"])
                l_bend_um = float(f["l_bend_um"])
                h_bend_um = float(f["h_bend_um"])
                l_out_um = float(f["l_out_um"])
                resolution = int(f["resolution"])

                nx = int(f["nx"])
                ny = int(f["ny"])
                dx = float(f["dx"])
                dy = float(f["dy"])
                Lx_um = float(f["Lx_um"])
                Ly_um = float(f["Ly_um"])

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
                "sparams/wg_width_um": np.array(wg_width_um, dtype=np.float32),
                "sparams/wavelength_um": np.array(wavelength_um, dtype=np.float32),
                "sparams/l_junction_um": np.array(l_junction_um, dtype=np.float32),
                "sparams/l_bend_um": np.array(l_bend_um, dtype=np.float32),
                "sparams/h_bend_um": np.array(h_bend_um, dtype=np.float32),
                "sparams/l_out_um": np.array(l_out_um, dtype=np.float32),
                "sparams/resolution": np.array(resolution, dtype=np.int32),

                # For completeness / debugging (not used by FDTDDataset currently)
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

            buffer.append((arrays1, meta1))

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

    if buffer:
        write_one_shard(buffer)

    if not save_index_every_shard:
        _atomic_write_index(index_path, index)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output root (default: <repo>/Data/y_branch_sweep)",
    )
    ap.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--n-procs", type=int, default=N_PROCS_DEFAULT)

    ap.add_argument("--shard-size", type=int, default=100, help="Samples per shard.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed for shards (smaller, slower).")
    ap.add_argument("--queue-max", type=int, default=64, help="Max queued temp files (bounds disk usage).")
    ap.add_argument("--index-every-shard", action="store_true", help="Write index.json after every shard (recommended).")

    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR = Path(args.out_dir) if args.out_dir is not None else (repo_root / "Data" / "y_branch_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = OUT_DIR / "tmp_geom"
    shards_dir = OUT_DIR / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    params = build_param_list(args.n_geo, seed_base=42)

    print(f"Unique geometries: {len(params)}")
    print(f"Total samples produced: {len(params)}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"TMP_DIR:   {tmp_dir}  (bounded by queue_max ~ {args.queue_max})")
    print(f"SHARDS:    {shards_dir}")
    print(f"n_procs:   {args.n_procs}")
    print(f"shard_size(samples): {args.shard_size}  => ~{math.ceil((len(params))/args.shard_size)} shards")
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
            "y_branch",
            "index.json",
            bool(args.index_every_shard),
        ),
        daemon=True,
    )
    writer_p.start()

    tasks = [
        (w, lam, lj, lb, hb, lo, args.resolution, str(tmp_dir))
        for (w, lam, lj, lb, hb, lo) in params
    ]

    successes, failures = 0, 0

    with mp.Pool(processes=args.n_procs) as pool:
        for status, payload in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="FDTD (ybranch) -> tmp"):
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

