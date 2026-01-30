import argparse
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossing import UniformCrossing2D
from sweep_utils import latin_hypercube, quantize_01, parse_wavelengths, assign_splits, ensure_dir, write_jsonl, make_geom_id


WAVELENGTHS_DEFAULT = [1.45, 1.50, 1.55, 1.60]
N_GEO_DEFAULT = 1500

WG_WIDTH_MIN, WG_WIDTH_MAX = 0.38, 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--wavelengths", type=str, default=",".join([str(x) for x in WAVELENGTHS_DEFAULT]))
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--dpml", type=float, default=None)
    ap.add_argument("--cell-x", type=float, default=None)
    ap.add_argument("--cell-y", type=float, default=None)
    ap.add_argument("--decay-tol", type=float, default=1e-5)
    ap.add_argument("--dir-plus", type=int, default=0)
    ap.add_argument("--dir-minus", type=int, default=1)
    args = ap.parse_args()

    wavelengths = parse_wavelengths(args.wavelengths)
    if not wavelengths:
        raise ValueError("No wavelengths provided")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "Data" / "crossing_sweep")
    sims_dir = out_dir / "sims"
    ensure_dir(sims_dir)

    u = latin_hypercube(args.n_geo, d=2, seed=args.seed)
    wg_h = WG_WIDTH_MIN + u[:, 0] * (WG_WIDTH_MAX - WG_WIDTH_MIN)
    wg_v = WG_WIDTH_MIN + u[:, 1] * (WG_WIDTH_MAX - WG_WIDTH_MIN)
    wg_h = quantize_01(wg_h, WG_WIDTH_MIN, WG_WIDTH_MAX)
    wg_v = quantize_01(wg_v, WG_WIDTH_MIN, WG_WIDTH_MAX)

    splits = assign_splits(args.n_geo, seed=args.split_seed)
    geom_rows = []
    sim_rows = []

    rng = np.random.default_rng(int(args.seed))

    for i in range(args.n_geo):
        geom_id = make_geom_id("crossing")
        input_port = int(rng.choice([1, 2, 3, 4]))

        params = {
            "wg_width_h_um": float(wg_h[i]),
            "wg_width_v_um": float(wg_v[i]),
        }

        geom_rows.append({
            "geometry_id": geom_id,
            "device": "crossing",
            "split": splits[i],
            "input_port": input_port,
            **params,
        })

        for lam in wavelengths:
            dev = UniformCrossing2D(
                wg_width_h=params["wg_width_h_um"],
                wg_width_v=params["wg_width_v_um"],
                wavelength_um=float(lam),
                resolution=args.resolution,
                dpml=args.dpml,
                cell_x=args.cell_x,
                cell_y=args.cell_y,
            )

            eps, Ez, S, (cell_x, cell_y) = dev.run_sim(
                input_port=input_port,
                decay_tol=float(args.decay_tol),
                dir_plus=args.dir_plus,
                dir_minus=args.dir_minus,
            )

            sim_id = f"{geom_id}_lam{float(lam):.4f}"
            out_path = sims_dir / f"{sim_id}.npz"

            np.savez_compressed(
                out_path,
                geometry_id=np.array(geom_id),
                device=np.array("crossing"),
                split=np.array(splits[i]),
                input_port=np.int32(input_port),
                wavelength_um=np.float32(lam),
                cell_x_um=np.float32(cell_x),
                cell_y_um=np.float32(cell_y),
                resolution=np.int32(dev.resolution),
                dpml_um=np.float32(dev.dpml),
                eps=np.asarray(eps, dtype=np.float32),
                Ez_real=np.asarray(Ez.real, dtype=np.float32),
                Ez_imag=np.asarray(Ez.imag, dtype=np.float32),
                **{f"sparams/{k}_real": np.float32(np.real(v)) for k, v in S.items()},
                **{f"sparams/{k}_imag": np.float32(np.imag(v)) for k, v in S.items()},
                **{f"params/{k}": np.float32(v) for k, v in params.items()},
            )

            sim_rows.append({
                "sim_id": sim_id,
                "geometry_id": geom_id,
                "device": "crossing",
                "split": splits[i],
                "input_port": input_port,
                "wavelength_um": float(lam),
                "file": str(out_path),
            })

    ensure_dir(out_dir)
    write_jsonl(out_dir / "geometries.jsonl", geom_rows)
    write_jsonl(out_dir / "sims.jsonl", sim_rows)


if __name__ == "__main__":
    main()





