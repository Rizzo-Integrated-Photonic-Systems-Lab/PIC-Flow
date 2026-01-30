import argparse
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ybranch import YBranch2D
from sweep_utils import latin_hypercube, quantize_01, parse_wavelengths, assign_splits, ensure_dir, write_jsonl, make_geom_id


WAVELENGTHS_DEFAULT = [1.45, 1.50, 1.55, 1.60]
N_GEO_DEFAULT = 1500

WG_WIDTH_MIN, WG_WIDTH_MAX = 0.38, 0.60
L_J_MIN, L_J_MAX = 1.2, 3.2
L_BEND_MIN, L_BEND_MAX = 4.0, 7.0
H_BEND_MIN, H_BEND_MAX = 0.6, 2.8
L_OUT_MIN, L_OUT_MAX = 1.0, 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--n-geo", type=int, default=N_GEO_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--wavelengths", type=str, default=",".join([str(x) for x in WAVELENGTHS_DEFAULT]))
    ap.add_argument("--resolution", type=int, default=20)
    ap.add_argument("--dpml", type=float, default=2.0 / 3.0)
    ap.add_argument("--crop-px", type=int, default=384)
    ap.add_argument("--decay-tol", type=float, default=1e-5)
    args = ap.parse_args()

    wavelengths = parse_wavelengths(args.wavelengths)
    if not wavelengths:
        raise ValueError("No wavelengths provided")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "Data" / "ybranch_sweep")
    sims_dir = out_dir / "sims"
    ensure_dir(sims_dir)

    u = latin_hypercube(args.n_geo, d=5, seed=args.seed)
    wg_widths = WG_WIDTH_MIN + u[:, 0] * (WG_WIDTH_MAX - WG_WIDTH_MIN)
    l_js = L_J_MIN + u[:, 1] * (L_J_MAX - L_J_MIN)
    l_bends = L_BEND_MIN + u[:, 2] * (L_BEND_MAX - L_BEND_MIN)
    h_bends = H_BEND_MIN + u[:, 3] * (H_BEND_MAX - H_BEND_MIN)
    l_outs = L_OUT_MIN + u[:, 4] * (L_OUT_MAX - L_OUT_MIN)
    wg_widths = quantize_01(wg_widths, WG_WIDTH_MIN, WG_WIDTH_MAX)

    splits = assign_splits(args.n_geo, seed=args.split_seed)
    geom_rows = []
    sim_rows = []

    # Random generator for input port selection
    rng = np.random.default_rng(int(args.seed))

    for i in range(args.n_geo):
        geom_id = make_geom_id("ybranch")
        # Randomly select input port: 1 = splitter, 2 or 3 = combiner
        input_port = int(rng.choice([1, 2, 3]))

        params = {
            "wg_width_um": float(wg_widths[i]),
            "l_junction_um": float(l_js[i]),
            "l_bend_um": float(l_bends[i]),
            "h_bend_um": float(h_bends[i]),
            "l_out_um": float(l_outs[i]),
        }

        geom_rows.append({
            "geometry_id": geom_id,
            "device": "ybranch",
            "split": splits[i],
            "input_port": input_port,
            **params,
        })

        for lam in wavelengths:
            dev = YBranch2D(
                wg_width_um=params["wg_width_um"],
                wavelength_um=float(lam),
                resolution=int(args.resolution),
                dpml=float(args.dpml),
                crop_px=int(args.crop_px),
                quantize_grid=True,
                fit_margin_um=0.5,
                l_junction_um=params["l_junction_um"],
                l_bend_um=params["l_bend_um"],
                h_bend_um=params["h_bend_um"],
                l_out_um=params["l_out_um"],
            )

            eps, Ez, _Hx, _Hy, S, (cell_x, cell_y) = dev.run_sim(input_port=input_port, decay_tol=float(args.decay_tol))

            sim_id = f"{geom_id}_lam{float(lam):.4f}"
            out_path = sims_dir / f"{sim_id}.npz"

            # S-parameters depend on input port
            sparams = {}
            for p in (1, 2, 3):
                sparams[f"S{p}{input_port}"] = S[(p, input_port)]

            np.savez_compressed(
                out_path,
                geometry_id=np.array(geom_id),
                device=np.array("ybranch"),
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
                **{f"sparams/{k}_real": np.float32(np.real(v)) for k, v in sparams.items()},
                **{f"sparams/{k}_imag": np.float32(np.imag(v)) for k, v in sparams.items()},
                **{f"params/{k}": np.float32(v) for k, v in params.items()},
            )

            sim_rows.append({
                "sim_id": sim_id,
                "geometry_id": geom_id,
                "device": "ybranch",
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





