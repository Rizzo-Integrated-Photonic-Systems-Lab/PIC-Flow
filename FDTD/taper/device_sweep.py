import argparse
import os
from pathlib import Path

import numpy as np
import meep as mp

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taper import TaperWaveguide2D
from utils import get_mode_alpha_2dir, pick_in_out_from_alpha
from sweep_utils import latin_hypercube, quantize_01, parse_wavelengths, assign_splits, ensure_dir, write_jsonl, make_geom_id


WAVELENGTHS_DEFAULT = [1.45, 1.50, 1.55, 1.60]
N_GEO_DEFAULT = 1500

WG_IN_MIN, WG_IN_MAX = 0.38, 0.60
WG_OUT_MIN, WG_OUT_MAX = 0.60, 2.50
TAPER_LEN_MIN, TAPER_LEN_MAX = 6.0, 16.0


def run_two_port_sim(dev, input_port, decay_tol, dir_plus, dir_minus):
    toward = {1: +1, 2: -1}
    sim, (m1, m2), dft, _fcen = dev._build_sim_single(input_port=int(input_port), df_frac=0.1)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

    eps_mid = sim.get_epsilon().T.astype(np.float32)
    Ez_mid = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

    alpha_1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
    alpha_2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)

    a1_in, b1_out = pick_in_out_from_alpha(alpha_1, toward[1], dir_plus=dir_plus, dir_minus=dir_minus)
    a2_in, b2_out = pick_in_out_from_alpha(alpha_2, toward[2], dir_plus=dir_plus, dir_minus=dir_minus)

    if int(input_port) == 1:
        S = {"S11": b1_out / a1_in, "S21": b2_out / a1_in}
    else:
        S = {"S22": b2_out / a2_in, "S12": b1_out / a2_in}

    sim.reset_meep()
    return eps_mid, Ez_mid, S, (dev.cell_x, dev.cell_y)


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
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "Data" / "taper_sweep")
    sims_dir = out_dir / "sims"
    ensure_dir(sims_dir)

    u = latin_hypercube(args.n_geo, d=3, seed=args.seed)
    wg_in = WG_IN_MIN + u[:, 0] * (WG_IN_MAX - WG_IN_MIN)
    wg_out = WG_OUT_MIN + u[:, 1] * (WG_OUT_MAX - WG_OUT_MIN)
    taper_len = TAPER_LEN_MIN + u[:, 2] * (TAPER_LEN_MAX - TAPER_LEN_MIN)
    wg_in = quantize_01(wg_in, WG_IN_MIN, WG_IN_MAX)
    wg_out = quantize_01(wg_out, WG_OUT_MIN, WG_OUT_MAX)
    taper_len = quantize_01(taper_len, TAPER_LEN_MIN, TAPER_LEN_MAX)

    splits = assign_splits(args.n_geo, seed=args.split_seed)
    geom_rows = []
    sim_rows = []

    rng = np.random.default_rng(int(args.seed))

    for i in range(args.n_geo):
        geom_id = make_geom_id("taper")
        input_port = int(rng.choice([1, 2]))

        params = {
            "wg_width_in_um": float(wg_in[i]),
            "wg_width_out_um": float(wg_out[i]),
            "taper_length_um": float(taper_len[i]),
        }

        geom_rows.append({
            "geometry_id": geom_id,
            "device": "taper",
            "split": splits[i],
            "input_port": input_port,
            **params,
        })

        for lam in wavelengths:
            dev = TaperWaveguide2D(
                wg_width_in=params["wg_width_in_um"],
                wg_width_out=params["wg_width_out_um"],
                taper_length_um=params["taper_length_um"],
                wavelength_um=float(lam),
                resolution=args.resolution,
                dpml=args.dpml,
                cell_x=args.cell_x,
                cell_y=args.cell_y,
            )

            eps, Ez, S, (cell_x, cell_y) = run_two_port_sim(
                dev,
                input_port=input_port,
                decay_tol=args.decay_tol,
                dir_plus=args.dir_plus,
                dir_minus=args.dir_minus,
            )

            sim_id = f"{geom_id}_lam{float(lam):.4f}"
            out_path = sims_dir / f"{sim_id}.npz"

            np.savez_compressed(
                out_path,
                geometry_id=np.array(geom_id),
                device=np.array("taper"),
                split=np.array(splits[i]),
                input_port=np.int32(input_port),
                wavelength_um=np.float32(lam),
                cell_x_um=np.float32(cell_x),
                cell_y_um=np.float32(cell_y),
                resolution=np.int32(dev.resolution),
                dpml_um=np.float32(dev.dpml),
                eps=eps.astype(np.float32),
                Ez_real=Ez.real.astype(np.float32),
                Ez_imag=Ez.imag.astype(np.float32),
                **{f"sparams/{k}_real": np.float32(np.real(v)) for k, v in S.items()},
                **{f"sparams/{k}_imag": np.float32(np.imag(v)) for k, v in S.items()},
                **{f"params/{k}": np.float32(v) for k, v in params.items()},
            )

            sim_rows.append({
                "sim_id": sim_id,
                "geometry_id": geom_id,
                "device": "taper",
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





