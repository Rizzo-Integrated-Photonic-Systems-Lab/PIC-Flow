#!/usr/bin/env python3
"""
Generate random geometry audit previews for all active device types.

Samples parameters from the active unified_sweep ranges, builds each geometry on
the active final-dataset rectangular domain, and saves:
  - 10 individual preview PNGs per device type
  - one JSON metadata file per device type

Usage:
    python scripts/preview_random_device_samples.py
    python scripts/preview_random_device_samples.py --samples-per-device 10 --seed 123
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import meep as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FDTD_DIR = REPO_ROOT / "FDTD"
sys.path.insert(0, str(FDTD_DIR))
for sub in [
    "straight_waveguide",
    "mmi",
    "sbend",
    "ybranch",
    "directional_coupler",
    "taper",
    "euler_bend",
    "circular_bend",
    "crossing",
]:
    sys.path.insert(0, str(FDTD_DIR / sub))

from unified_sweep import build_device, get_device_masks, sample_params, INPUT_PORTS  # noqa: E402


RESOLUTION = 20
DPML_RAW = 1.0
PML_PX = int(round(DPML_RAW * RESOLUTION))
DPML = float(PML_PX) / float(RESOLUTION)
WAVELENGTH = 1.55
RECT_CROP_X = 480
RECT_CROP_Y = 160
RECT_CELL_X = float(RECT_CROP_X + 2 * PML_PX) / float(RESOLUTION)
RECT_CELL_Y = float(RECT_CROP_Y + 2 * PML_PX) / float(RESOLUTION)
RECT_INT_X = RECT_CELL_X - 2 * DPML
RECT_INT_Y = RECT_CELL_Y - 2 * DPML

DEVICE_ORDER = [
    ("straight", "Straight Waveguide"),
    ("taper", "Taper"),
    ("sbend", "S-Bend (Euler)"),
    ("ybranch", "Y-Branch"),
    ("directional_coupler", "Directional Coupler"),
    ("mmi", "2x2 MMI"),
    ("euler_bend", "Euler Bend (90°)"),
    ("circular_bend", "Circular Bend (90°)"),
    ("crossing", "Waveguide Crossing"),
]


def get_epsilon_fast(dev):
    sim = mp.Simulation(
        cell_size=dev.cell,
        resolution=dev.resolution,
        boundary_layers=[mp.PML(dev.dpml)],
        geometry=dev.geometry,
        default_material=mp.Medium(index=1.444),
        sources=[],
    )
    sim.init_sim()
    eps = sim.get_epsilon().T.astype(np.float32)
    sim.reset_meep()
    return eps


def crop_pml(arr):
    if PML_PX > 0:
        return arr[PML_PX:-PML_PX, PML_PX:-PML_PX]
    return arr


def format_params(params):
    bits = []
    for key, value in params.items():
        label = key.replace("_um", "").replace("_", " ")
        bits.append(f"{label}={value:.3f}")
    return ", ".join(bits)


def plot_sample(ax, eps, src_mask, port_masks, port_ids, title):
    extent = [-RECT_INT_X / 2, RECT_INT_X / 2, -RECT_INT_Y / 2, RECT_INT_Y / 2]
    ax.imshow(
        eps,
        origin="lower",
        cmap="gray_r",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )

    src_rgba = np.zeros((*src_mask.shape, 4), dtype=np.float32)
    src_rgba[..., 0] = 1.0
    src_rgba[..., 3] = src_mask * 0.6
    ax.imshow(src_rgba, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

    port_colors = [(0.0, 0.5, 1.0), (0.0, 0.8, 0.2), (1.0, 0.6, 0.0), (0.8, 0.0, 0.8)]
    for i in range(port_masks.shape[0]):
        rgba = np.zeros((*port_masks[i].shape, 4), dtype=np.float32)
        c = port_colors[i % len(port_colors)]
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c
        rgba[..., 3] = port_masks[i] * 0.5
        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (um)", fontsize=7)
    ax.set_ylabel("y (um)", fontsize=7)
    ax.tick_params(labelsize=6)

    legend_items = [Patch(facecolor="red", alpha=0.6, label="Source")]
    for i, pid in enumerate(port_ids):
        c = port_colors[i % len(port_colors)]
        legend_items.append(Patch(facecolor=c, alpha=0.5, label=f"Port {pid}"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=5, framealpha=0.7)


def build_preview(device_type, display_name, params, input_port):
    dev = build_device(
        device_type=device_type,
        params=params,
        wavelength_um=WAVELENGTH,
        resolution=RESOLUTION,
        dpml=DPML,
        cell_x=RECT_CELL_X,
        cell_y=RECT_CELL_Y,
        crop_x_px=RECT_CROP_X,
        crop_y_px=RECT_CROP_Y,
    )
    eps = crop_pml(get_epsilon_fast(dev))
    ny, nx = eps.shape
    src_mask, port_ids, port_masks = get_device_masks(
        dev,
        device_type,
        input_port=input_port,
        cell_x=RECT_CELL_X,
        cell_y=RECT_CELL_Y,
        dpml=DPML,
        resolution=RESOLUTION,
        ny=ny,
        nx=nx,
        thickness_px=3,
    )
    title = f"{display_name}\ninput={input_port}\n{format_params(params)}"
    return {
        "eps": eps,
        "src_mask": src_mask,
        "port_ids": port_ids,
        "port_masks": port_masks,
        "title": title,
        "params": params,
        "input_port": int(input_port),
    }


def save_single_preview(device_type, sample_idx, preview, output_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    plot_sample(
        ax,
        preview["eps"],
        preview["src_mask"],
        preview["port_masks"],
        preview["port_ids"],
        preview["title"],
    )
    out_path = output_dir / f"{device_type}_sample_{sample_idx:02d}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-device", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "figures" / "random_device_audit_10")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    summary = []

    print(f"Saving random device previews to: {output_dir}")
    for idx, (device_type, display_name) in enumerate(DEVICE_ORDER):
        params_list = sample_params(device_type, args.samples_per_device, seed=args.seed + idx, resolution=RESOLUTION)
        metadata = []
        device_dir = output_dir / device_type
        device_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building {device_type}: {len(params_list)} samples")
        for sample_idx, params in enumerate(params_list):
            possible_ports = INPUT_PORTS[device_type]
            input_port = int(rng.choice(possible_ports))
            preview = build_preview(device_type, display_name, params, input_port)
            image_path = save_single_preview(device_type, sample_idx, preview, device_dir)
            metadata.append(
                {
                    "sample_idx": sample_idx,
                    "device": device_type,
                    "display_name": display_name,
                    "input_port": input_port,
                    "params": params,
                    "image": str(image_path),
                }
            )

        meta_path = output_dir / f"{device_type}_random_10.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        summary.append(
            {
                "device": device_type,
                "folder": str(device_dir),
                "metadata": str(meta_path),
            }
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
