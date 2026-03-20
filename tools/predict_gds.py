#!/usr/bin/env python3
"""
PHASE Field Predictor — Predict electromagnetic fields from GDS device layouts.

Usage:
    python tools/predict_gds.py \
        --gds device.gds \
        --checkpoint logs/checkpoints/0000080.pt \
        --wavelength 1.55 \
        --source-port left \
        --output output.npz

    # Custom layer-to-material mapping
    python tools/predict_gds.py \
        --gds device.gds \
        --checkpoint model.pt \
        --wavelength 1.55 \
        --source-port left \
        --layer-map '{"1/0": 12.25, "2/0": 2.07}' \
        --eps-clad 2.07

    # Specify source port by coordinates
    python tools/predict_gds.py \
        --gds device.gds \
        --checkpoint model.pt \
        --wavelength 1.55 \
        --source-xy -10.0 0.0 \
        --output output.npz
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# GDS loading & rasterization
# ---------------------------------------------------------------------------


def load_gds_polygons(gds_path: str, cell_name: str = None):
    """Load polygons from a GDS file using gdstk.

    Returns:
        cell: gdstk Cell object
        polygons_by_layer: dict mapping (layer, datatype) -> list of polygon arrays
    """
    try:
        import gdstk
    except ImportError:
        print("ERROR: gdstk is required. Install with: pip install gdstk", file=sys.stderr)
        sys.exit(1)

    lib = gdstk.read_gds(gds_path)

    if cell_name:
        cell = None
        for c in lib.cells:
            if c.name == cell_name:
                cell = c
                break
        if cell is None:
            available = [c.name for c in lib.cells]
            print(f"ERROR: Cell '{cell_name}' not found. Available: {available}", file=sys.stderr)
            sys.exit(1)
    else:
        # Use top-level cell (cell not referenced by others)
        top_cells = lib.top_level()
        if not top_cells:
            print("ERROR: No top-level cells found in GDS.", file=sys.stderr)
            sys.exit(1)
        cell = top_cells[0]
        if len(top_cells) > 1:
            print(f"WARNING: Multiple top cells found, using '{cell.name}'", file=sys.stderr)

    polygons_by_layer = {}
    for poly in cell.polygons:
        key = (poly.layer, poly.datatype)
        polygons_by_layer.setdefault(key, []).append(poly.points)

    for path in cell.paths:
        key = (path.layers[0], path.datatypes[0])
        # Convert paths to polygons
        for poly in path.to_polygons():
            polygons_by_layer.setdefault(key, []).append(poly)

    print(f"Loaded cell '{cell.name}' with layers: {sorted(polygons_by_layer.keys())}")
    return cell, polygons_by_layer


def rasterize_polygons(
    polygons_by_layer: dict,
    layer_to_eps: dict,
    eps_clad: float,
    dx: float,
    grid_shape: tuple,
    origin: tuple = None,
):
    """Rasterize GDS polygons to a permittivity grid.

    Args:
        polygons_by_layer: dict from load_gds_polygons
        layer_to_eps: dict mapping (layer, datatype) -> epsilon value
        eps_clad: cladding permittivity (background)
        dx: grid spacing in um
        grid_shape: (H, W) of output grid
        origin: (x_min, y_min) in um. If None, auto-computed from polygons.

    Returns:
        eps_grid: numpy array of shape (H, W)
        extent: (x_min, x_max, y_min, y_max) in um
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely import vectorized as sv
    except ImportError:
        sv = None

    H, W = grid_shape

    # Compute bounding box of all polygons
    all_pts = []
    for polys in polygons_by_layer.values():
        for pts in polys:
            all_pts.append(pts)
    if not all_pts:
        print("ERROR: No polygons found.", file=sys.stderr)
        sys.exit(1)

    all_pts_cat = np.concatenate(all_pts, axis=0)
    x_min_poly, y_min_poly = all_pts_cat.min(axis=0)
    x_max_poly, y_max_poly = all_pts_cat.max(axis=0)

    # Add padding (10% on each side)
    pad_x = (x_max_poly - x_min_poly) * 0.1
    pad_y = (y_max_poly - y_min_poly) * 0.1

    if origin is not None:
        x_min, y_min = origin
    else:
        x_min = x_min_poly - pad_x
        y_min = y_min_poly - pad_y

    x_max = x_min + W * dx
    y_max = y_min + H * dx

    # Create coordinate grids
    x_coords = np.linspace(x_min + dx / 2, x_max - dx / 2, W)
    y_coords = np.linspace(y_min + dx / 2, y_max - dx / 2, H)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Start with cladding
    eps_grid = np.full((H, W), eps_clad, dtype=np.float32)

    # Rasterize each layer
    for (layer, datatype), polys in polygons_by_layer.items():
        eps_val = layer_to_eps.get((layer, datatype))
        if eps_val is None:
            eps_val = layer_to_eps.get(f"{layer}/{datatype}")
        if eps_val is None:
            print(f"  Skipping layer ({layer}, {datatype}) — no epsilon mapping", file=sys.stderr)
            continue

        for pts in polys:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # Vectorized point-in-polygon if available
            if sv is not None:
                mask = sv.contains(poly, xx, yy)
            else:
                from shapely.geometry import Point
                mask = np.zeros((H, W), dtype=bool)
                for i in range(H):
                    for j in range(W):
                        mask[i, j] = poly.contains(Point(xx[i, j], yy[i, j]))

            eps_grid[mask] = eps_val

    extent = (x_min, x_max, y_min, y_max)
    return eps_grid, extent


def create_source_mask(
    grid_shape: tuple,
    extent: tuple,
    source_port: str = None,
    source_xy: tuple = None,
    source_width_um: float = 1.0,
    dx: float = 0.05,
):
    """Create a source mask on the grid.

    Args:
        grid_shape: (H, W)
        extent: (x_min, x_max, y_min, y_max)
        source_port: one of 'left', 'right', 'top', 'bottom'
        source_xy: (x, y) coordinates for source center
        source_width_um: width of source region in um
        dx: grid spacing

    Returns:
        src_mask: numpy array (H, W) with 1.0 at source pixels
    """
    H, W = grid_shape
    x_min, x_max, y_min, y_max = extent
    src_mask = np.zeros((H, W), dtype=np.float32)

    half_w = source_width_um / 2.0
    margin_px = 3  # pixels from edge

    if source_port is not None:
        port = source_port.lower()
        if port == "left":
            col = margin_px
            cy = H // 2
            hw = int(half_w / dx)
            src_mask[max(0, cy - hw) : min(H, cy + hw + 1), col] = 1.0
        elif port == "right":
            col = W - 1 - margin_px
            cy = H // 2
            hw = int(half_w / dx)
            src_mask[max(0, cy - hw) : min(H, cy + hw + 1), col] = 1.0
        elif port == "bottom":
            row = margin_px
            cx = W // 2
            hw = int(half_w / dx)
            src_mask[row, max(0, cx - hw) : min(W, cx + hw + 1)] = 1.0
        elif port == "top":
            row = H - 1 - margin_px
            cx = W // 2
            hw = int(half_w / dx)
            src_mask[row, max(0, cx - hw) : min(W, cx + hw + 1)] = 1.0
        else:
            print(f"ERROR: Unknown source port '{source_port}'. Use left/right/top/bottom.", file=sys.stderr)
            sys.exit(1)
    elif source_xy is not None:
        sx, sy = source_xy
        col = int((sx - x_min) / dx)
        row = int((sy - y_min) / dx)
        hw = int(half_w / dx)
        col = np.clip(col, 0, W - 1)
        src_mask[max(0, row - hw) : min(H, row + hw + 1), col] = 1.0
    else:
        # Default: left edge, center
        col = margin_px
        cy = H // 2
        hw = int(half_w / dx)
        src_mask[max(0, cy - hw) : min(H, cy + hw + 1), col] = 1.0

    return src_mask


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained PHASE model from checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Model"))
    from complex_physics_unet import ComplexPhysicsUNet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint if available
    model_args = ckpt.get("args", None)
    stats = ckpt.get("stats", {})

    # Build model — try to infer config from checkpoint
    state_dict = ckpt.get("ema", ckpt.get("model", None))
    if state_dict is None:
        print("ERROR: Checkpoint has no 'ema' or 'model' key.", file=sys.stderr)
        sys.exit(1)

    # Infer model_channels from first conv weight
    for key in state_dict:
        if "input_stem.conv_real.weight" in key:
            model_channels = state_dict[key].shape[0]
            break
    else:
        model_channels = 48  # fallback

    # Infer channel_mult from deepest layer
    max_ch = max(v.shape[0] for k, v in state_dict.items() if "conv_real.weight" in k and v.dim() == 4)
    n_mults = round(math.log2(max_ch / model_channels)) + 1
    channel_mult = tuple(2 ** i for i in range(n_mults))

    # Infer other params
    has_physics = any("helmholtz" in k for k in state_dict)
    cond_dim = 1  # wavelength only

    # Count res blocks per level from state dict keys
    num_res_blocks = 2  # safe default
    for k in state_dict:
        if "input_blocks" in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "input_blocks" and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        # rough estimate
                    except ValueError:
                        pass

    # Detect attention
    has_attention = any("qkv" in k for k in state_dict)
    attn_res = (8,) if has_attention else ()

    # Detect num_heads from attention QKV shape
    num_heads = 4
    for k, v in state_dict.items():
        if "middle_block" in k and "qkv.conv_real.weight" in k:
            qkv_out = v.shape[0]
            ch_at_attn = qkv_out // 3
            # common head dims: 32, 64
            for nh in [8, 4, 2, 1]:
                if ch_at_attn % nh == 0:
                    num_heads = nh
                    break
            break

    dx = float(stats.get("dx", 0.05))

    model = ComplexPhysicsUNet(
        in_channels=4,
        out_channels=2,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attn_res,
        dropout=0.0,
        num_heads=num_heads,
        cond_dim=cond_dim,
        dx=dx,
        dy=dx,
        omega=2.0 * math.pi / 1.55,
        enable_physics_features=has_physics,
        use_checkpoint=False,
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Set normalization stats if available
    if stats:
        model.set_normalization_stats(stats, normalize_eps=True)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model, stats


@torch.no_grad()
def predict_fields(
    model,
    eps_grid: np.ndarray,
    src_mask: np.ndarray,
    wavelength_um: float,
    stats: dict,
    num_steps: int = 50,
    device: str = "cuda",
):
    """Run flow-matching inference to predict EM fields.

    Args:
        model: loaded ComplexPhysicsUNet
        eps_grid: (H, W) permittivity map
        src_mask: (H, W) source mask
        wavelength_um: wavelength in microns
        stats: dataset statistics dict
        num_steps: number of ODE integration steps
        device: torch device

    Returns:
        Ez_real: (H, W) real part of Ez field
        Ez_imag: (H, W) imaginary part of Ez field
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Model"))
    from flow_matching import sample, SIG_MIN

    H, W = eps_grid.shape
    dtype = torch.float32

    # Normalize eps
    eps_mean = float(stats.get("eps_mean", 0.0))
    eps_std = float(stats.get("eps_std", 1.0))
    eps_norm = (eps_grid - eps_mean) / max(eps_std, 1e-8)

    # Build conditioning maps: [1, 2, H, W] = [eps_norm, src_mask]
    cond_maps = torch.tensor(
        np.stack([eps_norm, src_mask], axis=0)[None],
        dtype=dtype,
        device=device,
    )

    # Wavelength conditioning
    lam_mean = float(stats.get("lambda_um_mean", 1.55))
    lam_std = float(stats.get("lambda_um_std", 1.0))
    lam_norm = (wavelength_um - lam_mean) / max(lam_std, 1e-8)
    cond = torch.tensor([[lam_norm]], dtype=dtype, device=device)
    lambda_um = torch.tensor([[wavelength_um]], dtype=dtype, device=device)

    # Initial noise [1, 2, H, W]
    x_0 = torch.randn(1, 2, H, W, dtype=dtype, device=device)

    print(f"Running inference: {num_steps} steps, wavelength={wavelength_um:.3f} um, grid={H}x{W}")

    x_pred = sample(
        model,
        x_0,
        num_steps=num_steps,
        use_stoc_samp=False,
        cond_maps=cond_maps,
        cond=cond,
        lambda_um=lambda_um,
        phys_gate=1.0,
        sig_min=SIG_MIN,
    )

    # Denormalize fields
    ez_real_mean = float(stats.get("ez_real_mean", 0.0))
    ez_real_std = float(stats.get("ez_real_std", 1.0))
    ez_imag_mean = float(stats.get("ez_imag_mean", 0.0))
    ez_imag_std = float(stats.get("ez_imag_std", 1.0))

    Ez_real = x_pred[0, 0].cpu().numpy() * ez_real_std + ez_real_mean
    Ez_imag = x_pred[0, 1].cpu().numpy() * ez_imag_std + ez_imag_mean

    return Ez_real, Ez_imag


def save_results(
    output_path: str,
    Ez_real: np.ndarray,
    Ez_imag: np.ndarray,
    eps_grid: np.ndarray,
    src_mask: np.ndarray,
    extent: tuple,
    wavelength_um: float,
):
    """Save prediction results to .npz and generate visualization."""
    # Save data
    np.savez(
        output_path,
        Ez_real=Ez_real,
        Ez_imag=Ez_imag,
        eps=eps_grid,
        src_mask=src_mask,
        extent=np.array(extent),
        wavelength_um=wavelength_um,
    )
    print(f"Saved results to {output_path}")

    # Generate visualization
    fig_path = str(Path(output_path).with_suffix(".png"))
    Ez_mag = np.sqrt(Ez_real ** 2 + Ez_imag ** 2)
    Ez_phase = np.arctan2(Ez_imag, Ez_real)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x_min, x_max, y_min, y_max = extent
    ext = [x_min, x_max, y_min, y_max]

    ax = axes[0, 0]
    im = ax.imshow(eps_grid, extent=ext, origin="lower", cmap="gray_r", aspect="auto")
    ax.set_title("Permittivity")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    vmax = np.percentile(np.abs(Ez_real), 99)
    im = ax.imshow(Ez_real, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Ez (real)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(Ez_mag, extent=ext, origin="lower", cmap="hot", aspect="auto")
    ax.set_title("|Ez| (magnitude)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(Ez_phase, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="auto")
    ax.set_title("Phase(Ez)")
    plt.colorbar(im, ax=ax)

    for ax in axes.flat:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

    fig.suptitle(f"PHASE Prediction | wavelength = {wavelength_um:.3f} um", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_layer_map(s: str) -> dict:
    """Parse layer map from JSON string like '{"1/0": 12.25, "2/0": 2.07}'."""
    raw = json.loads(s)
    result = {}
    for k, v in raw.items():
        if "/" in k:
            layer, dt = k.split("/")
            result[(int(layer), int(dt))] = float(v)
        else:
            result[(int(k), 0)] = float(v)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="PHASE Field Predictor — predict EM fields from GDS layouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gds", required=True, help="Path to GDS file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--wavelength", type=float, required=True, help="Wavelength in microns (e.g. 1.55)")
    parser.add_argument("--output", default="prediction.npz", help="Output path (.npz)")

    # Source
    parser.add_argument("--source-port", choices=["left", "right", "top", "bottom"],
                        help="Source port location")
    parser.add_argument("--source-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="Source center coordinates in um")
    parser.add_argument("--source-width", type=float, default=0.5,
                        help="Source width in um (default: 0.5)")

    # Grid
    parser.add_argument("--dx", type=float, default=0.05, help="Grid spacing in um (default: 0.05)")
    parser.add_argument("--grid-height", type=int, default=160, help="Grid height in pixels (default: 160)")
    parser.add_argument("--grid-width", type=int, default=480, help="Grid width in pixels (default: 480)")

    # GDS
    parser.add_argument("--cell", default=None, help="GDS cell name (default: top-level cell)")
    parser.add_argument("--layer-map", default='{"1/0": 12.25}',
                        help='JSON mapping layer/datatype to epsilon (default: Si)')
    parser.add_argument("--eps-clad", type=float, default=2.09,
                        help="Cladding permittivity (default: 2.09 for SiO2)")

    # Inference
    parser.add_argument("--num-steps", type=int, default=50, help="ODE integration steps (default: 50)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device")

    args = parser.parse_args()

    if not Path(args.gds).exists():
        print(f"ERROR: GDS file not found: {args.gds}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # 1. Load GDS
    print(f"\n=== Loading GDS: {args.gds} ===")
    cell, polygons_by_layer = load_gds_polygons(args.gds, cell_name=args.cell)

    # 2. Rasterize to permittivity grid
    print("\n=== Rasterizing geometry ===")
    layer_map = parse_layer_map(args.layer_map)
    grid_shape = (args.grid_height, args.grid_width)
    eps_grid, extent = rasterize_polygons(
        polygons_by_layer, layer_map, args.eps_clad, args.dx, grid_shape
    )
    print(f"  Grid: {grid_shape[0]}x{grid_shape[1]}, dx={args.dx} um")
    print(f"  Extent: x=[{extent[0]:.2f}, {extent[1]:.2f}], y=[{extent[2]:.2f}, {extent[3]:.2f}] um")
    print(f"  Eps range: [{eps_grid.min():.2f}, {eps_grid.max():.2f}]")

    # 3. Create source mask
    src_mask = create_source_mask(
        grid_shape, extent,
        source_port=args.source_port,
        source_xy=args.source_xy,
        source_width_um=args.source_width,
        dx=args.dx,
    )
    print(f"  Source pixels: {int(src_mask.sum())}")

    # 4. Load model
    print(f"\n=== Loading model: {args.checkpoint} ===")
    model, stats = load_model(args.checkpoint, device=args.device)

    # 5. Run inference
    print("\n=== Running inference ===")
    Ez_real, Ez_imag = predict_fields(
        model, eps_grid, src_mask, args.wavelength, stats,
        num_steps=args.num_steps, device=args.device,
    )
    Ez_mag = np.sqrt(Ez_real ** 2 + Ez_imag ** 2)
    print(f"  |Ez| range: [{Ez_mag.min():.4e}, {Ez_mag.max():.4e}]")

    # 6. Save results
    print(f"\n=== Saving results ===")
    save_results(args.output, Ez_real, Ez_imag, eps_grid, src_mask, extent, args.wavelength)

    print("\nDone!")


if __name__ == "__main__":
    main()
