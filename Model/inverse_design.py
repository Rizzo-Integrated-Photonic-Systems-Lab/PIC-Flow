# inverse_design.py
"""
Standalone inverse design inference script.

Given target S-parameters and wavelength, generate a device geometry (eps)
and predicted fields using the jointly-trained flow matching model.

Usage:
    python inverse_design.py \
        --checkpoint logs/.../0000700.pt \
        --target-sparams "0.0,0.707+0.0j,0.707+0.0j,0.0" \
        --wavelength 1.55 \
        --src-mask src_mask.npy \
        --cfg-scale 3.0 \
        --num-samples 8 \
        --fm-steps 50
"""

import argparse
import math
import os

import numpy as np
import torch

from physics_unet import PhysicsUNet, HelmholtzResidual2D
from complex_physics_unet import ComplexPhysicsUNet
from flow_matching import sample_inverse, sample_joint, SIG_MIN


def parse_sparams(s: str) -> np.ndarray:
    """Parse S-parameter string like '0.0,0.707+0.0j,0.707+0.0j,0.0' into complex array."""
    parts = [p.strip() for p in s.split(",")]
    return np.array([complex(p) for p in parts], dtype=np.complex64)


def build_cond_vector(
    wavelength_um: float,
    target_sparams: np.ndarray,
    stats: dict,
    max_ports: int = 4,
) -> torch.Tensor:
    """Build the 16-dim conditioning vector for inverse design."""
    # Normalize wavelength
    lam_mean = float(stats.get("lambda_um_mean", 1.55))
    lam_std = float(stats.get("lambda_um_std", 1.0))
    if lam_std <= 0:
        lam_std = 1.0
    lam_norm = (wavelength_um - lam_mean) / lam_std

    # Geom params: set to zero (unknown for inverse design)
    cond = [lam_norm, 0.0, 0.0, 0.0]  # wavelength + 3 geom params

    # S-param Re/Im (8 entries for 4 ports)
    n_s = min(len(target_sparams), max_ports)
    sparam_entries = [0.0] * (2 * max_ports)
    for i in range(n_s):
        sparam_entries[2 * i] = float(np.real(target_sparams[i]))
        sparam_entries[2 * i + 1] = float(np.imag(target_sparams[i]))
    cond.extend(sparam_entries)

    # Port valid flags (4 entries)
    port_valid = [0.0] * max_ports
    for i in range(n_s):
        port_valid[i] = 1.0
    cond.extend(port_valid)

    return torch.tensor(cond, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Inverse design via joint flow matching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--target-sparams", type=str, required=True,
                        help="Target S-params, comma-separated complex (e.g. '0.0,0.707+0.0j,0.707+0.0j,0.0')")
    parser.add_argument("--wavelength", type=float, default=1.55, help="Wavelength in um")
    parser.add_argument("--src-mask", type=str, default=None, help="Path to source mask .npy file")
    parser.add_argument("--grid-h", type=int, default=128, help="Grid height (if no src-mask)")
    parser.add_argument("--grid-w", type=int, default=128, help="Grid width (if no src-mask)")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--fm-steps", type=int, default=50, help="Number of ODE integration steps")
    parser.add_argument("--output-dir", type=str, default="inverse_design_results", help="Output directory")
    parser.add_argument("--complex-unet", action="store_true", help="Use complex UNet architecture")
    parser.add_argument("--binarize-threshold", type=float, default=0.5,
                        help="Threshold for binarizing generated eps (0-1 normalized)")
    parser.add_argument("--eps-core", type=float, default=12.25, help="Core permittivity")
    parser.add_argument("--eps-clad", type=float, default=2.07, help="Cladding permittivity")
    parser.add_argument("--dx", type=float, default=1.0 / 24.0, help="Grid spacing (um)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args", None)

    # Determine model params from checkpoint (prefer ckpt_args, fall back to defaults)
    in_channels = int(stats.get("x_channels", 4))
    cond_dim = int(stats.get("cond_dim", 16))
    hidden_size = int(getattr(ckpt_args, "hidden_size", 128)) if ckpt_args else 128
    dropout = float(getattr(ckpt_args, "dropout", 0.0)) if ckpt_args else 0.0
    dx = float(getattr(ckpt_args, "dx", args.dx)) if ckpt_args else args.dx
    lam0 = float(getattr(ckpt_args, "lambda_um", 1.55)) if ckpt_args else 1.55
    omega = 2.0 * math.pi / lam0
    pml_cells = int(getattr(ckpt_args, "pml_cells", 0)) if ckpt_args else 0
    enable_physics = bool(getattr(ckpt_args, "physics_features", True)) if ckpt_args else True
    use_complex = bool(getattr(ckpt_args, "complex_unet", False)) if ckpt_args else args.complex_unet

    model_kwargs = dict(
        in_channels=in_channels,
        out_channels=3,  # joint training
        model_channels=hidden_size,
        num_res_blocks=4,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(8,),
        dropout=dropout,
        dims=2,
        num_heads=4,
        cond_dim=cond_dim,
        dx=dx,
        dy=dx,
        omega=omega,
        pml_cells=pml_cells,
        enable_physics_features=enable_physics,
    )

    if use_complex:
        model = ComplexPhysicsUNet(**model_kwargs).to(device)
    else:
        model = PhysicsUNet(**model_kwargs).to(device)

    # Load EMA weights; warn on mismatched keys
    missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
    if missing:
        print(f"WARNING: Missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    normalize_eps = bool(getattr(ckpt_args, "normalize_eps", True)) if ckpt_args else True
    model.set_normalization_stats(stats, normalize_eps=normalize_eps)
    model.eval()

    # Parse target S-params
    target_sparams = parse_sparams(args.target_sparams)
    print(f"Target S-params: {target_sparams}")
    print(f"Wavelength: {args.wavelength} um")
    print(f"Model: {'ComplexPhysicsUNet' if use_complex else 'PhysicsUNet'}, "
          f"channels={hidden_size}, physics={enable_physics}")

    # Build source mask
    if args.src_mask is not None:
        src_mask_np = np.load(args.src_mask).astype(np.float32)
        H, W = src_mask_np.shape[-2], src_mask_np.shape[-1]
        src_mask = torch.from_numpy(src_mask_np).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    else:
        H, W = args.grid_h, args.grid_w
        src_mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    # Build conditioning vector
    cond_vec = build_cond_vector(args.wavelength, target_sparams, stats).to(device)

    # Expand for batch
    B = args.num_samples
    src_mask_batch = src_mask.expand(B, -1, -1, -1)
    cond_batch = cond_vec.unsqueeze(0).expand(B, -1)

    # Wavelength for model
    lambda_um = torch.full((B, 1), args.wavelength, device=device, dtype=torch.float32)

    print(f"Generating {B} samples with CFG scale={args.cfg_scale}, steps={args.fm_steps}...")

    with torch.no_grad():
        results = sample_inverse(
            model,
            num_steps=args.fm_steps,
            src_mask=src_mask_batch,
            cond=cond_batch,
            lambda_um=lambda_um,
            cfg_scale=args.cfg_scale,
            sig_min=SIG_MIN,
            base_cond_dim=4,  # wavelength + 3 geom params
        )  # [B, 3, H, W]

    # Post-process
    fields_pred = results[:, :2]  # [B, 2, H, W] normalized
    eps_pred = results[:, 2:3]    # [B, 1, H, W] normalized

    # De-normalize
    ez_real_std = float(stats.get("ez_real_std", 1.0))
    ez_real_mean = float(stats.get("ez_real_mean", 0.0))
    ez_imag_std = float(stats.get("ez_imag_std", 1.0))
    ez_imag_mean = float(stats.get("ez_imag_mean", 0.0))
    eps_mean = float(stats.get("eps_mean", 0.0))
    eps_std = float(stats.get("eps_std", 1.0))

    fields_phys = torch.stack([
        fields_pred[:, 0] * ez_real_std + ez_real_mean,
        fields_pred[:, 1] * ez_imag_std + ez_imag_mean,
    ], dim=1)  # [B, 2, H, W]

    eps_phys = eps_pred * eps_std + eps_mean  # [B, 1, H, W]

    # Binarize eps
    eps_norm = ((eps_phys - args.eps_clad) / (args.eps_core - args.eps_clad + 1e-8)).clamp(0, 1)
    eps_binary = torch.where(eps_norm > args.binarize_threshold,
                             torch.tensor(args.eps_core, device=device),
                             torch.tensor(args.eps_clad, device=device))

    # Save results
    for i in range(B):
        sample_dir = os.path.join(args.output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "eps_continuous.npy"),
                eps_phys[i, 0].cpu().numpy())
        np.save(os.path.join(sample_dir, "eps_binary.npy"),
                eps_binary[i, 0].cpu().numpy())
        np.save(os.path.join(sample_dir, "Ez_real.npy"),
                fields_phys[i, 0].cpu().numpy())
        np.save(os.path.join(sample_dir, "Ez_imag.npy"),
                fields_phys[i, 1].cpu().numpy())

        print(f"  Sample {i}: eps range [{eps_phys[i].min():.2f}, {eps_phys[i].max():.2f}], "
              f"|Ez| max={torch.sqrt(fields_phys[i, 0]**2 + fields_phys[i, 1]**2).max():.4f}")

    # Save metadata
    meta = {
        "target_sparams_real": np.real(target_sparams).tolist(),
        "target_sparams_imag": np.imag(target_sparams).tolist(),
        "wavelength_um": args.wavelength,
        "cfg_scale": args.cfg_scale,
        "fm_steps": args.fm_steps,
        "num_samples": B,
        "eps_core": args.eps_core,
        "eps_clad": args.eps_clad,
        "binarize_threshold": args.binarize_threshold,
    }
    import json
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
