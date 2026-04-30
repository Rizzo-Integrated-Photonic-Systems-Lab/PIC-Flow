# Rayfield - Photonic Device Field Prediction

## Overview

Rayfield is a neural network-based electromagnetic field prediction system for photonic integrated circuit (PIC) devices. It uses **Physics-Embedded UNets** combined with **Flow Matching** (conditional generative modeling) and **complex-valued neural networks** to predict optical field distributions (Ez component - TE polarization).

**Primary Goal**: Predict complete electromagnetic field solutions for photonic devices given their geometry and wavelength, enabling fast inverse design and device optimization.

## Repository Structure

```
rayfield/
├── FDTD/                          # FDTD simulation & device definitions
│   ├── devices_base.py            # Base class for 2D photonic devices
│   ├── unified_sweep.py           # Multi-device dataset generation
│   ├── utils.py                   # Eigenmode tables, Meep utilities
│   ├── neff_tables/               # Pre-computed effective index tables
│   └── {device}/                  # Device-specific implementations
│       ├── device.py              # Device geometry class
│       └── sweep.py               # Parameter sweep for dataset
│
├── Model/                         # Training & inference
│   ├── train.py                   # Main training loop (DDP, multi-loss)
│   ├── dataset.py                 # Dataset loader with D4 augmentation
│   ├── dataset_fast.py            # Fast PyTorch tensor loader
│   ├── physics_unet.py            # Real-valued Physics UNet
│   ├── complex_physics_unet.py    # Complex-valued Physics UNet
│   ├── flow_matching.py           # Flow matching loss & physics losses
│   ├── sparams_loss.py            # S-parameter extraction & loss
│   ├── modal_sparams.py           # Modal decomposition for S-params
│   └── sparam_head.py             # Learned S-parameter prediction head
│
├── Data/                          # Dataset directory (FDTD results)
├── logs_physics_unet_pbfm/        # Training logs & checkpoints
└── wandb/                         # Experiment tracking
```

## Key Concepts

### Physics-Embedded Flow Matching

The model predicts electromagnetic fields using conditional flow matching:
- **Interpolation path**: `x_t = (1-(1-s)t)x₀ + t·x₁`
- **Velocity prediction**: Model learns `û_t(x_t, t)` to match target velocity
- **Physics constraint**: Helmholtz residual `R = ∇²E + k₀²εE` enforced via loss

### Complex-Valued Neural Networks

Native complex arithmetic preserves phase information:
- `ComplexConv2d`: Complex convolutions
- `ComplexGroupNorm`: Complex normalization
- `ModReLU`: Magnitude gating that preserves phase

### Multi-Loss Curriculum Training

Training proceeds in phases:
- **Phase A**: Flow matching only
- **Phase B**: Add physics residual loss
- **Phase C**: Enable phase, S-parameter, and auxiliary losses

## Supported Device Types

| Device | Ports | Description |
|--------|-------|-------------|
| Straight Waveguide | 2 | Baseline reference |
| Taper | 2 | Width transition / mode conversion |
| Euler Bend | 2 | Low-loss 90° bend with adiabatic curvature |
| Circular Bend | 2 | Constant-radius arc bend |
| S-Bend | 2 | Lateral offset with smooth transition |
| Y-Branch | 3 | 1→2 power splitter |
| Directional Coupler | 4 | 2×2 evanescent coupler |
| Waveguide Crossing | 4 | 90° intersection |

## Quick Start

### Training

```bash
# Single GPU
python Model/train.py --data-root Data/ --epochs 2000 --batch-size 8

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 Model/train.py --data-root Data/ --batch-size 32

# With complex UNet and all losses
python Model/train.py \
    --complex-unet \
    --lambda-residual 1.0 \
    --lambda-phase 0.1 \
    --lambda-sparam 0.1 \
    --phaseA-epochs 50 \
    --phaseB-epochs 200
```

### Dataset Generation

```bash
# Generate all device types
python FDTD/unified_sweep.py --output-dir Data/ --devices all

# Specific device
python FDTD/unified_sweep.py --output-dir Data/ --devices euler_bend,ybranch
```

## Data Format

Input tensors: `[C=4, H, W]`
- Channel 0-1: Real/imaginary parts of Ez field
- Channel 2: Permittivity εr (normalized to [0,1])
- Channel 3: Source mask

Auxiliary data includes:
- Port masks and IDs
- Ground truth S-parameters
- Wavelength (normalized)

## Important Files

| File | Purpose |
|------|---------|
| `Model/train.py` | Main training loop, loss computation, DDP |
| `Model/complex_physics_unet.py` | Complex-valued model architecture |
| `Model/flow_matching.py` | FM loss, physics residual, phase losses |
| `Model/dataset.py` | Data loading, phase anchoring, augmentation |
| `FDTD/unified_sweep.py` | Multi-device FDTD data generation |
| `FDTD/devices_base.py` | Base class for all device types |

## Dependencies

- PyTorch (with DDP, AMP support)
- Meep (FDTD simulations)
- wandb (experiment tracking)
- numpy, scipy, tqdm

See `requirements.txt` for full list.

## Branch Information

- **main**: Stable release branch
- **updatedComplexJQ**: Current development branch with complex JQ enhancements
