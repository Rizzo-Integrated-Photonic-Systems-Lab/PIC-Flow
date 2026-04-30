# Model Training & Loss Functions

## Overview

This directory contains the neural network architectures, training loops, and loss functions for the Rayfield photonic field prediction system. The core approach combines **conditional flow matching** with **physics-embedded constraints** to predict electromagnetic fields.

## Architecture

### Two UNet Variants

#### 1. Real-Valued Physics UNet (`physics_unet.py`)

Standard U-Net with physics embedding:
- **Input**: `[B, C, H, W]` with C ∈ {4, 4+K} (K = SDF channels)
- **Timestep conditioning**: Sinusoidal embedding + MLP
- **Blocks**: Residual blocks with timestep modulation (AdaGN)
- **Attention**: Optional multi-head self-attention at specified resolutions
- **Physics features**: Helmholtz residual computation embedded in forward pass

```
Input → Encoder → Bottleneck → Decoder → Output
         ↓                        ↑
      Skip connections ──────────┘
```

#### 2. Complex-Valued Physics UNet (`complex_physics_unet.py`)

Native complex arithmetic for phase-aware predictions:

**Complex Layers**:
- `ComplexConv2d`: `(Wr + i·Wi) × (xr + i·xi)` with proper complex multiplication
- `ComplexGroupNorm`: Normalizes real and imaginary parts jointly
- `ModReLU`: `f(z) = ReLU(|z| + b) × (z/|z|)` - gates magnitude, preserves phase
- `ComplexAttention`: Q, K, V all in complex domain

**Phase Features** (`PhaseFeatures2D`):
- Unit phasors: `u_r = Re(E)/|E|`, `u_i = Im(E)/|E|`
- Log-magnitude: `log(|E| + ε)`
- Phase gradients: `∂φ/∂x`, `∂φ/∂y` (local propagation direction)
- Coarse phase: Low-frequency phase envelope (pooled)
- Relative phase: Phase offset from source region

---

## Loss Functions

### Flow Matching Loss (`flow_matching.py`)

**Core FM Loss**:
```python
# Interpolation: x_t = (1 - (1-σ)t)·x₀ + t·x₁
# Target velocity: u_t = x₁ - (1-σ)·x₀
# Loss: ||model(x_t, t) - u_t||²

loss_fm = F.mse_loss(v_pred, v_target)
```

**Time Sampling**:
- 50% uniform: `t ~ U[0, 1]`
- 50% Beta: `t ~ Beta(2, 2)` (concentrates near t=0.5)

### Physics Residual Loss

**Helmholtz Equation**:
```python
# R = ∇²E + k₀²·ε·E = 0 (should vanish for valid fields)
# Loss penalizes non-zero residual

class HelmholtzResidual2D:
    def forward(self, E, epsilon, wavelength):
        laplacian = self.compute_laplacian(E)  # 5-point stencil
        k0 = 2 * pi / wavelength
        residual = laplacian + k0**2 * epsilon * E
        return residual

loss_residual = (residual.abs()**2).mean()
```

**PML Masking**: Residual computed only in interior (exclude absorbing boundaries)

### Phase Loss

**Direct Phase Comparison**:
```python
# Phase difference with wraparound handling
phase_pred = torch.atan2(E_pred.imag, E_pred.real)
phase_true = torch.atan2(E_true.imag, E_true.real)

# Weighted by amplitude (ignore low-signal regions)
weights = E_true.abs() / E_true.abs().max()
loss_phase = weighted_mse(phase_pred, phase_true, weights)
```

**Phase Gradient Loss**:
```python
# Penalize incorrect propagation direction
grad_phi_pred = spatial_gradient(phase_pred)
grad_phi_true = spatial_gradient(phase_true)
loss_phase_grad = mse(grad_phi_pred, grad_phi_true)
```

### S-Parameter Loss (`sparams_loss.py`)

**Port Projection Method**:
```python
# Extract field at each port via mask projection
a_p = sum(mask_p * E) / sum(mask_p)  # Port amplitude

# S-parameter = output / input
S_pred = a_out / a_in

# Loss on magnitude and phase
loss_mag = |S_pred.abs() - S_true.abs()|²
loss_phase = phase_diff(S_pred.angle(), S_true.angle())²
loss_sparam = loss_mag + λ · loss_phase
```

**Modal Decomposition** (`modal_sparams.py`):
- Projects field onto waveguide eigenmodes
- More accurate for multimode waveguides
- Requires pre-computed mode profiles

**Learned Head** (`sparam_head.py`):
- CNN head attached to UNet features
- Directly regresses S-parameters
- Faster than projection methods

### Endpoint Loss

```python
# Penalize error at t=1 (final field prediction)
x_1_pred = model.sample(x_0, steps=fm_steps)
loss_endpoint = mse(x_1_pred, x_1_true)
```

---

## Training Configuration

### Curriculum Learning (Phases)

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| A | 0 → phaseA_epochs | FM only |
| B | phaseA → phaseB | FM + Residual |
| C | phaseB → end | FM + Residual + Phase + S-param |

### Loss Weights

| Loss | Flag | Default | Warmup |
|------|------|---------|--------|
| Flow Matching | (always on) | 1.0 | - |
| Residual | `--lambda-residual` | 1.0 | 50 epochs |
| Phase | `--lambda-phase` | 0.1 | 50 epochs |
| Phase Gradient | `--lambda-phase-grad` | 0.0 | 100 epochs |
| Endpoint | `--lambda-endpoint` | 0.0 | - |
| S-Parameter | `--lambda-sparam` | 0.0 | 50 epochs |

### Key Training Arguments

```bash
# Model
--hidden-size 128          # Base channel count
--complex-unet             # Use complex-valued UNet
--physics-features         # Include Helmholtz embedding
--no-attention             # Disable attention layers

# Training
--epochs 2000
--batch-size 8             # Global (divided for DDP)
--lr 1e-4
--warmup-epochs 2
--min-lr 5e-6

# Curriculum
--phaseA-epochs 50
--phaseB-epochs 200

# Losses
--lambda-residual 1.0
--lambda-phase 0.1
--lambda-sparam 0.1
--residual-warmup-epochs 50
--sparam-warmup-epochs 50

# S-Parameters
--sparam-mode modal        # "modal", "project", or "head"
--sparam-every 1           # Compute every N batches

# Optimization
--amp                      # Mixed precision (FP16 forward, FP32 loss)
--use-checkpoint           # Gradient checkpointing
--config                   # ConFIG gradient surgery
```

---

## Gradient Surgery (ConFIG)

When `--config` is enabled, conflicting gradients between losses are resolved:

```python
# ConFIG: Conflict-Free Gradient optimization
# Ensures multi-task gradients don't cancel each other

from conflictfree import ConFIG

optimizer = ConFIG(
    losses=[loss_fm, loss_residual, loss_phase],
    model=model
)
optimizer.step()
```

Activates after `--config-start-epoch` (default: 200)

---

## Dataset Loading

### Standard Dataset (`dataset.py`)

```python
class FDTDDataset:
    def __getitem__(self, idx):
        # Load from NPZ or folder
        field = load_field(idx)  # [2, H, W] real/imag
        epsilon = load_epsilon(idx)  # [1, H, W]
        source = load_source(idx)  # [1, H, W]

        # Phase anchoring (robust global reference)
        field = anchor_phase(field, source, epsilon)

        # D4 augmentation (8 symmetries)
        if self.augment:
            field, epsilon, source = apply_d4(field, epsilon, source)

        # Normalize
        field = (field - self.field_mean) / self.field_std
        epsilon = epsilon / self.eps_max

        return {
            'x': torch.cat([field, epsilon, source], dim=0),
            'wavelength': wl_normalized,
            'aux': {'port_masks': ..., 'sparams_true': ...}
        }
```

**Phase Anchoring Strategies**:
1. Source mask in high-ε region
2. Input waveguide ROI `[40:140, 2:H-2]`
3. Central window fallback

### Fast Dataset (`dataset_fast.py`)

Pre-processed PyTorch tensors (10-20x faster):
```python
class FastFDTDDataset:
    def __getitem__(self, idx):
        # Direct tensor load, no processing
        return torch.load(f"sample_{idx}.pt")
```

---

## Training Loop (`train.py`)

### Main Flow

```python
def train_epoch(model, loader, optimizer, epoch, args):
    for batch in loader:
        x = batch['x']  # [B, 4, H, W]
        wl = batch['wavelength']
        aux = batch['aux']

        # Sample time and create interpolated state
        t = sample_time(B)  # Mixed uniform + Beta
        x_0 = torch.randn_like(x[:, :2])  # Noise
        x_1 = x[:, :2]  # Target field
        x_t = interpolate(x_0, x_1, t)

        # Condition on geometry
        cond = x[:, 2:]  # epsilon, source

        # Forward pass
        with autocast(enabled=args.amp):
            v_pred = model(x_t, t, cond, wl)

        # Losses (FP32)
        loss_fm = fm_loss(v_pred, x_0, x_1)
        loss_res = residual_loss(v_pred, cond, wl) if phase >= 'B'
        loss_phase = phase_loss(v_pred, x_1) if phase >= 'C'
        loss_sparam = sparam_loss(v_pred, aux) if phase >= 'C'

        # Weighted sum
        loss = loss_fm + λ_res*loss_res + λ_phase*loss_phase + λ_sparam*loss_sparam

        # Backward + optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        ema.update(model)
```

### DDP Setup

```python
# Distributed Data Parallel
torch.distributed.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
sampler = DistributedSampler(dataset)
```

### Checkpointing

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'ema_state_dict': ema.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
}, f"checkpoint_{epoch}.pt")

# Load
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Sampling / Inference

```python
def sample(model, cond, wavelength, steps=20):
    """Generate field prediction via reverse-time ODE"""
    x = torch.randn(B, 2, H, W)  # Start from noise

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt)
        v = model(x, t, cond, wavelength)
        x = x + v * dt  # Euler step

    return x  # Predicted field
```

For production: Use adaptive ODE solvers (dopri5) for better accuracy.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `loss_fm` | Flow matching velocity MSE |
| `loss_residual` | Helmholtz residual L2 |
| `loss_phase` | Phase MSE (amplitude-weighted) |
| `loss_sparam` | S-parameter error |
| `psnr` | Peak signal-to-noise ratio |
| `ssim` | Structural similarity |
| `sparam_mag_error` | Mean S-param magnitude error |
| `sparam_phase_error` | Mean S-param phase error |

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | ~1600 | Main training loop, DDP, logging |
| `complex_physics_unet.py` | ~900 | Complex-valued architecture |
| `physics_unet.py` | ~600 | Real-valued architecture |
| `flow_matching.py` | ~600 | FM loss, physics losses |
| `dataset.py` | ~800 | Data loading, augmentation |
| `dataset_fast.py` | ~200 | Fast tensor loading |
| `sparams_loss.py` | ~150 | S-parameter extraction |
| `modal_sparams.py` | ~300 | Modal decomposition |
| `sparam_head.py` | ~100 | Learned S-param head |
| `grad_utils.py` | ~100 | Gradient utilities |
