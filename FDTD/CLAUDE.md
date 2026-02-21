# FDTD Dataset - Photonic Device Documentation

## Overview

This directory contains FDTD (Finite-Difference Time-Domain) simulation code for generating electromagnetic field datasets across 8 photonic device types. All simulations use Meep for 2D TE-polarized fields (Ez component) in silicon-on-insulator waveguides.

## Material System

- **Core**: Silicon (Si), n ≈ 3.48 at 1550nm
- **Cladding**: Silicon dioxide (SiO₂), n ≈ 1.44
- **Wavelength Range**: 1500-1600nm (C-band telecommunications)
- **Polarization**: TE (Ez field component)

## Dataset Structure

Each simulation produces:
```
{device_type}_{params}/
├── field_real.npy       # Re(Ez) field [H, W]
├── field_imag.npy       # Im(Ez) field [H, W]
├── epsilon.npy          # Permittivity map [H, W]
├── source_mask.npy      # Source location [H, W]
├── port_masks.npy       # Per-port masks [P, H, W]
├── sparams.npy          # S-parameters [P] complex
└── metadata.json        # Device params, wavelength, port info
```

---

## Device Types

### 1. Straight Waveguide (`straight_waveguide/`)

**Class**: `StraightWaveguide2D`

**Description**: A simple rectangular waveguide of constant width. Serves as the baseline reference for transmission measurements and loss characterization. The fundamental mode propagates without any geometric perturbation.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `width` | 0.4-0.6 μm | Waveguide core width |
| `length` | 5-20 μm | Propagation length |

**Ports**: 2 (input left, output right)

**Physics**: Single-mode propagation with effective index determined by width. Used to calibrate simulation accuracy and measure propagation loss.

**S-Parameters**:
- S21: Transmission (ideally |S21| ≈ 1)
- S11: Reflection (ideally |S11| ≈ 0)

---

### 2. Taper (`taper/`)

**Class**: `TaperWaveguide2D`

**Description**: A linearly tapered waveguide that transitions between two different widths. Used for mode conversion between waveguides of different dimensions, coupling to fibers, or matching to other components.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `in_width` | 0.4-0.6 μm | Input waveguide width |
| `out_width` | 0.8-3.0 μm | Output waveguide width |
| `length` | 5-50 μm | Taper length |

**Ports**: 2 (narrow end, wide end)

**Physics**: Adiabatic mode conversion when taper is sufficiently long. Shorter tapers cause mode mismatch and scattering. The taper angle determines conversion efficiency.

**Design Considerations**:
- Longer tapers → more adiabatic, higher transmission
- Shorter tapers → compact but lossy
- Typical efficiency: >95% for well-designed tapers

**S-Parameters**:
- S21: Mode conversion efficiency
- S11: Back-reflection (increases with steeper tapers)

---

### 3. Euler Bend (`euler_bend/`)

**Class**: `EulerBend2D`

**Description**: A 90° waveguide bend using Euler spiral (clothoid) geometry. The curvature increases linearly from zero at the input to maximum at the center, then decreases back to zero. This adiabatic curvature transition minimizes mode mismatch and radiation loss.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `width` | 0.4-0.6 μm | Waveguide width |
| `radius` | 2-10 μm | Effective bend radius |
| `angle` | 90° | Bend angle (fixed) |

**Ports**: 2 (horizontal input, vertical output)

**Physics**: The Euler spiral parametrization:
```
κ(s) = s/R²  (curvature as function of arc length)
```
This provides zero curvature at interfaces, eliminating abrupt mode transitions that cause loss in circular bends.

**Advantages over Circular Bends**:
- 50-70% lower radiation loss at same footprint
- Reduced mode mismatch at straight-to-bend interface
- Better performance at smaller radii

**S-Parameters**:
- S21: Bend transmission (typically |S21| > 0.95 for R > 5μm)
- S11: Reflection from mode mismatch

---

### 4. Circular Bend (`circular_bend/`)

**Class**: `CircularBend2D`

**Description**: A waveguide bend following a circular arc with constant radius. Simpler geometry than Euler bends but with higher loss due to abrupt curvature changes at interfaces.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `width` | 0.4-0.6 μm | Waveguide width |
| `radius` | 2-20 μm | Bend radius |
| `angle` | 45-180° | Bend angle |

**Ports**: 2 (input tangent, output tangent)

**Physics**: Constant curvature causes:
1. **Mode mismatch** at straight-to-bend interfaces
2. **Radiation loss** from centrifugal effect pushing mode outward
3. **Whispering gallery modes** at very tight radii

**Loss Mechanisms**:
- Radiation loss ∝ exp(-αR) where α depends on index contrast
- Mode mismatch loss at each interface
- Total loss increases rapidly below critical radius (~3μm for Si)

**S-Parameters**:
- S21: Bend transmission
- S11: Interface reflections

---

### 5. S-Bend (`sbend/`)

**Class**: `EulerSBend2D`

**Description**: An S-shaped lateral offset waveguide that displaces the optical path vertically while maintaining horizontal propagation direction. Uses paired Euler curves for smooth transitions.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `width` | 0.4-0.6 μm | Waveguide width |
| `length` | 10-50 μm | Horizontal extent |
| `offset` | 1-10 μm | Vertical displacement |

**Ports**: 2 (input left, output right offset)

**Physics**: Two concatenated Euler bends of opposite sign create the S-shape. The curvature profile:
```
κ(s) = +s/R² for first half
κ(s) = -s/R² for second half
```

**Applications**:
- Waveguide routing around obstacles
- Port pitch matching between components
- Avoiding crossings in dense layouts

**S-Parameters**:
- S21: Transmission through S-curve
- S11: Reflections (minimal for gradual bends)

---

### 6. Y-Branch (`ybranch/`)

**Class**: `YBranch2D`

**Description**: A 1×2 power splitter that divides input light equally between two output waveguides. The junction uses a symmetric Y-shape with tapered transition region to minimize excess loss.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `width` | 0.4-0.6 μm | Waveguide width |
| `length` | 5-20 μm | Junction length |
| `split_angle` | 5-30° | Half-angle of split |

**Ports**: 3
- Port 1: Input (single waveguide)
- Port 2: Upper output
- Port 3: Lower output

**Physics**: At the junction, the input mode couples to the even supermode of the two output waveguides. Ideally:
- 50% power to each output (3dB split)
- No reflection back to input
- Symmetric splitting

**Loss Sources**:
- Mode mismatch at junction → excess loss
- Asymmetry → unequal splitting
- Sharp angles → radiation loss

**S-Parameters**:
- S21, S31: Output powers (ideally |S21|² = |S31|² = 0.5)
- S11: Back-reflection
- Excess loss = 1 - |S21|² - |S31|²

---

### 7. Directional Coupler (`directional_coupler/`)

**Class**: `DirectionalCoupler2D`

**Description**: A 2×2 coupler where two parallel waveguides exchange power via evanescent field overlap. The coupling ratio depends on gap size, interaction length, and wavelength.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `wg_width` | 0.4-0.6 μm | Waveguide width |
| `gap` | 0.1-0.5 μm | Separation between cores |
| `length` | 5-50 μm | Coupling region length |

**Ports**: 4
- Port 1: Upper input (bar)
- Port 2: Lower input
- Port 3: Upper output (bar)
- Port 4: Lower output (cross)

**Physics**: Coupled-mode theory describes power exchange:
```
P_cross(L) = sin²(κL)
P_bar(L) = cos²(κL)
```
where κ is the coupling coefficient (depends on gap, wavelength).

**Key Lengths**:
- L = π/4κ → 50:50 coupler (3dB)
- L = π/2κ → 100% crossover
- L = π/κ → 100% bar (complete cycle)

**Applications**:
- Power splitters/combiners
- Mach-Zehnder interferometer arms
- Ring resonator coupling
- Wavelength filters

**S-Parameters** (4×4 matrix):
- S31: Bar transmission (through)
- S41: Cross transmission (coupled)
- S11, S21: Reflections (ideally zero)
- Unitarity: |S31|² + |S41|² ≈ 1

---

### 8. Waveguide Crossing (`crossing/`)

**Class**: `UniformCrossing2D`

**Description**: A 90° intersection where two waveguides cross without intentional coupling. The goal is maximum transmission with minimum crosstalk between the two paths.

**Parameters**:
| Parameter | Range | Description |
|-----------|-------|-------------|
| `wg_width_h` | 0.4-0.6 μm | Horizontal waveguide width |
| `wg_width_v` | 0.4-0.6 μm | Vertical waveguide width |

**Ports**: 4
- Port 1: Left (horizontal input)
- Port 2: Right (horizontal output)
- Port 3: Bottom (vertical input)
- Port 4: Top (vertical output)

**Physics**: At the intersection, modes from orthogonal waveguides overlap. Design goals:
1. High transmission: |S21|², |S43|² → 1
2. Low crosstalk: |S31|², |S41|² → 0
3. Low reflection: |S11|² → 0

**Optimization Strategies**:
- Expand mode at crossing (reduces diffraction)
- Use multimode interference (MMI) region
- Elliptical or tapered intersection shapes

**Typical Performance**:
- Insertion loss: 0.1-0.5 dB
- Crosstalk: -30 to -40 dB
- Reflection: -20 to -30 dB

**S-Parameters**:
- S21: Horizontal transmission
- S43: Vertical transmission
- S31, S41: Crosstalk
- S11: Reflection

---

## Simulation Parameters

### Grid Resolution
- Default: 20 pixels/μm (50nm resolution)
- PML layers: 1.0 μm thickness on all boundaries

### Source Configuration
- Type: Eigenmode source (fundamental TE mode)
- Position: 1 μm from input port boundary
- Bandwidth: 50nm for broadband, CW for single-frequency

### Convergence Criteria
- Field decay: 1e-6 relative to peak
- Maximum runtime: 1000 periods

## File Descriptions

| File | Description |
|------|-------------|
| `devices_base.py` | Abstract base class `Device2DBase` with common methods |
| `unified_sweep.py` | Multi-device parameter sweep and batch generation |
| `utils.py` | Eigenmode calculation, neff lookup, Meep helpers |
| `{device}/device.py` | Device geometry implementation |
| `{device}/sweep.py` | Parameter ranges and sweep configuration |

## Usage

### Generate Single Device
```python
from FDTD.euler_bend.device import EulerBend2D

device = EulerBend2D(width=0.5, radius=5.0)
device.run_simulation(wavelength=1.55)
device.save_fields("output/")
```

### Batch Generation
```bash
python FDTD/unified_sweep.py \
    --output-dir Data/ \
    --devices euler_bend,ybranch,directional_coupler \
    --num-samples 1000
```

## Port Mask Convention

Port masks are binary arrays indicating modal overlap regions:
- Masks extend 0.5 μm into waveguide from boundary
- Used for S-parameter extraction via field projection
- Stored as `[num_ports, H, W]` array

## Wavelength Normalization

Wavelengths are stored in μm and normalized for training:
```python
wl_normalized = (wavelength - wl_mean) / wl_std
# Typical: wl_mean ≈ 1.55, wl_std ≈ 0.03
```
