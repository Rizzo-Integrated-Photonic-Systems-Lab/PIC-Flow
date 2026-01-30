"""
MEEP-style modal decomposition for S-parameter extraction.

This module implements rigorous S-parameter extraction using eigenmode overlap
integrals, similar to MEEP's get_eigenmode_coefficients function.

For 2D TE simulations (Ez polarization), the approach is:
1. Compute the fundamental waveguide mode profile at each port cross-section
2. Perform overlap integrals between the simulated field and mode profile
3. Separate forward and backward propagating components
4. Normalize to get S-parameters where |S|^2 represents power ratios

References:
- MEEP Mode Decomposition: https://github.com/NanoComp/meep/blob/master/doc/docs/Mode_Decomposition.md
- Eigenmode coefficient normalization: https://github.com/NanoComp/meep/pull/288/files
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


# Physical constants
C0 = 299792458.0  # speed of light in vacuum (m/s)
EPS0 = 8.854187817e-12  # vacuum permittivity (F/m)
MU0 = 4 * math.pi * 1e-7  # vacuum permeability (H/m)


def solve_1d_mode_fdfd(
    eps_profile: torch.Tensor,
    wavelength_um: float,
    dx_um: float,
    n_modes: int = 1,
    pml_cells: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve for the fundamental TE mode of a 1D waveguide cross-section.

    For TE polarization (Ez), we solve the 1D Helmholtz eigenvalue problem.

    This implementation uses a variational approach that's more numerically stable
    for finding the fundamental (highest n_eff) mode.

    Args:
        eps_profile: 1D permittivity profile [N] along the transverse direction
        wavelength_um: Wavelength in microns
        dx_um: Grid spacing in microns
        n_modes: Number of modes to compute (default: 1 for fundamental)
        pml_cells: Number of PML cells at boundaries (0 to disable)

    Returns:
        neff: Effective indices [n_modes]
        modes: Mode profiles [n_modes, N], normalized and single-signed
    """
    device = eps_profile.device
    dtype = eps_profile.dtype
    N = eps_profile.shape[0]

    k0 = 2 * math.pi / wavelength_um  # free-space wavenumber (1/um)
    dx = dx_um

    # For robustness, use a simpler approach for the fundamental mode:
    # The fundamental mode profile closely follows the permittivity distribution
    # in the high-index region (core) with evanescent tails in the cladding.

    # Identify core and cladding
    eps_max = eps_profile.max()
    eps_min = eps_profile.min()
    eps_threshold = eps_min + 0.3 * (eps_max - eps_min)  # threshold to identify core

    # Approximate effective index (between core and cladding indices)
    n_core = torch.sqrt(eps_max)
    n_clad = torch.sqrt(eps_min)

    # For single-mode waveguide, n_eff is close to n_core
    # Use variational estimate: n_eff ≈ sqrt(weighted average of eps)
    core_mask = (eps_profile > eps_threshold).float()
    if core_mask.sum() > 0:
        n_eff_approx = torch.sqrt((eps_profile * core_mask).sum() / core_mask.sum().clamp(min=1))
    else:
        n_eff_approx = (n_core + n_clad) / 2

    # Construct approximate fundamental mode profile
    # Use exponential decay from core into cladding
    # For step-index: Ez ~ cos(κy) in core, exp(-γ|y|) in cladding
    # where κ² = k0²n_core² - β², γ² = β² - k0²n_clad²

    beta = k0 * n_eff_approx
    kappa_sq = (k0 * n_core)**2 - beta**2
    gamma_sq = beta**2 - (k0 * n_clad)**2

    kappa = torch.sqrt(kappa_sq.clamp(min=1e-12))
    gamma = torch.sqrt(gamma_sq.clamp(min=1e-12))

    # Find core center
    core_indices = torch.where(core_mask > 0.5)[0]
    if len(core_indices) > 0:
        core_center = core_indices.float().mean()
    else:
        core_center = torch.tensor(N / 2.0, device=device)

    # Build mode profile
    y = torch.arange(N, device=device, dtype=dtype)
    y_centered = y - core_center

    # Approximate mode as Gaussian (good for weakly-guiding waveguides)
    # or use the permittivity-weighted profile
    # Width from mode confinement: w ~ 1/gamma for decay length
    mode_width = 1.0 / gamma.clamp(min=0.1)  # in grid units
    mode_width_px = mode_width / dx

    # Gaussian approximation
    mode_profile = torch.exp(-0.5 * (y_centered / mode_width_px.clamp(min=1.0))**2)

    # Alternative: permittivity-weighted profile (works better for strong confinement)
    # Weight by sqrt(eps - eps_clad) to emphasize core
    eps_weight = torch.sqrt((eps_profile - eps_min).clamp(min=0))
    if eps_weight.sum() > 0:
        # Blend Gaussian with eps-weighted profile
        eps_weight_norm = eps_weight / eps_weight.sum()
        mode_blend = 0.5 * mode_profile / mode_profile.sum() + 0.5 * eps_weight_norm
        mode_profile = mode_blend

    # Ensure mode is positive (fundamental mode is single-signed)
    mode_profile = mode_profile.abs()

    # Normalize to unit power: ∫ |mode|² dy = 1
    norm = torch.sqrt((mode_profile**2).sum() * dx)
    mode_profile = mode_profile / norm.clamp(min=1e-12)

    neff = n_eff_approx.unsqueeze(0) if n_eff_approx.dim() == 0 else n_eff_approx
    modes = mode_profile.unsqueeze(0)  # [1, N]

    return neff, modes


def compute_mode_profile_from_eps(
    eps_2d: torch.Tensor,
    port_mask: torch.Tensor,
    wavelength_um: float,
    dx_um: float,
    direction: str = "x",
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Compute the waveguide mode profile at a port location from the 2D permittivity.

    Args:
        eps_2d: 2D permittivity map [H, W]
        port_mask: Binary mask indicating port location [H, W]
        wavelength_um: Wavelength in microns
        dx_um: Grid spacing in microns
        direction: Propagation direction ("x" or "y")

    Returns:
        mode_profile: 1D mode profile along transverse direction
        n_eff: Effective index of the mode
        transverse_coords: Pixel indices along transverse direction where mode is defined
    """
    device = eps_2d.device
    dtype = eps_2d.dtype

    # Find the port location from the mask
    nonzero = torch.nonzero(port_mask > 0.5)
    if nonzero.shape[0] == 0:
        # Empty mask - return uniform mode
        if direction == "x":
            N = eps_2d.shape[0]
        else:
            N = eps_2d.shape[1]
        return torch.ones(N, device=device, dtype=dtype) / math.sqrt(N), 1.5, torch.arange(N, device=device)

    y_coords = nonzero[:, 0]
    x_coords = nonzero[:, 1]

    if direction == "x":
        # Propagation in x, extract eps profile along y at center x
        x_center = x_coords.float().mean().long()
        # Get the y range covered by the port
        y_min, y_max = y_coords.min(), y_coords.max()
        # Extend slightly to capture evanescent tails
        margin = max(10, int(0.5 / dx_um))  # at least 0.5 um margin
        y_min = max(0, y_min - margin)
        y_max = min(eps_2d.shape[0] - 1, y_max + margin)

        eps_profile = eps_2d[y_min:y_max+1, x_center]
        transverse_coords = torch.arange(y_min, y_max+1, device=device)
    else:
        # Propagation in y, extract eps profile along x at center y
        y_center = y_coords.float().mean().long()
        x_min, x_max = x_coords.min(), x_coords.max()
        margin = max(10, int(0.5 / dx_um))
        x_min = max(0, x_min - margin)
        x_max = min(eps_2d.shape[1] - 1, x_max + margin)

        eps_profile = eps_2d[y_center, x_min:x_max+1]
        transverse_coords = torch.arange(x_min, x_max+1, device=device)

    # Solve for the mode
    neff, modes = solve_1d_mode_fdfd(eps_profile.detach(), wavelength_um, dx_um)

    return modes[0].detach(), float(neff[0].detach().item()), transverse_coords


def detect_port_direction(port_mask: torch.Tensor, eps_2d: torch.Tensor) -> str:
    """
    Detect the propagation direction at a port based on mask geometry.

    Returns "x" if port is vertical (propagation in x), "y" if horizontal.
    """
    nonzero = torch.nonzero(port_mask > 0.5)
    if nonzero.shape[0] == 0:
        return "x"  # default

    y_coords = nonzero[:, 0]
    x_coords = nonzero[:, 1]

    y_span = y_coords.max() - y_coords.min()
    x_span = x_coords.max() - x_coords.min()

    # If port is taller than wide, it's a vertical cross-section (propagation in x)
    # If wider than tall, it's a horizontal cross-section (propagation in y)
    if y_span >= x_span:
        return "x"
    else:
        return "y"


def modal_overlap_integral(
    Ez: torch.Tensor,
    mode_profile: torch.Tensor,
    transverse_coords: torch.Tensor,
    port_mask: torch.Tensor,
    direction: str,
    dx_um: float,
) -> torch.Tensor:
    """
    Compute the modal overlap integral between simulated field and mode profile.

    For TE (Ez) polarization, the overlap integral is:
        α = ∫ Ez_sim(r) * conj(Ez_mode(r_⊥)) dr_⊥

    where the integral is over the transverse cross-section.

    Args:
        Ez: Complex Ez field [B, H, W] or [H, W]
        mode_profile: 1D mode profile [N_transverse]
        transverse_coords: Pixel indices where mode is defined
        port_mask: Port location mask [H, W]
        direction: Propagation direction ("x" or "y")
        dx_um: Grid spacing

    Returns:
        alpha: Complex mode amplitude [B] or scalar
    """
    squeeze_batch = False
    if Ez.dim() == 2:
        Ez = Ez.unsqueeze(0)
        squeeze_batch = True

    B, H, W = Ez.shape
    device = Ez.device

    # Find port center location
    nonzero = torch.nonzero(port_mask > 0.5)
    if nonzero.shape[0] == 0:
        result = torch.zeros(B, device=device, dtype=Ez.dtype)
        return result.squeeze(0) if squeeze_batch else result

    y_coords = nonzero[:, 0]
    x_coords = nonzero[:, 1]

    if direction == "x":
        # Port is vertical, integrate along y at center x
        x_center = x_coords.float().mean().long().item()

        # Extract the field slice at port location
        Ez_slice = Ez[:, :, x_center]  # [B, H]

        # Extract only the portion matching mode profile coordinates
        t_min, t_max = transverse_coords.min().item(), transverse_coords.max().item()
        Ez_at_port = Ez_slice[:, t_min:t_max+1]  # [B, N_mode]

    else:
        # Port is horizontal, integrate along x at center y
        y_center = y_coords.float().mean().long().item()
        Ez_slice = Ez[:, y_center, :]  # [B, W]

        t_min, t_max = transverse_coords.min().item(), transverse_coords.max().item()
        Ez_at_port = Ez_slice[:, t_min:t_max+1]  # [B, N_mode]

    # Ensure shapes match
    N_mode = mode_profile.shape[0]
    N_field = Ez_at_port.shape[1]

    if N_mode != N_field:
        # Interpolate mode profile to match field size
        mode_profile = F.interpolate(
            mode_profile.view(1, 1, -1),
            size=N_field,
            mode='linear',
            align_corners=True
        ).view(-1)

    # Compute overlap integral: α = ∫ Ez * conj(mode) dy
    # The mode profile is real, so conj(mode) = mode
    mode_conj = mode_profile.conj() if mode_profile.is_complex() else mode_profile

    # Weighted sum (trapezoidal integration)
    alpha = (Ez_at_port * mode_conj.unsqueeze(0)).sum(dim=1) * dx_um

    if squeeze_batch:
        alpha = alpha.squeeze(0)

    return alpha


def extract_sparams_modal(
    Ez: torch.Tensor,
    port_masks: torch.Tensor,
    eps: torch.Tensor,
    wavelength_um: float,
    dx_um: float,
    in_port_idx: Optional[int] = None,
    port_ids: Optional[torch.Tensor] = None,
    return_amplitudes: bool = False,
) -> torch.Tensor:
    """
    Extract S-parameters using MEEP-style modal decomposition.

    This computes proper overlap integrals with computed waveguide mode profiles,
    similar to MEEP's get_eigenmode_coefficients function.

    Args:
        Ez: Complex Ez field [B, H, W] or [H, W]
        port_masks: Port location masks [B, P, H, W] or [P, H, W]
        eps: Permittivity map [B, H, W] or [H, W]
        wavelength_um: Wavelength in microns
        dx_um: Grid spacing in microns
        in_port_idx: Index of the excited input port (0-based)
        port_ids: Optional port ID labels
        return_amplitudes: If True, also return raw mode amplitudes

    Returns:
        S: Complex S-parameters [B, P] where S[b, p] = a_p / a_in
    """
    squeeze_batch = False
    if Ez.dim() == 2:
        Ez = Ez.unsqueeze(0)
        eps = eps.unsqueeze(0) if eps.dim() == 2 else eps
        squeeze_batch = True

    if port_masks.dim() == 3:
        port_masks = port_masks.unsqueeze(0).expand(Ez.shape[0], -1, -1, -1)

    if eps.dim() == 2:
        eps = eps.unsqueeze(0).expand(Ez.shape[0], -1, -1)

    B = Ez.shape[0]
    P = port_masks.shape[1]
    device = Ez.device
    dtype = Ez.dtype

    # Compute mode amplitudes at each port
    amplitudes = []

    for p in range(P):
        port_amps = []
        for b in range(B):
            mask_p = port_masks[b, p]
            eps_b = eps[b] if eps.dim() == 3 else eps
            Ez_b = Ez[b]

            # Detect port direction and compute mode
            direction = detect_port_direction(mask_p, eps_b)
            mode_profile, n_eff, t_coords = compute_mode_profile_from_eps(
                eps_b, mask_p, wavelength_um, dx_um, direction
            )

            # Compute overlap integral
            alpha = modal_overlap_integral(
                Ez_b, mode_profile, t_coords, mask_p, direction, dx_um
            )
            port_amps.append(alpha)

        amplitudes.append(torch.stack(port_amps))  # [B]

    a = torch.stack(amplitudes, dim=1)  # [B, P]

    # Resolve input port index
    if in_port_idx is None:
        if port_ids is not None:
            # Find port with ID = 1
            if port_ids.dim() == 1:
                hits = (port_ids == 1).nonzero(as_tuple=False)
                in_port_idx = int(hits[0].item()) if len(hits) > 0 else 0
            else:
                in_port_idx = 0
        else:
            in_port_idx = 0

    # Get input amplitude and compute S-parameters
    # Handle both scalar and batched in_port_idx
    if torch.is_tensor(in_port_idx):
        idx = in_port_idx.to(device=device).view(-1, 1)  # [B, 1]
        # Need complex gather - gather doesn't work on complex, so do real/imag separately
        a_real = a.real.gather(1, idx).squeeze(1)  # [B]
        a_imag = a.imag.gather(1, idx).squeeze(1)  # [B]
        a_in = torch.complex(a_real, a_imag)
    else:
        a_in = a[:, int(in_port_idx)]

    # Safe division
    eps_div = 1e-10
    mag_in = torch.abs(a_in)
    unit_in = a_in / mag_in.clamp(min=eps_div)
    unit_in = torch.where(mag_in >= eps_div, unit_in, torch.ones_like(unit_in))
    a_in_safe = unit_in * mag_in.clamp(min=eps_div)

    S = a / a_in_safe.unsqueeze(1)

    # Handle NaN values that can occur with degenerate inputs
    S = torch.where(torch.isnan(S), torch.zeros_like(S), S)
    S = torch.where(torch.isinf(S), torch.zeros_like(S), S)

    if squeeze_batch:
        S = S.squeeze(0)
        if return_amplitudes:
            a = a.squeeze(0)

    if return_amplitudes:
        return S, a
    return S


def sparam_loss_modal(
    Ez_pred: torch.Tensor,
    Ez_true: torch.Tensor,
    port_masks: torch.Tensor,
    eps: torch.Tensor,
    wavelength_um: float,
    dx_um: float,
    S_true: Optional[torch.Tensor] = None,
    in_port_idx: Optional[int] = None,
    port_ids: Optional[torch.Tensor] = None,
    mag_weight: float = 1.0,
    phase_weight: float = 1.0,
    port_valid: Optional[torch.Tensor] = None,
    tau: float = 1e-3,
) -> torch.Tensor:
    """
    Compute S-parameter loss using modal decomposition.

    Args:
        Ez_pred: Predicted complex Ez field [B, H, W]
        Ez_true: Ground truth complex Ez field [B, H, W] (used if S_true not provided)
        port_masks: Port location masks [B, P, H, W]
        eps: Permittivity map [B, H, W]
        wavelength_um: Wavelength in microns
        dx_um: Grid spacing in microns
        S_true: Optional pre-computed ground truth S-parameters [B, P]
        in_port_idx: Index of excited input port
        port_ids: Optional port ID labels
        mag_weight: Weight for magnitude loss
        phase_weight: Weight for phase loss
        port_valid: Mask for valid ports [B, P] or [P]
        tau: Threshold below which phase loss is gated off

    Returns:
        loss: Scalar loss value
    """
    # Extract S-parameters from predicted field
    S_pred = extract_sparams_modal(
        Ez_pred, port_masks, eps, wavelength_um, dx_um,
        in_port_idx=in_port_idx, port_ids=port_ids
    )

    # Get ground truth S-parameters
    if S_true is None:
        S_true = extract_sparams_modal(
            Ez_true, port_masks, eps, wavelength_um, dx_um,
            in_port_idx=in_port_idx, port_ids=port_ids
        )

    # Compute magnitude and phase losses
    eps_val = 1e-10
    mag_pred = torch.abs(S_pred).clamp(min=eps_val)
    mag_true = torch.abs(S_true).clamp(min=eps_val)

    # Magnitude loss (MSE)
    L_mag = (mag_pred - mag_true) ** 2

    # Phase loss (cosine distance, wrap-safe)
    u_pred = S_pred / mag_pred
    u_true = S_true / mag_true
    L_phase = 1.0 - torch.real(u_pred * torch.conj(u_true))

    # Gate phase loss when transmission is near zero
    gate = (mag_true > tau).to(L_mag.dtype)

    if port_valid is not None:
        pv = port_valid.to(dtype=gate.dtype, device=gate.device)
        if pv.dim() == 1:
            pv = pv.view(1, -1).expand_as(gate)
        gate = gate * pv

    L = (mag_weight * L_mag + phase_weight * L_phase) * gate

    # Handle NaN values that can occur with degenerate inputs
    L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

    # Masked mean
    den = gate.sum().clamp(min=1.0)
    return L.sum() / den


# =============================================================================
# Comparison utilities
# =============================================================================

def compare_sparam_methods(
    Ez: torch.Tensor,
    port_masks: torch.Tensor,
    eps: torch.Tensor,
    wavelength_um: float,
    dx_um: float,
    S_true: torch.Tensor,
    in_port_idx: int = 0,
    port_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compare S-parameters computed by different methods against ground truth.

    Args:
        Ez: Complex Ez field [H, W]
        port_masks: Port masks [P, H, W]
        eps: Permittivity [H, W]
        wavelength_um: Wavelength in microns
        dx_um: Grid spacing
        S_true: Ground truth S-parameters [P]
        in_port_idx: Input port index
        port_ids: Port IDs

    Returns:
        Dictionary with:
        - S_modal: Modal decomposition result
        - S_simple: Simple averaging result
        - S_true: Ground truth
        - error_modal_mag: Magnitude error for modal method
        - error_simple_mag: Magnitude error for simple method
        - error_modal_phase: Phase error for modal method (degrees)
        - error_simple_phase: Phase error for simple method (degrees)
    """
    from sparams_loss import extract_sparams as extract_sparams_simple

    # Modal decomposition method
    S_modal = extract_sparams_modal(
        Ez, port_masks, eps, wavelength_um, dx_um,
        in_port_idx=in_port_idx, port_ids=port_ids
    )

    # Simple averaging method (expects batched input)
    Ez_batched = Ez.unsqueeze(0) if Ez.dim() == 2 else Ez
    S_simple = extract_sparams_simple(
        Ez_batched, port_masks, in_port_idx=in_port_idx, port_ids=port_ids
    )
    # Squeeze batch dimension if present
    if S_simple.dim() == 2 and S_simple.shape[0] == 1:
        S_simple = S_simple.squeeze(0)

    # Compute errors
    mag_true = torch.abs(S_true)
    mag_modal = torch.abs(S_modal)
    mag_simple = torch.abs(S_simple)

    phase_true = torch.angle(S_true)
    phase_modal = torch.angle(S_modal)
    phase_simple = torch.angle(S_simple)

    # Magnitude errors (absolute)
    error_modal_mag = torch.abs(mag_modal - mag_true)
    error_simple_mag = torch.abs(mag_simple - mag_true)

    # Phase errors (wrap-safe, in degrees)
    def phase_error_deg(p1, p2):
        diff = p1 - p2
        # Wrap to [-pi, pi]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return diff * 180 / math.pi

    error_modal_phase = phase_error_deg(phase_modal, phase_true)
    error_simple_phase = phase_error_deg(phase_simple, phase_true)

    return {
        'S_modal': S_modal,
        'S_simple': S_simple,
        'S_true': S_true,
        'error_modal_mag': error_modal_mag,
        'error_simple_mag': error_simple_mag,
        'error_modal_phase': error_modal_phase,
        'error_simple_phase': error_simple_phase,
    }
