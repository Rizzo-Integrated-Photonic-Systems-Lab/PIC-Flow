"""
Improved S-Parameter Prediction Head for Complex Physics UNet.

WHY A LEARNED HEAD WORKS (Physics Intuition):
=============================================

The fundamental challenge is that proper S-parameter extraction requires separating
forward and backward propagating modes, which MEEP does using both E and H fields:
    a+ = (E + Z*H)  [forward]
    a- = (E - Z*H)  [backward]
    S11 = a- / a+   [reflection at input]
    S21 = a+(out) / a+(in)  [transmission]

We only have Ez, so we can't directly compute this. However, a neural network CAN
learn to infer this information because:

1. **Phase Gradients Encode Propagation Direction:**
   - Forward waves: phase advances in +x direction (dφ/dx > 0 for +x propagation)
   - Backward waves: phase advances in -x direction (dφ/dx < 0)
   - Standing waves (interference): oscillating amplitude pattern

   The network can learn: "if phase gradient at input port points inward AND
   amplitude is high, there's significant reflection"

2. **Interference Patterns Indicate Reflection:**
   - Pure transmission: smooth phase progression through device
   - Reflection present: standing wave ripples in amplitude
   - The spatial pattern encodes the forward/backward decomposition implicitly

3. **UNet Features Are Rich:**
   - Bottleneck captures global field structure (entire device response)
   - Local features at ports capture the specific field pattern there
   - Complex features preserve phase relationships throughout

4. **Supervised Learning Against MEEP:**
   - MEEP computes true S-parameters with proper mode decomposition
   - Network learns the mapping: field_pattern -> S_parameters
   - Given enough data, it can learn the implicit relationship

ARCHITECTURE:
============
1. Extract bottleneck features (global context)
2. Extract features at each port using port masks (local context)
3. Compute explicit phase gradient features (propagation direction hint)
4. Combine with conditioning (wavelength) and input port indicator
5. MLP predicts S-parameters directly

The key insight is that we're NOT trying to do physics-based extraction.
We're learning a direct mapping from field features to S-parameters,
supervised by MEEP's ground truth.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PortFeatureExtractor(nn.Module):
    """
    Extract features at port locations using masked pooling and attention.

    For each port, we:
    1. Use the port mask to extract local field features
    2. Compute amplitude and phase statistics
    3. Compute phase gradients (propagation direction indicator)
    """

    def __init__(self, max_ports: int = 4):
        super().__init__()
        self.max_ports = max_ports

    def forward(
        self,
        E: torch.Tensor,  # Complex field [B, H, W]
        port_masks: torch.Tensor,  # [B, P, H, W]
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features at each port.

        Returns:
            amp_features: [B, P] - amplitude at each port
            phase_features: [B, P] - mean phase at each port
            phase_grad_x: [B, P] - x phase gradient (propagation direction)
            phase_grad_y: [B, P] - y phase gradient
        """
        B = E.shape[0]
        device = E.device
        dtype = E.real.dtype

        if port_masks.dim() == 3:
            port_masks = port_masks.unsqueeze(0).expand(B, -1, -1, -1)

        P = min(port_masks.shape[1], self.max_ports)

        # Compute field properties
        amp = torch.abs(E).clamp_min(eps)  # [B, H, W]
        phase = torch.angle(E)  # [B, H, W]

        # Phase gradients (central differences)
        # These indicate propagation direction!
        phase_dx = torch.zeros_like(phase)
        phase_dy = torch.zeros_like(phase)

        # Unwrap phase locally for gradient computation
        # Use complex gradient: d(angle(E))/dx ≈ Im(dE/dx * conj(E)) / |E|^2
        dE_dx = torch.zeros_like(E)
        dE_dy = torch.zeros_like(E)
        dE_dx[:, :, 1:-1] = (E[:, :, 2:] - E[:, :, :-2]) / 2.0
        dE_dy[:, 1:-1, :] = (E[:, 2:, :] - E[:, :-2, :]) / 2.0

        amp_sq = (amp ** 2).clamp_min(eps)
        phase_dx = torch.imag(dE_dx * torch.conj(E)) / amp_sq
        phase_dy = torch.imag(dE_dy * torch.conj(E)) / amp_sq

        # Extract features at each port
        amp_features = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        phase_features = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        phase_grad_x = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        phase_grad_y = torch.zeros((B, self.max_ports), device=device, dtype=dtype)

        for p in range(P):
            mask = port_masks[:, p].to(dtype)  # [B, H, W]
            mask_sum = mask.sum(dim=(1, 2)).clamp_min(1.0)  # [B]

            # Amplitude-weighted pooling (emphasize high-field regions)
            weights = mask * amp
            weight_sum = weights.sum(dim=(1, 2)).clamp_min(eps)

            # Mean amplitude at port
            amp_features[:, p] = (weights * amp).sum(dim=(1, 2)) / weight_sum

            # Mean phase at port (amplitude-weighted)
            phase_features[:, p] = (weights * phase).sum(dim=(1, 2)) / weight_sum

            # Mean phase gradients at port (amplitude-weighted)
            # These are KEY - they indicate if wave is propagating toward or away from port
            phase_grad_x[:, p] = (weights * phase_dx).sum(dim=(1, 2)) / weight_sum
            phase_grad_y[:, p] = (weights * phase_dy).sum(dim=(1, 2)) / weight_sum

        return amp_features, phase_features, phase_grad_x, phase_grad_y


class ImprovedSParamHead(nn.Module):
    """
    Improved S-Parameter prediction head.

    Key improvements over the simple version:
    1. Uses explicit phase gradient features (propagation direction)
    2. Deeper architecture with residual connections
    3. Separate processing for amplitude and phase
    4. Port-wise attention mechanism
    """

    def __init__(
        self,
        max_ports: int = 4,
        cond_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_ports = max_ports
        self.cond_dim = cond_dim

        self.port_extractor = PortFeatureExtractor(max_ports=max_ports)

        # Input features per port:
        # - amplitude (1)
        # - phase (1)
        # - phase_grad_x (1)
        # - phase_grad_y (1)
        # - is_input_port (1)
        # Total: 5 per port
        port_feat_dim = 5

        # Global features:
        # - conditioning (cond_dim)
        # - global amplitude stats (2: mean, std)
        # - global phase gradient magnitude (1)
        global_feat_dim = cond_dim + 3

        total_input_dim = port_feat_dim * max_ports + global_feat_dim

        # Feature encoder
        self.input_norm = nn.LayerNorm(total_input_dim)
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                ResidualMLPBlock(hidden_dim, dropout=dropout)
            )

        # Separate heads for magnitude and phase (different learning dynamics)
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_ports),
            nn.Softplus(),  # Magnitudes are positive
        )

        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_ports),
            nn.Tanh(),  # Phase in [-1, 1], scale to [-pi, pi]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        E: torch.Tensor,  # Complex field [B, H, W]
        port_masks: torch.Tensor,  # [B, P, H, W]
        in_port_idx: torch.Tensor,  # [B] or scalar
        cond: Optional[torch.Tensor] = None,  # [B, cond_dim]
    ) -> torch.Tensor:
        """
        Predict S-parameters from field and port information.

        Returns:
            S: Complex S-parameters [B, max_ports]
        """
        B = E.shape[0]
        device = E.device
        dtype = E.real.dtype

        # Extract port features
        amp_feat, phase_feat, pg_x, pg_y = self.port_extractor(E, port_masks)

        # Build input port indicator
        if not torch.is_tensor(in_port_idx):
            in_port_idx = torch.full((B,), int(in_port_idx), device=device, dtype=torch.long)
        in_port_idx = in_port_idx.to(device=device, dtype=torch.long).view(-1)
        if in_port_idx.numel() == 1:
            in_port_idx = in_port_idx.expand(B)

        in_port_oh = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        in_port_idx_clamped = in_port_idx.clamp(0, self.max_ports - 1)
        in_port_oh.scatter_(1, in_port_idx_clamped.view(B, 1), 1.0)

        # Normalize amplitude features (log scale for better range)
        amp_feat_norm = torch.log1p(amp_feat)  # log(1 + x) for stability

        # Stack port features: [B, P, 5]
        port_features = torch.stack([
            amp_feat_norm,
            phase_feat,
            pg_x,
            pg_y,
            in_port_oh,
        ], dim=-1)  # [B, P, 5]
        port_features_flat = port_features.view(B, -1)  # [B, P*5]

        # Global features
        amp_global = torch.abs(E)
        global_amp_mean = amp_global.mean(dim=(1, 2))
        global_amp_std = amp_global.std(dim=(1, 2))
        global_pg_mag = torch.sqrt(pg_x.pow(2).mean(dim=1) + pg_y.pow(2).mean(dim=1))

        if cond is None:
            cond = torch.zeros((B, self.cond_dim), device=device, dtype=dtype)
        elif cond.shape[1] < self.cond_dim:
            pad = torch.zeros((B, self.cond_dim - cond.shape[1]), device=device, dtype=dtype)
            cond = torch.cat([cond, pad], dim=1)
        elif cond.shape[1] > self.cond_dim:
            cond = cond[:, :self.cond_dim]

        global_features = torch.stack([
            global_amp_mean,
            global_amp_std,
            global_pg_mag,
        ], dim=-1)  # [B, 3]
        global_features = torch.cat([cond, global_features], dim=-1)  # [B, cond_dim + 3]

        # Combine all features
        x = torch.cat([port_features_flat, global_features], dim=-1)

        # Process through network
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = F.silu(x)

        for block in self.blocks:
            x = block(x)

        # Predict magnitude and phase separately
        mag = self.mag_head(x)  # [B, P], positive
        phase_normalized = self.phase_head(x)  # [B, P], in [-1, 1]
        phase = phase_normalized * math.pi  # Scale to [-pi, pi]

        # Construct complex S-parameters
        S = mag * torch.exp(1j * phase)

        return S


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with pre-norm."""

    def __init__(self, dim: int, dropout: float = 0.1, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class BottleneckSParamHead(nn.Module):
    """
    S-Parameter head that uses UNet bottleneck features.

    This version takes the bottleneck features from the UNet middle block,
    which contain rich global information about the field structure.

    The bottleneck is processed through:
    1. Global average pooling (complex-aware)
    2. Port-specific feature extraction using masks
    3. Combined MLP prediction
    """

    def __init__(
        self,
        bottleneck_channels: int,  # Number of complex channels in bottleneck
        max_ports: int = 4,
        cond_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.max_ports = max_ports
        self.cond_dim = cond_dim

        # Bottleneck features: 2 * bottleneck_channels (real + imag) after GAP
        # Plus port features and conditioning
        bottleneck_feat_dim = 2 * bottleneck_channels
        port_feat_dim = 2 * max_ports  # amplitude and phase per port
        input_dim = bottleneck_feat_dim + port_feat_dim + cond_dim + max_ports  # +max_ports for input indicator

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(ResidualMLPBlock(hidden_dim, dropout=dropout))

        # Output: complex S-parameters (2 * max_ports for real/imag)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * max_ports),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        bottleneck_r: torch.Tensor,  # [B, C, H', W'] real part
        bottleneck_i: torch.Tensor,  # [B, C, H', W'] imag part
        E: torch.Tensor,  # Complex field [B, H, W] for port extraction
        port_masks: torch.Tensor,  # [B, P, H, W]
        in_port_idx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = bottleneck_r.shape[0]
        device = bottleneck_r.device
        dtype = bottleneck_r.dtype

        # Global average pool bottleneck
        bn_r = bottleneck_r.mean(dim=(2, 3))  # [B, C]
        bn_i = bottleneck_i.mean(dim=(2, 3))  # [B, C]
        bottleneck_feat = torch.cat([bn_r, bn_i], dim=-1)  # [B, 2C]

        # Extract port amplitudes and phases
        if port_masks.dim() == 3:
            port_masks = port_masks.unsqueeze(0).expand(B, -1, -1, -1)
        P = min(port_masks.shape[1], self.max_ports)

        amp_feat = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        phase_feat = torch.zeros((B, self.max_ports), device=device, dtype=dtype)

        amp = torch.abs(E).clamp_min(1e-8)
        phase = torch.angle(E)

        for p in range(P):
            mask = port_masks[:, p].to(dtype)
            weights = mask * amp
            weight_sum = weights.sum(dim=(1, 2)).clamp_min(1.0)
            amp_feat[:, p] = (weights * amp).sum(dim=(1, 2)) / weight_sum
            phase_feat[:, p] = (weights * phase).sum(dim=(1, 2)) / weight_sum

        port_feat = torch.cat([torch.log1p(amp_feat), phase_feat], dim=-1)  # [B, 2P]

        # Input port indicator
        if not torch.is_tensor(in_port_idx):
            in_port_idx = torch.full((B,), int(in_port_idx), device=device, dtype=torch.long)
        in_port_idx = in_port_idx.to(device=device, dtype=torch.long).view(-1)
        if in_port_idx.numel() == 1:
            in_port_idx = in_port_idx.expand(B)

        in_port_oh = torch.zeros((B, self.max_ports), device=device, dtype=dtype)
        in_port_oh.scatter_(1, in_port_idx.clamp(0, self.max_ports - 1).view(B, 1), 1.0)

        # Conditioning
        if cond is None:
            cond = torch.zeros((B, self.cond_dim), device=device, dtype=dtype)
        elif cond.shape[1] != self.cond_dim:
            if cond.shape[1] < self.cond_dim:
                pad = torch.zeros((B, self.cond_dim - cond.shape[1]), device=device, dtype=dtype)
                cond = torch.cat([cond, pad], dim=1)
            else:
                cond = cond[:, :self.cond_dim]

        # Combine features
        x = torch.cat([bottleneck_feat, port_feat, cond, in_port_oh], dim=-1)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        out = self.output(x)
        S_re = out[:, :self.max_ports]
        S_im = out[:, self.max_ports:]

        return torch.complex(S_re, S_im)


# For backward compatibility, keep the simple version too
class SimpleSParamHead(nn.Module):
    """Simple S-param head (original design) for comparison."""

    def __init__(self, max_ports: int = 4, cond_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.max_ports = max_ports
        self.cond_dim = cond_dim

        input_dim = 2 * max_ports + cond_dim + max_ports
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2 * max_ports),
        )

    def forward(self, amp_real, amp_imag, cond, in_port_oh):
        x = torch.cat([amp_real, amp_imag, cond, in_port_oh], dim=-1)
        out = self.mlp(x)
        return torch.complex(out[:, :self.max_ports], out[:, self.max_ports:])
