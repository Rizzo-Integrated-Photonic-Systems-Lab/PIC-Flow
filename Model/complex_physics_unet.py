# complex_physics_unet.py
"""
Complex-valued Physics-embedded UNet for electromagnetic field prediction.

Key features:
- Complex convolutions that natively handle phase relationships
- Physics embedding (Helmholtz residual) computed in complex domain
- Real auxiliary inputs (eps, src_mask, conditioning) merged into complex backbone
- modReLU activations that preserve phase while allowing magnitude gating

Notes (implementation choices):
- ComplexConv2d/ComplexLinear implement correct complex bias: yr += b_r, yi += b_i (no cross-coupling).
- Attention weights use the real part of QK^H ("Hermitian-like"). Docstrings match behavior.
- attention_resolutions is interpreted as DOWNsample factors (ds = 1,2,4,8,...) to match common diffusion UNet codebases.

Updates:
- Fix: when enable_physics_features=False, do NOT concatenate Hr/Hi into the stem input (prevents shape mismatch).
- Stability: zero-init emb_proj final Linear so residual modulation starts at identity.
- Robustness: ComplexGroupNorm uses reshape instead of view (handles non-contiguous tensors safely).
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


# ---------------------------------------------------------------------------
# Complex arithmetic helpers
# ---------------------------------------------------------------------------

def _complex_glorot_normal_(wr: torch.Tensor, wi: torch.Tensor, fan_in: int, fan_out: int):
    """
    Complex Glorot (Xavier) normal init for split real/imag weights.

    Var[wr] = Var[wi] = 1/(fan_in + fan_out)
    so Var[wr + i*wi] = 2/(fan_in + fan_out)
    """
    denom = float(max(1, fan_in + fan_out))
    std = math.sqrt(1.0 / denom)
    nn.init.normal_(wr, mean=0.0, std=std)
    nn.init.normal_(wi, mean=0.0, std=std)


def complex_mul(ar: torch.Tensor, ai: torch.Tensor,
                br: torch.Tensor, bi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complex multiplication: (ar + i*ai) * (br + i*bi)."""
    cr = ar * br - ai * bi
    ci = ar * bi + ai * br
    return cr, ci


def complex_conj_mul(ar: torch.Tensor, ai: torch.Tensor,
                     br: torch.Tensor, bi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complex conjugate multiplication: (ar + i*ai) * conj(br + i*bi)."""
    cr = ar * br + ai * bi
    ci = ai * br - ar * bi
    return cr, ci


# ---------------------------------------------------------------------------
# Complex layers
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    """
    Complex 2D convolution: (Wr + i*Wi) * (xr + i*xi)

    y = W * x
    yr = Wr*xr - Wi*xi
    yi = Wr*xi + Wi*xr

    IMPORTANT: Complex bias must be (b_r, b_i) added independently to yr and yi.
    We therefore disable internal real Conv2d biases and implement explicit complex bias.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Disable internal biases to avoid cross-coupling in complex formula.
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )

        self.use_bias = bool(bias)
        if self.use_bias:
            self.bias_r = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            self.bias_i = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        self._init_weights()

    def _init_weights(self):
        kx, ky = self.conv_real.kernel_size
        fan_in = int(self.in_channels * kx * ky)
        fan_out = int(self.out_channels * kx * ky)
        _complex_glorot_normal_(self.conv_real.weight, self.conv_imag.weight, fan_in=fan_in, fan_out=fan_out)

        if self.use_bias:
            nn.init.zeros_(self.bias_r)
            nn.init.zeros_(self.bias_i)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yr = self.conv_real(xr) - self.conv_imag(xi)
        yi = self.conv_real(xi) + self.conv_imag(xr)
        if self.use_bias:
            yr = yr + self.bias_r
            yi = yi + self.bias_i
        return yr, yi


class ComplexLinear(nn.Module):
    """Complex linear layer with correct complex bias."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear_real = nn.Linear(in_features, out_features, bias=False)
        self.linear_imag = nn.Linear(in_features, out_features, bias=False)

        self.use_bias = bool(bias)
        if self.use_bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        self._init_weights(in_features, out_features)

    def _init_weights(self, fan_in: int, fan_out: int):
        _complex_glorot_normal_(self.linear_real.weight, self.linear_imag.weight, fan_in=fan_in, fan_out=fan_out)
        if self.use_bias:
            nn.init.zeros_(self.bias_r)
            nn.init.zeros_(self.bias_i)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yr = self.linear_real(xr) - self.linear_imag(xi)
        yi = self.linear_real(xi) + self.linear_imag(xr)
        if self.use_bias:
            yr = yr + self.bias_r
            yi = yi + self.bias_i
        return yr, yi


class ModReLU(nn.Module):
    """
    Modulus ReLU: Apply ReLU to magnitude with learnable bias, preserve phase.

    z_out = ReLU(|z| + b) * exp(i * arg(z))
    """
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1) - 0.1)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(xr * xr + xi * xi + self.eps)
        mag_activated = F.relu(mag + self.bias)

        phase_r = xr / mag
        phase_i = xi / mag

        yr = mag_activated * phase_r
        yi = mag_activated * phase_i
        return yr, yi


class CReLU(nn.Module):
    """Cartesian ReLU: Apply ReLU to real and imaginary parts separately."""
    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.relu(xr), F.relu(xi)


class ComplexGroupNorm(nn.Module):
    """
    Complex Group Normalization.
    Normalizes real and imaginary parts jointly using complex variance.
    """
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = min(num_groups, num_channels)
        while num_channels % self.num_groups != 0 and self.num_groups > 1:
            self.num_groups -= 1
        self.eps = eps

        self.gamma_r = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma_i = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta_r = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta_i = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = xr.shape
        G = self.num_groups

        # reshape (not view) for non-contiguous safety
        xr_g = xr.reshape(B, G, C // G, H, W)
        xi_g = xi.reshape(B, G, C // G, H, W)

        mean_r = xr_g.mean(dim=(2, 3, 4), keepdim=True)
        mean_i = xi_g.mean(dim=(2, 3, 4), keepdim=True)

        var = ((xr_g - mean_r) ** 2 + (xi_g - mean_i) ** 2).mean(dim=(2, 3, 4), keepdim=True)
        std = torch.sqrt(var + self.eps)

        xr_g = (xr_g - mean_r) / std
        xi_g = (xi_g - mean_i) / std

        xr_n = xr_g.reshape(B, C, H, W)
        xi_n = xi_g.reshape(B, C, H, W)

        yr = self.gamma_r * xr_n - self.gamma_i * xi_n + self.beta_r
        yi = self.gamma_r * xi_n + self.gamma_i * xr_n + self.beta_i
        return yr, yi


class ComplexDownsample(nn.Module):
    """Downsample complex tensor using strided complex conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = ComplexConv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv(xr, xi)


class ComplexUpsample(nn.Module):
    """Upsample complex tensor using interpolation + conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = ComplexConv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xr = F.interpolate(xr, scale_factor=2, mode="nearest")
        xi = F.interpolate(xi, scale_factor=2, mode="nearest")
        return self.conv(xr, xi)


# ---------------------------------------------------------------------------
# Complex ResBlock
# ---------------------------------------------------------------------------

class ComplexResBlock(nn.Module):
    """
    Complex residual block with timestep/conditioning embedding.

    Architecture (pre-act style):
    x -> Norm -> Conv -> modReLU -> (emb) -> Norm -> modReLU -> Conv -> + x
    """
    def __init__(self, channels: int, emb_dim: int, dropout: float = 0.0,
                 out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.emb_dim = emb_dim

        self.norm1 = ComplexGroupNorm(channels)
        self.conv1 = ComplexConv2d(channels, self.out_channels, kernel_size=3, padding=1)
        self.act1 = ModReLU(self.out_channels)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * self.out_channels),
        )

        # Stability: start modulation at ~0 so block starts as near-identity
        nn.init.zeros_(self.emb_proj[1].weight)
        nn.init.zeros_(self.emb_proj[1].bias)

        self.norm2 = ComplexGroupNorm(self.out_channels)
        self.act2 = ModReLU(self.out_channels)
        self.conv2 = ComplexConv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if channels != self.out_channels:
            self.skip = ComplexConv2d(channels, self.out_channels, kernel_size=1)
        else:
            self.skip = None

        # Zero-init last conv for stable training
        nn.init.zeros_(self.conv2.conv_real.weight)
        nn.init.zeros_(self.conv2.conv_imag.weight)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor,
                emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.skip is not None:
            skip_r, skip_i = self.skip(xr, xi)
        else:
            skip_r, skip_i = xr, xi

        hr, hi = self.norm1(xr, xi)
        hr, hi = self.conv1(hr, hi)
        hr, hi = self.act1(hr, hi)

        emb_out = self.emb_proj(emb).view(emb.shape[0], -1, 1, 1)
        scale_r, scale_i = emb_out[:, :self.out_channels], emb_out[:, self.out_channels:]

        # Complex scale injection (residual-style modulation)
        hr_new = scale_r * hr - scale_i * hi + hr
        hi_new = scale_r * hi + scale_i * hr + hi
        hr, hi = hr_new, hi_new

        hr, hi = self.norm2(hr, hi)
        hr, hi = self.act2(hr, hi)
        hr = self.dropout(hr)
        hi = self.dropout(hi)
        hr, hi = self.conv2(hr, hi)

        return skip_r + hr, skip_i + hi


# ---------------------------------------------------------------------------
# Complex Attention
# ---------------------------------------------------------------------------

class ComplexAttentionBlock(nn.Module):
    """
    Complex self-attention with complex Q,K,V.

    Attention logits are computed from the REAL part of QK^H (Hermitian-like),
    then softmaxed and applied to complex V.
    """
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = ComplexGroupNorm(channels)
        self.qkv = ComplexConv2d(channels, 3 * channels, kernel_size=1)
        self.proj = ComplexConv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.proj.conv_real.weight)
        nn.init.zeros_(self.proj.conv_imag.weight)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = xr.shape

        hr, hi = self.norm(xr, xi)
        qkv_r, qkv_i = self.qkv(hr, hi)

        qkv_r = qkv_r.view(B, 3, self.num_heads, self.head_dim, H * W)
        qkv_i = qkv_i.view(B, 3, self.num_heads, self.head_dim, H * W)

        qr, kr, vr = qkv_r[:, 0], qkv_r[:, 1], qkv_r[:, 2]
        qi, ki, vi = qkv_i[:, 0], qkv_i[:, 1], qkv_i[:, 2]

        # qk = Q K^H = (qr+i qi)(kr - i ki)
        # real(qk) = qr*kr + qi*ki
        qk_r = torch.einsum("bhdn,bhdm->bhnm", qr, kr) + torch.einsum("bhdn,bhdm->bhnm", qi, ki)
        attn = qk_r / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out_r = torch.einsum("bhnm,bhdm->bhdn", attn, vr)
        out_i = torch.einsum("bhnm,bhdm->bhdn", attn, vi)

        out_r = out_r.reshape(B, C, H, W)
        out_i = out_i.reshape(B, C, H, W)

        out_r, out_i = self.proj(out_r, out_i)
        return xr + out_r, xi + out_i


# ---------------------------------------------------------------------------
# Complex Helmholtz Residual
# ---------------------------------------------------------------------------

class ComplexHelmholtz2D(nn.Module):
    """
    Compute Helmholtz residual in complex domain.

    H = ∇²E + k0² * eps * E  (should be ~0 for homogeneous-source-free regions)

    Returns complex residual (Hr, Hi).
    """
    def __init__(self, dx: float = 1.0, dy: float = 1.0, omega: float = 1.0, pml_cells: int = 0):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.omega = omega
        self.pml_cells = int(pml_cells)

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        laplacian = torch.tensor(
            [[0.0,     inv_dy2, 0.0],
             [inv_dx2, -(2.0 * inv_dx2 + 2.0 * inv_dy2), inv_dx2],
             [0.0,     inv_dy2, 0.0]],
            dtype=torch.float32
        )
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    def forward(self, Er: torch.Tensor, Ei: torch.Tensor,
                eps: torch.Tensor, k0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lap_r = F.conv2d(Er, self.laplacian, padding=1)
        lap_i = F.conv2d(Ei, self.laplacian, padding=1)

        if k0 is None:
            # Interprets omega as k0 (consistent if omega is set to 2π/λ in dataset units).
            k0_sq = self.omega ** 2
        else:
            k0_sq = k0.view(-1, 1, 1, 1) ** 2

        k2n2 = k0_sq * eps

        Hr = lap_r + k2n2 * Er
        Hi = lap_i + k2n2 * Ei
        return Hr, Hi


# ---------------------------------------------------------------------------
# Real-to-Complex and Complex-to-Real bridges
# ---------------------------------------------------------------------------

class RealToComplex(nn.Module):
    """Convert real auxiliary channels to complex representation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        kx, ky = self.conv_r.kernel_size
        fan_in = int(in_channels * kx * ky)
        fan_out = int(out_channels * kx * ky)
        _complex_glorot_normal_(self.conv_r.weight, self.conv_i.weight, fan_in=fan_in, fan_out=fan_out)
        if self.conv_r.bias is not None:
            nn.init.zeros_(self.conv_r.bias)
        if self.conv_i.bias is not None:
            nn.init.zeros_(self.conv_i.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_r(x), self.conv_i(x)


class ComplexToReal(nn.Module):
    """Convert complex representation to real output."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([xr, xi], dim=1))


# ---------------------------------------------------------------------------
# ComplexPhysicsUNet
# ---------------------------------------------------------------------------

class ComplexPhysicsUNet(nn.Module):
    """
    Complex-valued Physics-embedded UNet.

    Input x: (B, C, H, W) where C = [Er, Ei, eps, src_mask, ...]
    Output u_t_pred: (B, 2, H, W) velocity field [vr, vi]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        attention_resolutions: Sequence[int] = (),
        dropout: float = 0.0,
        num_heads: int = 1,
        cond_dim: int = 0,
        dx: float = 1.0,
        dy: float = 1.0,
        omega: float = 1.0,
        enable_physics_features: bool = True,
        enable_sparam_head: bool = False,
        # Compatibility args
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        pml_cells: int = 0,
    ):
        super().__init__()

        if dims != 2:
            raise ValueError("ComplexPhysicsUNet is currently implemented for 2D only.")
        if out_channels != 2:
            raise ValueError("ComplexPhysicsUNet outputs 2 channels (real+imag velocity). Set out_channels=2.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = set(attention_resolutions)  # interpreted as ds factors
        self.dropout = dropout
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.enable_physics_features = enable_physics_features
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.pml_cells = int(pml_cells)

        # Complex field channels (Er, Ei) = 1 complex channel
        self.n_complex_in = 1
        # Real auxiliary channels include eps, src_mask, etc. (everything after Er,Ei)
        self.n_aux_in = in_channels - 2

        self.helmholtz = ComplexHelmholtz2D(dx=dx, dy=dy, omega=omega, pml_cells=pml_cells)
        self.n_phys_channels = 1 if enable_physics_features else 0  # 1 complex channel (Helmholtz residual)

        # Normalization stats
        self.normalize_inputs = False
        self.normalize_eps = False
        self.register_buffer("ez_real_mean", torch.tensor(0.0))
        self.register_buffer("ez_real_std", torch.tensor(1.0))
        self.register_buffer("ez_imag_mean", torch.tensor(0.0))
        self.register_buffer("ez_imag_std", torch.tensor(1.0))
        self.register_buffer("eps_mean", torch.tensor(0.0))
        self.register_buffer("eps_std", torch.tensor(1.0))

        # For compatibility
        self.base_in_channels = in_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels + cond_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.n_aux_in > 0:
            self.aux_encoder = RealToComplex(self.n_aux_in, model_channels // 2)
        else:
            self.aux_encoder = None

        stem_in_channels = self.n_complex_in + self.n_phys_channels
        if self.aux_encoder is not None:
            stem_in_channels += model_channels // 2
        self.input_stem = ComplexConv2d(stem_in_channels, model_channels, kernel_size=3, padding=1)

        # ---- Encoder ----
        self.input_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([
                    ComplexResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels))
                ])
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(ComplexAttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(layers)
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([ComplexDownsample(ch)]))
                input_block_chans.append(ch)
                ds *= 2

        # ---- Middle ----
        self.middle_block = nn.ModuleList([
            ComplexResBlock(ch, time_embed_dim, dropout),
            ComplexAttentionBlock(ch, num_heads=num_heads),
            ComplexResBlock(ch, time_embed_dim, dropout),
        ])

        # ---- Decoder ----
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = nn.ModuleList([
                    ComplexResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(mult * model_channels))
                ])
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(ComplexAttentionBlock(ch, num_heads=num_heads))
                if level > 0 and i == num_res_blocks:
                    layers.append(ComplexUpsample(ch))
                    ds //= 2
                self.output_blocks.append(layers)

        # ---- Output ----
        self.out_norm = ComplexGroupNorm(ch)
        self.out_act = ModReLU(ch)
        self.out_conv = ComplexConv2d(ch, 1, kernel_size=3, padding=1)  # 1 complex channel out

        # S-param head (optional) - uses ImprovedSParamHead with phase gradient features
        self.enable_sparam_head = enable_sparam_head
        self.max_ports = 4
        if enable_sparam_head:
            from sparam_head import ImprovedSParamHead
            self.sparam_head = ImprovedSParamHead(
                max_ports=self.max_ports,
                cond_dim=cond_dim,
                hidden_dim=max(256, 2 * model_channels),
                num_layers=4,
                dropout=0.1,
            )
        else:
            self.sparam_head = None

    @torch.no_grad()
    def set_normalization_stats(self, stats: dict, normalize_eps: bool = True):
        self.normalize_inputs = True
        self.normalize_eps = bool(normalize_eps)

        self.ez_real_mean.fill_(float(stats["ez_real_mean"]))
        self.ez_real_std.fill_(float(stats["ez_real_std"]))
        self.ez_imag_mean.fill_(float(stats["ez_imag_mean"]))
        self.ez_imag_std.fill_(float(stats["ez_imag_std"]))
        self.eps_mean.fill_(float(stats["eps_mean"]))
        self.eps_std.fill_(float(stats["eps_std"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        lambda_um: Optional[torch.Tensor] = None,
        phys_gate: float = 1.0,
        phase_gate: float = 1.0,  # accepted for compatibility; unused
        return_sparams: bool = False,
        aux: Optional[dict] = None,
        sig_min: float = 0.0,
        **kwargs,
    ):
        B, C, H, W = x.shape

        Er_norm = x[:, 0:1]
        Ei_norm = x[:, 1:2]
        eps_norm = x[:, 2:3]
        aux_real = x[:, 2:]  # includes eps, src_mask, etc.

        if getattr(self, "normalize_inputs", False):
            Er = Er_norm * self.ez_real_std + self.ez_real_mean
            Ei = Ei_norm * self.ez_imag_std + self.ez_imag_mean
            if getattr(self, "normalize_eps", False):
                eps = eps_norm * self.eps_std + self.eps_mean
            else:
                eps = eps_norm
        else:
            Er, Ei, eps = Er_norm, Ei_norm, eps_norm

        # PML mask (no inplace ops)
        p = self.pml_cells
        if p > 0:
            p2 = min(p + 4, max(0, H // 2 - 1), max(0, W // 2 - 1))
            if p2 > 0:
                y_coords = torch.arange(H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
                x_coords = torch.arange(W, device=x.device, dtype=torch.float32).view(1, 1, 1, W)
                pml_m = (
                    (y_coords >= p2) & (y_coords < H - p2) &
                    (x_coords >= p2) & (x_coords < W - p2)
                ).to(dtype=x.dtype)
                pml_m = pml_m.expand(B, 1, H, W)
            else:
                pml_m = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype, requires_grad=False)
        else:
            pml_m = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype, requires_grad=False)

        # Source mask (optional)
        src_mask = x[:, 3:4] if C >= 4 else None

        # k0 from wavelength (units consistent if dx,dy,lambda in same length unit)
        k0 = None
        if lambda_um is not None:
            k0 = (2.0 * math.pi) / lambda_um.view(-1)

        # physics gate (float or tensor)
        phys_gate_is_tensor = torch.is_tensor(phys_gate)

        # ------------------------------------------------------------------
        # Physics features (Helmholtz residual), only if enabled.
        # ------------------------------------------------------------------
        if self.enable_physics_features and (phys_gate_is_tensor or float(phys_gate) > 0.0):
            Hr, Hi = self.helmholtz(Er, Ei, eps, k0=k0)

            # If a source exists, do not force residual to zero inside source region.
            if src_mask is not None:
                Hr = Hr * (1.0 - src_mask)
                Hi = Hi * (1.0 - src_mask)

            # masked magnitude-based normalization (avoid PML dominating)
            den = pml_m.float().sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            mag = torch.sqrt(Hr * Hr + Hi * Hi + 1e-8)
            phys_scale = (mag * pml_m).sum(dim=(2, 3), keepdim=True) / den
            phys_scale = phys_scale.clamp_min(1e-6).detach()

            Hr = (Hr / phys_scale) * pml_m
            Hi = (Hi / phys_scale) * pml_m

            # time-gate physics (ramp on after t > 0.2)
            if t is not None and torch.is_tensor(t):
                t4 = t.view(-1, 1, 1, 1)
                t_gate = torch.clamp((t4 - 0.2) / 0.4, 0.0, 1.0)
                Hr = Hr * t_gate.to(dtype=Hr.dtype)
                Hi = Hi * t_gate.to(dtype=Hi.dtype)

            # apply phys gate
            if phys_gate_is_tensor:
                pg = phys_gate.view(-1, 1, 1, 1).to(device=x.device, dtype=Hr.dtype)
                Hr = Hr * pg
                Hi = Hi * pg
            else:
                pg = float(phys_gate)
                Hr = Hr * pg
                Hi = Hi * pg
        else:
            Hr = None
            Hi = None

        # ------------------------------------------------------------------
        # Build complex UNet input
        # IMPORTANT FIX: Only concatenate physics channels if enabled.
        # ------------------------------------------------------------------
        if self.enable_physics_features:
            if Hr is None:
                Hr = torch.zeros_like(Er)
                Hi = torch.zeros_like(Ei)
            field_r = torch.cat([Er_norm, Hr], dim=1)
            field_i = torch.cat([Ei_norm, Hi], dim=1)
        else:
            field_r = Er_norm
            field_i = Ei_norm

        field_r = field_r * pml_m
        field_i = field_i * pml_m

        if self.aux_encoder is not None:
            aux_r, aux_i = self.aux_encoder(aux_real * pml_m)
            field_r = torch.cat([field_r, aux_r], dim=1)
            field_i = torch.cat([field_i, aux_i], dim=1)

        # Time embedding
        t_emb = self._timestep_embedding(t, self.model_channels).to(x.dtype)
        if cond is not None:
            t_emb = torch.cat([t_emb, cond], dim=-1)
        emb = self.time_embed(t_emb)

        def _maybe_ckpt(fn, *args):
            if not self.use_checkpoint:
                return fn(*args)
            return _ckpt(fn, *args, use_reentrant=False)

        # Stem
        hr, hi = _maybe_ckpt(self.input_stem, field_r, field_i)

        # Encoder
        skips_r, skips_i = [hr], [hi]
        for block in self.input_blocks:
            for layer in block:
                if isinstance(layer, ComplexResBlock):
                    hr, hi = _maybe_ckpt(lambda a, b, e, layer=layer: layer(a, b, e), hr, hi, emb)
                elif isinstance(layer, (ComplexAttentionBlock, ComplexDownsample)):
                    hr, hi = _maybe_ckpt(lambda a, b, layer=layer: layer(a, b), hr, hi)
                else:
                    raise ValueError(f"Unknown layer type: {type(layer)}")
            skips_r.append(hr)
            skips_i.append(hi)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ComplexResBlock):
                hr, hi = _maybe_ckpt(lambda a, b, e, layer=layer: layer(a, b, e), hr, hi, emb)
            else:
                hr, hi = _maybe_ckpt(lambda a, b, layer=layer: layer(a, b), hr, hi)

        # Decoder
        for block in self.output_blocks:
            sr, si = skips_r.pop(), skips_i.pop()
            hr = torch.cat([hr, sr], dim=1)
            hi = torch.cat([hi, si], dim=1)

            for layer in block:
                if isinstance(layer, ComplexResBlock):
                    hr, hi = _maybe_ckpt(lambda a, b, e, layer=layer: layer(a, b, e), hr, hi, emb)
                elif isinstance(layer, (ComplexAttentionBlock, ComplexUpsample)):
                    hr, hi = _maybe_ckpt(lambda a, b, layer=layer: layer(a, b), hr, hi)

        # Output
        hr, hi = self.out_norm(hr, hi)
        hr, hi = self.out_act(hr, hi)
        vr, vi = _maybe_ckpt(self.out_conv, hr, hi)

        v = torch.cat([vr, vi], dim=1)
        u_t_pred = v * pml_m.to(dtype=v.dtype)

        if not return_sparams:
            return u_t_pred

        if self.sparam_head is None:
            raise RuntimeError("return_sparams=True but sparam_head is disabled on this model.")
        if aux is None:
            raise ValueError("return_sparams=True requires aux dict (for port masks / in_port).")

        # implied x1 under same FM path used in loss
        fields_t = x[:, 0:2]
        t4 = t
        if t4.dim() == 1:
            t4 = t4.view(-1, 1, 1, 1)
        elif t4.dim() == 2:
            t4 = t4.view(-1, 1, 1, 1)
        a = (1.0 - float(sig_min))
        b = (1.0 - a * t4)
        x1_fields_pred = a * fields_t + b * u_t_pred

        dev_type = x.device.type
        with torch.autocast(dev_type, enabled=False):
            x1f = x1_fields_pred.to(dtype=torch.float32)
            if getattr(self, "normalize_inputs", False):
                Er_phys = x1f[:, 0] * self.ez_real_std.to(dtype=torch.float32) + self.ez_real_mean.to(dtype=torch.float32)
                Ei_phys = x1f[:, 1] * self.ez_imag_std.to(dtype=torch.float32) + self.ez_imag_mean.to(dtype=torch.float32)
            else:
                Er_phys = x1f[:, 0]
                Ei_phys = x1f[:, 1]
            E_pred_phys_raw = torch.complex(Er_phys, Ei_phys)
            S_pred = self.predict_sparams(E_pred_phys_raw, aux=aux, cond=cond)

        return u_t_pred, S_pred

    def _timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def predict_sparams(self, E_pred_phys_raw: torch.Tensor, aux: dict, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict S-parameters using the improved learned head.

        The ImprovedSParamHead uses:
        1. Phase gradient features (to infer propagation direction)
        2. Amplitude features at each port
        3. Input port indicator
        4. Conditioning (wavelength)

        This allows the network to learn the mapping from field patterns to
        S-parameters without needing explicit forward/backward mode separation.
        """
        if self.sparam_head is None:
            raise RuntimeError(
                "S-parameter head is disabled on this model (enable_sparam_head=False). "
                "Enable it by running with --sparam-mode head and --lambda-sparam > 0."
            )

        B = E_pred_phys_raw.shape[0]
        device = E_pred_phys_raw.device

        pm = aux.get("port_masks", None)
        if pm is None:
            # No port masks - return zeros
            return torch.zeros((B, self.max_ports), device=device, dtype=torch.complex64)

        if pm.dim() == 3:
            pm = pm.unsqueeze(0).expand(B, -1, -1, -1)

        # Get input port index
        in_idx = aux.get("in_port_idx", None)
        if in_idx is None:
            in_idx = torch.zeros((B,), device=device, dtype=torch.long)
        elif torch.is_tensor(in_idx):
            in_idx = in_idx.to(device=device, dtype=torch.long).view(-1)
            if in_idx.numel() == 1:
                in_idx = in_idx.expand(B)
        else:
            in_idx = torch.full((B,), int(in_idx), device=device, dtype=torch.long)

        # Prepare conditioning
        if cond is None or self.cond_dim <= 0:
            cond_feat = torch.zeros((B, max(1, int(self.cond_dim))), device=device, dtype=torch.float32)
        else:
            cond_feat = cond.to(device=device, dtype=torch.float32)

        # Call the improved head - it handles all the feature extraction internally
        S_pred = self.sparam_head(
            E=E_pred_phys_raw,
            port_masks=pm,
            in_port_idx=in_idx,
            cond=cond_feat,
        )

        return S_pred


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ComplexPhysicsUNet(
        in_channels=5,   # Er, Ei, eps, src_mask, cond_channel
        out_channels=2,  # enforced
        model_channels=32,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        # attention_resolutions interpreted as ds factors (1,2,4,8,...).
        # For this depth, ds hits 1,2,4,8. Use (4,8) if you want attention to execute.
        attention_resolutions=(4, 8),
        dropout=0.0,
        num_heads=4,
        cond_dim=1,
        dx=0.0417,
        dy=0.0417,
        omega=2 * 3.14159 / 1.55,  # treated as k0 when lambda_um is None
        enable_physics_features=True,
        use_checkpoint=False,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"ComplexPhysicsUNet: {total/1e6:.2f}M parameters")

    B, H, W = 2, 128, 128
    x = torch.randn(B, 5, H, W, device=device)
    t = torch.rand(B, device=device)
    cond = torch.rand(B, 1, device=device)

    with torch.no_grad():
        v = model(x, t, cond=cond)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {v.shape}")
    print("✓ Forward pass successful!")
