# physics_unet.py
import math
from abc import abstractmethod
from typing import Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper functions (minimal versions of guided-diffusion.nn utilities)
# ---------------------------------------------------------------------------

def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Return a Conv layer for 1D / 2D / 3D."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dims={dims}")


def linear(in_dim: int, out_dim: int) -> nn.Linear:
    return nn.Linear(in_dim, out_dim)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dims={dims}")


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize the weights of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def normalization(channels: int) -> nn.Module:
    """Simple GroupNorm used in many UNets."""
    num_groups = min(32, channels)
    # Ensure num_groups divides channels (torch requirement). Walk down until it fits.
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings as in guided-diffusion.

    timesteps: (B,) either int or float
    returns:   (B, dim)
    """
    half = dim // 2
    timesteps = timesteps.float()
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def checkpoint(fn, inputs, params, use_checkpoint: bool):
    """
    Minimal wrapper: currently just runs fn directly.

    If you want real gradient checkpointing later, replace with:
    torch.utils.checkpoint.checkpoint(fn, *inputs)
    """
    return fn(*inputs)


# ---------------------------------------------------------------------------
# Timestep-aware base class and sequential container
# ---------------------------------------------------------------------------


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Upsample / Downsample blocks
# ---------------------------------------------------------------------------


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    If dims == 3, only upsamples spatial dims (H, W), not the first (e.g. time).
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    If dims == 3, only downsamples spatial dims (H, W).
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------------------------------------------------------------------------
# ResBlock with timestep embedding
# ---------------------------------------------------------------------------


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    This is the main building block for the UNet.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        # expand to (B, C, 1, 1, ...)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------


class QKVAttention(nn.Module):
    """
    A simple multi-head self-attention over flattened spatial positions.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        qkv: (B, 3*H*C, T) where T = spatial positions, H = heads
        returns: (B, H*C, T)
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        # NOTE: use reshape (not view) because chunked tensors may be non-contiguous.
        q = (q * scale).reshape(bs * self.n_heads, ch, length)
        k = (k * scale).reshape(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)

        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    Spatial self-attention over feature maps.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x_in = x
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        h = h.reshape(b, c, *spatial)
        return x_in + h
    
class FiniteDiff2D(nn.Module):
    """
    Fixed finite-difference operators for 2D on a uniform grid.

    dx, dy are the physical spacing per pixel (in your length units).
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, pad_mode: str = "replicate"):
        super().__init__()
        self.dx = float(dx)
        self.dy = float(dy)
        self.pad_mode = pad_mode

        # Central differences:
        # d/dx ~ (f[i,j+1] - f[i,j-1])/(2 dx)
        kx = torch.tensor(
            [[0.0,  0.0, 0.0],
             [-0.5, 0.0, 0.5],
             [0.0,  0.0, 0.0]],
            dtype=torch.float32,
        ) / self.dx

        # d/dy ~ (f[i+1,j] - f[i-1,j])/(2 dy)
        ky = torch.tensor(
            [[0.0, -0.5, 0.0],
             [0.0,  0.0, 0.0],
             [0.0,  0.5, 0.0]],
            dtype=torch.float32,
        ) / self.dy

        # Anisotropic 5-point Laplacian:
        # ∇²f ~ (f[i,j+1]-2f[i,j]+f[i,j-1])/dx^2 + (f[i+1,j]-2f[i,j]+f[i-1,j])/dy^2
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        klap = torch.tensor(
            [[0.0,     inv_dy2, 0.0],
             [inv_dx2, -2.0*(inv_dx2 + inv_dy2), inv_dx2],
             [0.0,     inv_dy2, 0.0]],
            dtype=torch.float32,
        )

        self.register_buffer("kx",   kx[None, None, :, :])
        self.register_buffer("ky",   ky[None, None, :, :])
        self.register_buffer("klap", klap[None, None, :, :])

    def _depthwise_conv3(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        # IMPORTANT (autograd safety under heavy unrolling/DDP):
        # Use a fresh, contiguous kernel tensor per call to avoid any in-place versioning
        # issues with views of registered buffers being saved for backward.
        weight = k.detach().clone().repeat(C, 1, 1, 1)

        # Explicit padding (avoids implicit zero outside the domain)
        xpad = F.pad(x, (1, 1, 1, 1), mode=self.pad_mode)
        return F.conv2d(xpad, weight, padding=0, groups=C)

    def diff_x(self, x: torch.Tensor) -> torch.Tensor:
        return self._depthwise_conv3(x, self.kx)

    def diff_y(self, x: torch.Tensor) -> torch.Tensor:
        return self._depthwise_conv3(x, self.ky)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return self._depthwise_conv3(x, self.klap)


class HelmholtzResidual2D(nn.Module):
    def __init__(self, dx: float, dy: float, omega: float, c0: float = 1.0, pml_cells: int = 0, normalize: bool = False):
        super().__init__()
        self.diff = FiniteDiff2D(dx, dy)
        self.k0 = omega / c0
        self.pml_cells = int(pml_cells)
        self.normalize = bool(normalize)

    def forward(self, x: torch.Tensor, eps: torch.Tensor, k0: Optional[torch.Tensor] = None) -> torch.Tensor:
        Ez_r, Ez_i = x[:, 0:1], x[:, 1:2]
        lap_r = self.diff.laplacian(Ez_r)
        lap_i = self.diff.laplacian(Ez_i)

        if k0 is None:
            k0_sq_eps = (self.k0 ** 2) * eps
        else:
            k0 = k0.view(-1, 1, 1, 1).to(device=eps.device, dtype=eps.dtype)
            k0_sq_eps = (k0 ** 2) * eps

        R_r = lap_r + k0_sq_eps * Ez_r
        R_i = lap_i + k0_sq_eps * Ez_i
        R = torch.cat([R_r, R_i], dim=1)

        # mask PML band (+2 margin)
        p = self.pml_cells
        if p > 0:
            B, C, H, W = R.shape
            p2 = min(p + 4, max(0, H // 2 - 1), max(0, W // 2 - 1))
            if p2 > 0:
                # Create mask without inplace operations for autograd safety
                y_coords = torch.arange(H, device=R.device, dtype=torch.float32).view(1, 1, H, 1)
                x_coords = torch.arange(W, device=R.device, dtype=torch.float32).view(1, 1, 1, W)
                m = ((y_coords >= p2) & (y_coords < H - p2) & 
                     (x_coords >= p2) & (x_coords < W - p2)).to(dtype=R.dtype)
                m = m.expand(B, 1, H, W)
                R = R * m

        if self.normalize:
            denom = R.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            R = R / denom

        return R

class PhaseFeatures2D(nn.Module):
    """
    Phase-aware, amplitude-robust features derived from complex field E = Er + j Ei.

    Outputs (B, 8, H, W):
      LOCAL PHASE (5 channels):
        [u_r, u_i, log|E|, dphi_dx, dphi_dy]
      GLOBAL PHASE (3 channels):
        [coarse_phi_cos, coarse_phi_sin, relative_phase]
      
    where u = E / (|E| + eps) and
      dphi = Im(conj(E) * dE) / (|E|^2 + eps)
      coarse_phi = low-frequency phase envelope (captures global phase structure)
      relative_phase = phase offset from source reference region
    """
    N_FEATURES = 8  # class constant for external access
    
    def __init__(self, dx: float, dy: float, pad_mode: str = "replicate", eps: float = 1e-8,
                 coarse_pool_size: int = 8):
        super().__init__()
        self.diff = FiniteDiff2D(dx=dx, dy=dy, pad_mode=pad_mode)
        self.eps = float(eps)
        self.coarse_pool_size = int(coarse_pool_size)

    def forward(self, E: torch.Tensor, source_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            E: (B, 2, H, W) => [Er, Ei]
            source_mask: optional (B, 1, H, W) or (B, H, W) mask indicating source region for reference phase
        Returns:
            (B, 8, H, W) phase features
        """
        Er = E[:, 0:1]
        Ei = E[:, 1:2]
        B, _, H, W = Er.shape

        # amplitude
        A2 = Er * Er + Ei * Ei
        A  = torch.sqrt(A2 + self.eps)

        # unit phasor (cos phi, sin phi) - LOCAL absolute phase
        invA = 1.0 / (A + self.eps)
        u_r = Er * invA
        u_i = Ei * invA

        # log amplitude (stabilizes learning near small |E|)
        logA = torch.log(A + self.eps)

        # ============================================================
        # LOCAL PHASE: spatial derivatives (wavevector)
        # ============================================================
        dEr_dx = self.diff.diff_x(Er)
        dEi_dx = self.diff.diff_x(Ei)
        dEr_dy = self.diff.diff_y(Er)
        dEi_dy = self.diff.diff_y(Ei)

        # Im(conj(E) * dE) = Er*dEi - Ei*dEr
        imag_conjE_dE_dx = Er * dEi_dx - Ei * dEr_dx
        imag_conjE_dE_dy = Er * dEi_dy - Ei * dEr_dy

        invA2 = 1.0 / (A2 + self.eps)
        dphi_dx = imag_conjE_dE_dx * invA2
        dphi_dy = imag_conjE_dE_dy * invA2

        kmax = 50.0  # tune; start 20–100
        dphi_dx = torch.clamp(dphi_dx, -kmax, kmax)
        dphi_dy = torch.clamp(dphi_dy, -kmax, kmax)

        # ============================================================
        # GLOBAL PHASE: coarse low-frequency phase structure
        # ============================================================
        # Compute phase angle
        phi = torch.atan2(Ei, Er)  # (B, 1, H, W)
        
        # Low-frequency phase envelope via pooling + upsampling
        # We pool cos/sin (not phi directly) to handle wrapping correctly
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        pool_size = min(self.coarse_pool_size, H, W)
        if pool_size > 1:
            cos_coarse = F.avg_pool2d(cos_phi, pool_size, stride=pool_size)
            sin_coarse = F.avg_pool2d(sin_phi, pool_size, stride=pool_size)
            # Upsample back to original resolution
            coarse_phi_cos = F.interpolate(cos_coarse, size=(H, W), mode='bilinear', align_corners=False)
            coarse_phi_sin = F.interpolate(sin_coarse, size=(H, W), mode='bilinear', align_corners=False)
        else:
            coarse_phi_cos = cos_phi
            coarse_phi_sin = sin_phi

        # ============================================================
        # GLOBAL PHASE: relative phase from source reference
        # ============================================================
        if source_mask is not None:
            # Ensure mask is (B, 1, H, W)
            if source_mask.dim() == 3:
                source_mask = source_mask.unsqueeze(1)
            sm = source_mask.to(dtype=Er.dtype, device=Er.device)
            
            # Weighted average of cos/sin in source region (handles wrapping)
            denom = sm.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            ref_cos = (cos_phi * sm).sum(dim=(2, 3), keepdim=True) / denom  # (B,1,1,1)
            ref_sin = (sin_phi * sm).sum(dim=(2, 3), keepdim=True) / denom
            
            # Reference phase angle
            ref_phi = torch.atan2(ref_sin, ref_cos)  # (B,1,1,1)
            
            # Relative phase = phi - ref_phi, wrapped to [-pi, pi]
            rel_phi = phi - ref_phi
            # Wrap to [-pi, pi]
            relative_phase = torch.atan2(torch.sin(rel_phi), torch.cos(rel_phi))
        else:
            # No source mask: use global mean phase as reference
            mean_cos = cos_phi.mean(dim=(2, 3), keepdim=True)
            mean_sin = sin_phi.mean(dim=(2, 3), keepdim=True)
            ref_phi = torch.atan2(mean_sin, mean_cos)
            rel_phi = phi - ref_phi
            relative_phase = torch.atan2(torch.sin(rel_phi), torch.cos(rel_phi))
        
        # Normalize relative_phase to [-1, 1] for network input
        relative_phase = relative_phase / math.pi

        return torch.cat([
            u_r, u_i, logA,              # amplitude & local phase (3)
            dphi_dx, dphi_dy,             # local wavevector (2)
            coarse_phi_cos, coarse_phi_sin,  # global phase structure (2)
            relative_phase,               # phase relative to source (1)
        ], dim=1)



# ---------------------------------------------------------------------------
# PhysicsUNet backbone (UNet with time + optional conditioning)
# ---------------------------------------------------------------------------


class PhysicsUNet(nn.Module):
    """
    UNet backbone with timestep + optional conditioning embedding.

    Use this as your backbone velocity model in flow matching / PBFM:

        v_theta = PhysicsUNet(x_t, t, cond)

    - x_t:   [B, C_in, H, W]  (e.g. fields and maybe eps channels)
    - t:     [B]              (continuous "time" in [0,1] or [0,T])
    - cond:  [B, cond_dim]    (wavelength, device params, etc., optional)
    - output: [B, C_out, H, W] (e.g. velocity over the state)
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
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        cond_dim: int = 0,  # additional conditioning vector dim
        enable_sparam_head: bool = False,
        dx: float = 1.0,
        dy: float = 1.0,
        omega: float = 1.0,
        # Default to 0 because newer datasets are typically saved already-cropped to the non-PML window.
        pml_cells: int = 0,
        # Ablation flag: set False to disable all embedded physics features (Helmholtz + phase)
        # and train a vanilla UNet for comparison.
        enable_physics_features: bool = True,
    ):
        super().__init__()

        if dims != 2:
            raise ValueError("PhysicsUNet is currently implemented for 2D only.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = set(attention_resolutions)
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        # Ablation: enable/disable physics features (Helmholtz residual + phase features)
        self.enable_physics_features = bool(enable_physics_features)
        # S-parameter head (optional). IMPORTANT: if this head exists but S-param loss is disabled,
        # DDP will error (unused parameters) when find_unused_parameters=False. So we only create
        # the head when explicitly enabled by training.
        self.enable_sparam_head = bool(enable_sparam_head)
        # We pad aux to max_ports=4 in the dataset.
        self.max_ports = 4
        if self.enable_sparam_head:
            # Use ImprovedSParamHead with phase gradient features for better S-param prediction
            from sparam_head import ImprovedSParamHead
            self.sparam_head = ImprovedSParamHead(
                max_ports=self.max_ports,
                cond_dim=int(cond_dim),
                hidden_dim=max(256, 2 * model_channels),
                num_layers=4,
                dropout=0.1,
            )
        else:
            self.sparam_head = None

        # Physics module (Helmholtz residual features)
        self.helmholtz = HelmholtzResidual2D(dx=dx, dy=dy, omega=omega, pml_cells=int(pml_cells))
        self.n_phys_feats = 2  # Helmholtz residual real/imag

        # Phase features module (local + global phase)
        self.phase_features = PhaseFeatures2D(dx=dx, dy=dy, pad_mode="replicate", eps=1e-8)
        self.n_phase_feats = PhaseFeatures2D.N_FEATURES  # 8: local (5) + global (3)

        # effective input channels seen by the UNet
        self.base_in_channels = in_channels
        if self.enable_physics_features:
            self.in_channels = in_channels + self.n_phys_feats + self.n_phase_feats
        else:
            # Vanilla UNet: no extra physics channels
            self.in_channels = in_channels
        
        # time + conditioning embedding
        time_embed_dim = model_channels * 4
        # input to this MLP is [t_embed (model_channels) ; cond (cond_dim)]
        self.time_embed = nn.Sequential(
            linear(model_channels + cond_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # ---- input blocks ----
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(conv_nd(dims, self.in_channels, model_channels, 3, padding=1))
        )
        ch = model_channels
        input_block_chans: List[int] = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                # downsample
                if resblock_updown:
                    down = ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                else:
                    down = Downsample(ch, conv_resample, dims=dims, out_channels=ch)

                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_chans.append(ch)
                ds *= 2

        # ---- middle block ----
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ---- output blocks ----
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    # upsample
                    if resblock_updown:
                        up = ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                    else:
                        up = Upsample(ch, conv_resample, dims=dims, out_channels=ch)
                    layers.append(up)
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, min(out_channels, 2), 3, padding=1)),
        )

        # Joint training: separate eps velocity head (real-valued)
        self.joint_eps_out = None
        if out_channels == 3:
            self.joint_eps_out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, ch, 1, 3, padding=1)),
            )

        # --- normalization flags + stats buffers (set later from dataset stats) ---
        self.normalize_inputs = True   # set True if your x is normalized in the dataloader
        self.normalize_eps = False     # will be set from args.normalize_eps

        # Defaults; will be overwritten by set_normalization_stats(...)
        self.register_buffer("ez_real_mean", torch.tensor(0.0))
        self.register_buffer("ez_real_std",  torch.tensor(1.0))
        self.register_buffer("ez_imag_mean", torch.tensor(0.0))
        self.register_buffer("ez_imag_std",  torch.tensor(1.0))
        self.register_buffer("eps_mean",     torch.tensor(0.0))
        self.register_buffer("eps_std",      torch.tensor(1.0))
    
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

        Args:
          E_pred_phys_raw: complex tensor [B,H,W] (no label-based phase alignment)
          aux: dict with at least 'port_masks' (float [B,P,H,W] or [P,H,W]) and optional 'in_port_idx'
          cond: optional [B,cond_dim]

        Returns:
          complex tensor [B,max_ports]
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

    @torch.no_grad()
    def set_normalization_stats(self, stats: dict, normalize_eps: bool = True):
        """Set normalization statistics from dataset stats dict."""
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
        phys_gate=1.0,
        phase_gate=1.0,
        # S-param head integration (DDP-safe): compute S_pred *inside* forward when requested.
        return_sparams: bool = False,
        aux: Optional[dict] = None,
        sig_min: float = 0.0,
    ):
        # x: [B,4,H,W] = [Re(Ez), Im(Ez), eps, src] (typically normalized)
        Ez  = x[:, 0:2]
        eps = x[:, 2:3]

        # extra_maps = x[:, 3:self.base_in_channels]

        # per-sample k0 from physical lambda_um (um)
        k0 = None
        if lambda_um is not None:
            k0 = (2.0 * math.pi) / lambda_um.view(-1)  # [B]

        # de-normalize for physics computation (if inputs are normalized)
        Ez_phys = Ez
        eps_phys = eps
        if getattr(self, "normalize_inputs", False):
            # Denormalize without inplace operations for autograd safety
            Ez_phys = torch.stack([
                Ez[:, 0] * self.ez_real_std + self.ez_real_mean,
                Ez[:, 1] * self.ez_imag_std + self.ez_imag_mean,
            ], dim=1)

            eps_phys = eps.clone()
            if getattr(self, "normalize_eps", False):
                eps_phys = eps_phys * self.eps_std + self.eps_mean
        
        # build non-PML mask (same logic as elsewhere)
        pml_m = torch.ones(
            (x.shape[0], 1, Ez_phys.shape[-2], Ez_phys.shape[-1]),
            device=x.device,
            dtype=Ez_phys.dtype,
        )
        p = int(self.helmholtz.pml_cells)
        # If pml_cells == 0, do NOT mask any border. (Datasets already cropped to non-PML should use pml_cells=0.)
        p2 = 0
        if p > 0:
            p2 = min(
                p + 4,
                max(0, Ez_phys.shape[-2] // 2 - 1),
                max(0, Ez_phys.shape[-1] // 2 - 1),
            )
        if p2 > 0:
            # Create mask without inplace operations for autograd safety
            H_phys, W_phys = Ez_phys.shape[-2], Ez_phys.shape[-1]
            y_coords = torch.arange(H_phys, device=x.device, dtype=torch.float32).view(1, 1, H_phys, 1)
            x_coords = torch.arange(W_phys, device=x.device, dtype=torch.float32).view(1, 1, 1, W_phys)
            pml_m = ((y_coords >= p2) & (y_coords < H_phys - p2) & 
                    (x_coords >= p2) & (x_coords < W_phys - p2)).to(dtype=Ez_phys.dtype)
            pml_m = pml_m.expand(x.shape[0], 1, H_phys, W_phys)

        B, _, H, W = x.shape

        # masked scale (avoid PML dominating)
        den = pml_m.float().sum(dim=(2,3), keepdim=True).clamp_min(1.0)

        # Extract source mask for global phase reference (channel 3 if available)
        source_mask = None
        if self.base_in_channels >= 4:
            source_mask = x[:, 3:4]  # (B, 1, H, W)

        # =====================================================
        # ABLATION: Skip physics features for vanilla UNet
        # =====================================================
        if not self.enable_physics_features:
            # Vanilla UNet: just use the raw input (no physics/phase features)
            x_masked = x * pml_m.to(dtype=x.dtype)
            x_aug = x_masked
        else:
            # --------------------
            # Physics features (gated)
            # --------------------
            if phys_gate is None:
                phys_gate = 1.0
            phys_gate_f = float(phys_gate) if not torch.is_tensor(phys_gate) else phys_gate

            if (not torch.is_tensor(phys_gate_f) and phys_gate_f <= 0.0):
                phys_feats = torch.zeros((B, self.n_phys_feats, H, W), device=x.device, dtype=x.dtype)
            else:
                phys_feats = self.helmholtz(Ez_phys, eps_phys, k0=k0)  # (B,2,H,W)
                phys_scale = (phys_feats.abs() * pml_m).sum(dim=(2,3), keepdim=True) / den
                phys_scale = phys_scale.clamp_min(1e-6)
                phys_scale = phys_scale.detach()
                phys_feats = (phys_feats / phys_scale) * pml_m  # mask

                # Time-gate physics features so we don't feed "physics of pure noise" to the UNet.
                # For FM, t=0 is x0 (noise) and t=1 is ~x1 (clean). Ramp physics on later in time.
                if torch.is_tensor(t):
                    t4 = t
                    if t4.dim() == 2:      # [B,1] -> [B,1,1,1]
                        t4 = t4.view(-1, 1, 1, 1)
                    elif t4.dim() == 1:    # [B] -> [B,1,1,1]
                        t4 = t4.view(-1, 1, 1, 1)
                    phys_t_gate = torch.clamp((t4 - 0.2) / 0.4, 0.0, 1.0)
                    phys_feats = phys_feats * phys_t_gate.to(dtype=phys_feats.dtype)

                # apply gate (broadcastable)
                if torch.is_tensor(phys_gate_f):
                    while phys_gate_f.dim() < 4:
                        phys_gate_f = phys_gate_f.view(-1, 1, 1, 1)
                    phys_feats = phys_feats * phys_gate_f.to(device=x.device, dtype=phys_feats.dtype)
                else:
                    phys_feats = phys_feats * float(phys_gate_f)

                phys_feats = phys_feats.to(dtype=x.dtype)

            # --------------------
            # Phase features (gated) - now with global phase support
            # --------------------
            if phase_gate is None:
                phase_gate = 1.0
            phase_gate_f = float(phase_gate) if not torch.is_tensor(phase_gate) else phase_gate

            if (not torch.is_tensor(phase_gate_f) and phase_gate_f <= 0.0):
                phase_feats = torch.zeros((B, self.n_phase_feats, H, W), device=x.device, dtype=x.dtype)
            else:
                # Pass source_mask for global phase reference computation
                phase_feats = self.phase_features(Ez_phys, source_mask=source_mask)  # (B,8,H,W)
                phase_scale = (phase_feats.abs() * pml_m).sum(dim=(2,3), keepdim=True) / den
                phase_scale = phase_scale.clamp_min(1e-6)
                phase_scale = phase_scale.detach()
                phase_feats = (phase_feats / phase_scale) * pml_m

                # OPTIONAL: time-gate phase features so "phase of noise" doesn't dominate.
                # t is usually shaped [B,1,1,1] in your training code.
                # This ramps phase features on for t >= ~0.2 and fully on by t~0.6.
                if torch.is_tensor(t):
                    t4 = t
                    if t4.dim() == 2:      # [B,1] -> [B,1,1,1]
                        t4 = t4.view(-1, 1, 1, 1)
                    elif t4.dim() == 1:    # [B] -> [B,1,1,1]
                        t4 = t4.view(-1, 1, 1, 1)
                    t_gate = torch.clamp((t4 - 0.2) / 0.4, 0.0, 1.0)
                    phase_feats = phase_feats * t_gate.to(dtype=phase_feats.dtype)

                # apply gate
                if torch.is_tensor(phase_gate_f):
                    while phase_gate_f.dim() < 4:
                        phase_gate_f = phase_gate_f.view(-1, 1, 1, 1)
                    phase_feats = phase_feats * phase_gate_f.to(device=x.device, dtype=phase_feats.dtype)
                else:
                    phase_feats = phase_feats * float(phase_gate_f)

                phase_feats = phase_feats.to(dtype=x.dtype)

            # --------------------
            # Augmented input (keep channel count constant!)
            # --------------------
            x_masked = x * pml_m.to(dtype=x.dtype)
            x_aug = torch.cat([x_masked, phys_feats, phase_feats], dim=1)  # [B, in+2+8, H, W]


        # timestep embedding wants [B]
        t_in = t.view(t.shape[0]) if t.dim() > 1 else t
        t_emb = timestep_embedding(t_in, self.model_channels)

        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(x.shape[0], self.cond_dim, device=x.device, dtype=t_emb.dtype)
            emb_input = torch.cat([t_emb, cond.to(dtype=t_emb.dtype)], dim=-1)
        else:
            emb_input = t_emb

        emb = self.time_embed(emb_input)

        h = x_aug.type(self.dtype)
        hs: List[torch.Tensor] = []

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        out = self.out(h)

        # Joint training: append eps velocity channel
        if self.joint_eps_out is not None:
            v_eps = self.joint_eps_out(h)
            out = torch.cat([out, v_eps], dim=1)  # [B,3,H,W]

        u_t_pred = out * pml_m.to(dtype=out.dtype)

        if not return_sparams:
            return u_t_pred

        if self.sparam_head is None:
            raise RuntimeError("return_sparams=True but sparam_head is disabled on this model.")
        if aux is None:
            raise ValueError("return_sparams=True requires aux dict (for port masks / in_port).")

        # Compute the implied clean endpoint field x1 under the same FM path used in loss.
        # This keeps the head tied to the main forward graph (DDP-safe).
        # NOTE: x and u_t_pred are in *normalized* field units.
        fields_t = x[:, 0:2]
        t4 = t
        if t4.dim() == 1:
            t4 = t4.view(-1, 1, 1, 1)
        elif t4.dim() == 2:
            t4 = t4.view(-1, 1, 1, 1)
        a = (1.0 - float(sig_min))
        b = (1.0 - a * t4)
        x1_fields_pred = a * fields_t + b * u_t_pred  # [B,2,H,W] normalized

        # De-normalize to physical units for S-params. Run head math in fp32 for stability.
        dev_type = x.device.type
        with torch.autocast(dev_type, enabled=False):
            x1f = x1_fields_pred.to(dtype=torch.float32)
            if getattr(self, "normalize_inputs", False):
                Er = x1f[:, 0] * self.ez_real_std.to(dtype=torch.float32) + self.ez_real_mean.to(dtype=torch.float32)
                Ei = x1f[:, 1] * self.ez_imag_std.to(dtype=torch.float32) + self.ez_imag_mean.to(dtype=torch.float32)
            else:
                Er = x1f[:, 0]
                Ei = x1f[:, 1]
            E_pred_phys_raw = torch.complex(Er, Ei)  # [B,H,W]
            S_pred = self.predict_sparams(E_pred_phys_raw, aux=aux, cond=cond)

        return u_t_pred, S_pred



