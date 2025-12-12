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

        q = (q * scale).view(bs * self.n_heads, ch, length)
        k = (k * scale).view(bs * self.n_heads, ch, length)
        v = v.view(bs * self.n_heads, ch, length)

        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.view(bs, -1, length)


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
    Fixed finite-difference operators for 2D

    Assumes uniform grid spacing dx, dy
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        super().__init__()
        self.dx = dx
        self.dy = dy

        # first derivatives
        kx = torch.tensor(
            [[0, 0, 0],
             [-0.5, 0.0, 0.5],
             [0, 0, 0]],
            dtype=torch.float32,
        ) / dx

        ky = torch.tensor(
            [[0, -0.5, 0],
             [0,  0.0, 0],
             [0,  0.5, 0]],
            dtype=torch.float32,
        ) / dy

        # 5-point Laplacian (assuming dx == dy; good for your grids)
        klap = torch.tensor(
            [[0.0,  1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0,  1.0, 0.0]],
            dtype=torch.float32,
        ) / (dx * dx)

        self.register_buffer("kx",   kx[None, None, :, :])
        self.register_buffer("ky",   ky[None, None, :, :])
        self.register_buffer("klap", klap[None, None, :, :])

    def diff_x(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = self.kx.expand(C, 1, 3, 3)
        return F.conv2d(x, weight, padding=1, groups=C)

    def diff_y(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = self.ky.expand(C, 1, 3, 3)
        return F.conv2d(x, weight, padding=1, groups=C)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D Laplacian ∇²x using a 5-point stencil.
        """
        C = x.shape[1]
        weight = self.klap.expand(C, 1, 3, 3)
        return F.conv2d(x, weight, padding=1, groups=C)

class HelmholtzResidual2D(nn.Module):
    """
    Scalar Helmholtz residual for Ez in 2D:

        (∇² + k0^2 * eps) Ez = 0

    We assume Ez is complex, stored as [Re(Ez), Im(Ez)].
    Returns [Re(R), Im(R)] as physics feature channels.
    """

    def __init__(self, dx: float, dy: float, omega: float, c0: float = 1.0):
        super().__init__()
        self.diff = FiniteDiff2D(dx, dy)
        # in normalized units you can treat k0 = omega / c0
        self.k0 = omega / c0

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, 2, H, W]  -> [Re(Ez), Im(Ez)]
        eps: [B, 1, H, W]
        returns: [B, 2, H, W]  -> [Re(R), Im(R)]
        """
        Ez_r = x[:, 0:1]
        Ez_i = x[:, 1:2]

        lap_r = self.diff.laplacian(Ez_r)
        lap_i = self.diff.laplacian(Ez_i)

        k0_sq_eps = (self.k0 ** 2) * eps

        # R = (∇² + k0^2 eps) Ez
        R_r = lap_r + k0_sq_eps * Ez_r
        R_i = lap_i + k0_sq_eps * Ez_i

        return torch.cat([R_r, R_i], dim=1)   # [B, 2, H, W]


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
        dx: float = 1.0,
        dy: float = 1.0,
        omega: float = 1.0,
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

        # Physics module
        self.helmholtz = HelmholtzResidual2D(dx=dx, dy=dy, omega=omega)
        self.n_phys_feats = 2 # R1, R2, R3 real/imag

        # effective input channels seen by the UNet
        self.base_in_channels = in_channels
        self.in_channels = in_channels + self.n_phys_feats
        

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
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, 3, H, W] = [Re(Ez), Im(Ez), eps]
        """
        # Split fields and geometry
        Ez = x[:, 0:2]    # [B, 2, H, W]
        eps = x[:, 2:3]   # [B, 1, H, W]

        # physics features from scalar Helmholtz residual
        phys_feats = self.helmholtz(Ez, eps)          # [B, 2, H, W]
        x_aug = torch.cat([x, phys_feats], dim=1)     # [B, 5, H, W]

        # timestep embedding (unchanged)
        t_emb = timestep_embedding(t, self.model_channels)  # [B, model_channels]
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(x.shape[0], self.cond_dim, device=x.device, dtype=x.dtype)
            emb_input = torch.cat([t_emb, cond], dim=-1)
        else:
            emb_input = t_emb
        emb = self.time_embed(emb_input)

        h = x_aug.type(self.dtype)
        hs: List[torch.Tensor] = []

        # down path
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # middle
        h = self.middle_block(h, emb)

        # up path
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)

