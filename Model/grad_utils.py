import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from findiff import FinDiff
from torch.func import jacfwd, jacrev, vmap


def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # subtracting batch and pixel dimensions
    pattern = "b " + " ".join([f"c{i}" for i in range(num_dims)]) + " x y -> b (x y) " + " ".join([f"c{i}" for i in range(num_dims)])
    return rearrange(tensor, pattern)


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = "b (x y) " + " ".join([f"c{i}" for i in range(num_dims)]) + f" -> b " + " ".join([f"c{i}" for i in range(num_dims)]) + " x y"
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)


class StencilGradientComputation(nn.Module):
    """
    Warning: This is hard-coded for finite differences on images with 2nd order accuracy.
    """

    def __init__(self, stencils, periodic=False, device="cpu"):
        super(StencilGradientComputation, self).__init__()

        # identify max kernel size
        self.max_inner_offset = 0
        self.max_offset = 0
        for key, stencil in stencils.items():
            for (i, j), value in stencil.items():
                if key == ("C", "C"):
                    self.max_inner_offset = max(self.max_inner_offset, abs(i), abs(j))
                else:
                    self.max_offset = max(self.max_offset, abs(i), abs(j))
        self.max_inner_kernel_size = 2 * self.max_inner_offset + 1  # include center and in both directions
        self.max_kernel_size = 2 * self.max_offset + 1  # include center and in both directions

        self.kernels = {}
        mid_inner = self.max_inner_offset  # center of the kernel
        mid = self.max_offset  # center of the kernel
        for key, stencil in stencils.items():
            if key == ("C", "C"):
                kernel = torch.zeros((1, 1, self.max_inner_kernel_size, self.max_inner_kernel_size), device=device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid_inner + i, mid_inner + j] = value
            else:
                kernel = torch.zeros((1, 1, self.max_kernel_size, self.max_kernel_size), device=device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid + i, mid + j] = value
            self.kernels[key] = kernel

            self.periodic = periodic

    def forward(self, x):

        original_size = x.size()
        batch_size, *channels, height, width = original_size

        # flatten the channel dimensions
        x = x.view(batch_size, -1, height, width)
        channels = x.size(1)

        interior_kernel = self.kernels[("C", "C")]
        interior_kernel = interior_kernel.repeat((channels, 1, 1, 1))

        if self.periodic:
            # pad the image with the opposite boundary
            padding = (self.max_inner_offset, self.max_inner_offset, self.max_inner_offset, self.max_inner_offset)
            x = F.pad(x, padding, mode="circular")
            x_grads = F.conv2d(x, interior_kernel, groups=channels)
            return x_grads.view(original_size)

        interior_conv = F.conv2d(x, interior_kernel, groups=channels)

        # manually apply boundary stencils
        # we extend the image by max_offset since kernel is centered
        x_ext = F.pad(x, (self.max_offset, self.max_offset, self.max_offset, self.max_offset), mode="constant", value=0)

        # only consider the part of x that is at the boundary for the convolution (while being consistent with the convolution kernels)
        reduced_conv_offset = 2 * self.max_offset + self.max_inner_offset

        # top boundary
        top_kernel = self.kernels[("L", "C")]
        top_kernel = top_kernel.repeat((channels, 1, 1, 1))
        top_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, :], top_kernel, groups=channels)

        # bottom boundary
        bottom_kernel = self.kernels[("H", "C")]
        bottom_kernel = bottom_kernel.repeat((channels, 1, 1, 1))
        bottom_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, :], bottom_kernel, groups=channels)

        # left boundary
        left_kernel = self.kernels[("C", "L")]
        left_kernel = left_kernel.repeat((channels, 1, 1, 1))
        left_conv = F.conv2d(x_ext[:, :, :, 0:reduced_conv_offset], left_kernel, groups=channels)

        # right boundary
        right_kernel = self.kernels[("C", "H")]
        right_kernel = right_kernel.repeat((channels, 1, 1, 1))
        right_conv = F.conv2d(x_ext[:, :, :, -reduced_conv_offset:], right_kernel, groups=channels)

        # top-left corner
        tl_corner_kernel = self.kernels[("L", "L")]
        tl_corner_kernel = tl_corner_kernel.repeat((channels, 1, 1, 1))
        tl_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, 0:reduced_conv_offset], tl_corner_kernel, groups=channels)

        # top-right corner
        tr_corner_kernel = self.kernels[("L", "H")]
        tr_corner_kernel = tr_corner_kernel.repeat((channels, 1, 1, 1))
        tr_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, -reduced_conv_offset:], tr_corner_kernel, groups=channels)

        # bottom-left corner
        bl_corner_kernel = self.kernels[("H", "L")]
        bl_corner_kernel = bl_corner_kernel.repeat((channels, 1, 1, 1))
        bl_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, 0:reduced_conv_offset], bl_corner_kernel, groups=channels)

        # bottom-right corner
        br_corner_kernel = self.kernels[("H", "H")]
        br_corner_kernel = br_corner_kernel.repeat((channels, 1, 1, 1))
        br_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, -reduced_conv_offset:], br_corner_kernel, groups=channels)

        # combine the results from interior, boundaries, and corners
        x_grads = torch.zeros_like(x)
        x_grads[:, :, self.max_inner_offset : -self.max_inner_offset, self.max_inner_offset : -self.max_inner_offset] = interior_conv
        x_grads[:, :, 0 : self.max_inner_offset, :] = top_conv
        x_grads[:, :, -self.max_inner_offset :, :] = bottom_conv
        x_grads[:, :, :, 0 : self.max_inner_offset] = left_conv
        x_grads[:, :, :, -self.max_inner_offset :] = right_conv
        x_grads[:, :, 0 : self.max_inner_offset, 0 : self.max_inner_offset] = tl_corner_conv
        x_grads[:, :, 0 : self.max_inner_offset, -self.max_inner_offset :] = tr_corner_conv
        x_grads[:, :, -self.max_inner_offset :, 0 : self.max_inner_offset] = bl_corner_conv
        x_grads[:, :, -self.max_inner_offset :, -self.max_inner_offset :] = br_corner_conv

        # reshape back to the original dimensions
        x_grads = x_grads.view(original_size)
        return x_grads


class StencilGradients(nn.Module):
    """
    This is hard-coded for finite differences on images with n-th order accuracy (for first and second derivatives).
    """

    def __init__(self, d0=1, d1=1, fd_acc=2, periodic=False, device="cpu"):
        super(StencilGradients, self).__init__()
        self.d_d0 = StencilGradientComputation(FinDiff(0, d0, 1, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d1 = StencilGradientComputation(FinDiff(1, d1, 1, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d00 = StencilGradientComputation(FinDiff(0, d0, 2, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d11 = StencilGradientComputation(FinDiff(1, d1, 2, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d01 = StencilGradientComputation(FinDiff((0, d0, 1), (1, d1, 1), acc=fd_acc).stencil((99, 99)).data, periodic, device)

    def forward(self, x, mode):
        if mode == "all":
            return self.d_d0(x), self.d_d1(x), self.d_d00(x), self.d_d11(x), self.d_d01(x)
        elif mode == "d_d0":
            return self.d_d0(x)
        elif mode == "d_d1":
            return self.d_d1(x)
        elif mode == "d_d00":
            return self.d_d00(x)
        elif mode == "d_d11":
            return self.d_d11(x)
        elif mode == "d_d01":
            return self.d_d01(x)
        else:
            raise NotImplementedError


class GradientsHelper:
    """
    Photonics GradientsHelper: compute Helmholtz residual for Ez / eps_r.

    We assume:
      - input x_phys has shape [B, 3, H, W]
      - channel 0: Ez (real)
      - channel 1: Ez (imag)
      - channel 2: eps_r

    PDE enforced (per pixel):

        Δ Ez + k0² eps_r Ez = 0

    where k0 = 2π / λ, and λ can be per-sample.
    """

    def __init__(
        self,
        device,
        dx: float = 0.04,
        dy: float = 0.04,
        pixels_per_dim: int = 64,
        wavelength_um: float = 1.55,  # default λ if none is passed later
        fd_acc: int = 2,
        periodic: bool = False,
    ):
        self.device = torch.device(device)
        self.pixels_per_dim = pixels_per_dim
        self.dx = dx
        self.dy = dy
        self.fd_acc = fd_acc
        self.periodic = periodic

        # store default wavelength and corresponding k0^2
        self.lambda_default_um = float(wavelength_um)
        self.k0_default_sq = (2.0 * torch.pi / wavelength_um) ** 2

        # finite-diff stencils for 2nd derivatives in x, y
        self.stencil_gradients = StencilGradients(
            d0=self.dx,
            d1=self.dy,
            fd_acc=self.fd_acc,
            periodic=self.periodic,
            device=self.device,
        )

    def compute_residual(
        self,
        x_phys: torch.Tensor,
        reduce: str = "none",
        wavelength_um: torch.Tensor | None = None,
    ):
        """
        Compute Helmholtz residual for batch of fields.

        x_phys: [B, 3, H, W] in *physical units*:
            x_phys[:, 0] = Re(Ez)
            x_phys[:, 1] = Im(Ez)
            x_phys[:, 2] = eps_r

        wavelength_um (optional): [B,1] or [B,1,1,1]
            - if provided, we use k0 = 2π / λ per sample
            - if None, fall back to default wavelength set in __init__
        """
        if x_phys.ndim != 4:
            raise ValueError(f"Expected x_phys of shape [B, C, H, W], got {x_phys.shape}")

        B, C, H, W = x_phys.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels (Ez_real, Ez_imag, eps_r), got C={C}")

        # --- split channels ---
        Ez_re = x_phys[:, 0:1, :, :]   # [B,1,H,W]
        Ez_im = x_phys[:, 1:2, :, :]   # [B,1,H,W]
        eps_r = x_phys[:, 2:3, :, :]   # [B,1,H,W]

        # second derivatives for Re(Ez)
        Ezre_d00 = self.stencil_gradients(Ez_re, mode="d_d00")  # [B,1,H,W]
        Ezre_d11 = self.stencil_gradients(Ez_re, mode="d_d11")  # [B,1,H,W]
        lap_re = Ezre_d00 + Ezre_d11

        # second derivatives for Im(Ez)
        Ezim_d00 = self.stencil_gradients(Ez_im, mode="d_d00")
        Ezim_d11 = self.stencil_gradients(Ez_im, mode="d_d11")
        lap_im = Ezim_d00 + Ezim_d11

        # -------- k0^2 (per-sample if wavelength_um is given) --------
        if wavelength_um is None:
            # scalar default k0^2
            k0_sq = self.k0_default_sq.to(self.device)
            # broadcast to [B,1,1,1] for consistency
            k0_sq = k0_sq.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            # wavelength_um can be [B,1] or [B,1,1,1]
            lam = wavelength_um.to(device=self.device, dtype=x_phys.dtype)
            if lam.ndim == 2:
                lam = lam.view(B, 1, 1, 1)
            elif lam.ndim == 4:
                # assume already [B,1,1,1] or [B,1,H,W]
                if lam.shape[2:] == (1, 1):
                    pass
                elif lam.shape[2:] == (H, W):
                    # fine: per-pixel wavelength, keep as is
                    pass
                else:
                    raise ValueError(f"Unexpected wavelength_um shape {lam.shape}")
            else:
                raise ValueError(f"Unsupported wavelength_um shape {lam.shape}")

            k0_sq = (2.0 * torch.pi / lam) ** 2  # [B,1,1,1] or [B,1,H,W]

        # Helmholtz PDE residuals:
        #   ΔEz_re + k0² eps_r Ez_re = 0
        #   ΔEz_im + k0² eps_r Ez_im = 0
        res_re = lap_re + k0_sq * eps_r * Ez_re
        res_im = lap_im + k0_sq * eps_r * Ez_im

        # clamp to avoid insane values in early training
        res_re = torch.clamp(res_re, min=-1e6, max=1e6)
        res_im = torch.clamp(res_im, min=-1e6, max=1e6)

        res_sq = res_re ** 2 + res_im ** 2
        res_mag = torch.sqrt(res_sq + 1e-12)  # [B,1,H,W]

        # Flatten spatial dims to match PBFM’s [B, H*W, 1] convention
        residual_flat = generalized_image_to_b_xy_c(res_mag)
        residual_sq_flat = generalized_image_to_b_xy_c(res_sq)

        output = {"residual": residual_flat, "residual_sq": residual_sq_flat}

        if reduce == "full":
            return {k: v.mean() for k, v in output.items()}
        elif reduce == "per-batch":
            return {
                k: v.mean(dim=tuple(range(1, v.ndim))) if v.ndim > 1 else v
                for k, v in output.items()
            }
        elif reduce == "none":
            return output
        else:
            raise ValueError("Unknown reduction method.")



