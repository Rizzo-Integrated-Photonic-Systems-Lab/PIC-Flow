#!/usr/bin/env python3
"""
Gradient-based inverse design of an asymmetric power-split Y-branch.

Treats the trained flow-matching surrogate as a differentiable forward
operator (eps_map -> Ez) and optimizes ALL of the Y-branch's geometric
parameters except the input waveguide width so that the field-power
split between the two output ports matches a target ratio (default
70/30; see --target-top).

Learnable parameters (independent for top/bottom arms where applicable):
  * widths_top, widths_bot : two 13-segment half-width profiles for the
    upper and lower edges of the multimode junction (asymmetric).
  * h_top, h_bot           : vertical bend offsets for the two arms.
  * l_bend_top, l_bend_bot : bend lengths for the two arms.
  * l_in                   : input stem length.
  * l_junction             : junction taper length.

The waveguide width itself (wg_width) is fixed so the input stem and
output arms remain single-mode at the trained core index.

Loss (target_top in (0, 1), target_bot = 1 - target_top):
    L = (target_bot * P_top - target_top * P_bot)^2 / (P_top + P_bot)^2,
    where P_k = sum(port_mask_k * (Re(Ez)^2 + Im(Ez)^2)).
The denominator makes the loss invariant to overall field-amplitude scale.

The eps map is rendered in pure PyTorch with sigmoid-soft edges so
gradients flow from loss -> field -> eps map -> all learnable parameters.
Soft box constraints keep each parameter inside the trained Y-branch
sweep ranges to limit OOD drift.

Memory: a Heun (RK2) sampler does 2*N model calls. We checkpoint each
model call so activations are O(1) in N at the cost of one recomputation
per call. Default N=10 sampling steps; tune via --num-steps.

Example (asymmetric 70/30 split):
    python tools/inverse_design_ybranch_5050.py \\
        --ckpt /dartfs/.../REAL_VS_COMPLEX_real_h56_phase_residual_v100x12/checkpoints/0000300.pt \\
        --target-top 0.7 --num-steps 10 --iters 150 --lr 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Model"
FDTD_DIR = REPO_ROOT / "FDTD"
TOOLS_DIR = REPO_ROOT / "tools"
for _p in (str(MODEL_DIR), str(FDTD_DIR), str(TOOLS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from predict_parametric_device import (  # noqa: E402
    _build_model_from_checkpoint,
    _checkpoint_state_dict,
    _ckpt_get,
    _build_cond_vector,
    _sdf_nm_from_eps,
    _sdf_feature,
)
from utils import neff_siwire_from_tables  # noqa: E402


DEFAULT_CKPT = (
    Path("/dartfs/rc/lab/R/RizzoA/f0071mj/logs_physics_unet_pbfm")
    / "REAL_VS_COMPLEX_real_h56_phase_residual_v100x12"
    / "checkpoints"
    / "0000300.pt"
)

DEFAULT_RESOLUTION = 20
DEFAULT_DPML = 1.0
DEFAULT_CROP_X_PX = 480
DEFAULT_CROP_Y_PX = 160
DEFAULT_WAVELENGTH_UM = 1.55
N_CLAD = 1.444

# Fixed waveguide width (input stem + output arms) — single-mode at trained core index.
WG_WIDTH_UM = 0.45

# Initial values for learnable scalars (taken from the symmetric trained Y-branch
# defaults — optimization breaks symmetry away from this starting point).
INIT_PARAMS = dict(
    h_top_um=2.0,
    h_bot_um=2.0,
    l_bend_top_um=6.0,
    l_bend_bot_um=6.0,
    l_in_um=1.0,
    l_junction_um=2.0,
)

# Soft bounds for each learnable parameter — chosen to keep optimization within
# the trained Y-branch sweep so the surrogate stays in-distribution.
PARAM_BOUNDS = dict(
    half_width_um=(0.20, 0.75),     # half-widths along the junction edges
    h_um=(0.40, 2.50),              # arm vertical offset
    l_bend_um=(4.0, 7.5),           # arm bend length
    l_in_um=(0.5, 2.5),             # stem length
    l_junction_um=(1.0, 3.0),       # junction taper length
)

# Default 13-width junction profile from the Tidy3D Y-branch example,
# used by FDTD/ybranch/ybranch.py:163-175. Stored as full-widths; we
# halve to initialize the per-edge half-width arrays for top and bottom.
TIDY3D_BASE_WIDTHS = np.array(
    [0.5, 0.5, 0.6, 0.7, 0.9, 1.26, 1.4, 1.4, 1.4, 1.4, 1.31, 1.2, 1.2],
    dtype=np.float32,
)


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


# ---------------------------------------------------------------------------
# Differentiable Y-branch eps renderer
# ---------------------------------------------------------------------------
class DifferentiableYBranch:
    """
    Renders the Y-branch eps map (and source / port masks) as a differentiable
    function of every geometric parameter except the input waveguide width.

    Layout (interior, x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2]):
        | input stem | asym. junction | top S-bend | top output straight |
                     |                | bot S-bend | bot output straight |

    The junction has independent upper-edge (widths_top) and lower-edge
    (widths_bot) half-width profiles, each sampled at 13 anchor x-positions,
    so the junction itself can be asymmetric in y. Top and bottom S-bends
    have independent lengths and vertical offsets.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        dx_um: float,
        dy_um: float,
        wavelength_um: float,
        wg_width_um: float = WG_WIDTH_UM,
        port_y_pad_um: float = 0.4,
        source_shift_um: float = 0.5,
        edge_sigma_um: float = 0.04,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx_um = float(dx_um)
        self.dy_um = float(dy_um)
        self.wavelength_um = float(wavelength_um)
        self.wg_width_um = float(wg_width_um)
        self.port_y_pad_um = float(port_y_pad_um)
        self.source_shift_um = float(source_shift_um)
        self.edge_sigma_um = float(edge_sigma_um)

        self.Lx = self.nx * self.dx_um
        self.Ly = self.ny * self.dy_um
        self.x_left = -0.5 * self.Lx
        self.x_right = +0.5 * self.Lx

        # Index materials: clad = 1.444 -> eps_clad ~ 2.085; core from neff table.
        n_core = float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
        self.eps_core = float(n_core ** 2)
        self.eps_clad = float(N_CLAD ** 2)

        # Pixel-center coordinate grids (µm), origin at cell center.
        # torch.meshgrid(x, y, indexing="xy") returns shape (len(y), len(x)) = (ny, nx),
        # which matches the imshow/dataset convention; no transpose needed.
        ix = torch.arange(self.nx, device=device, dtype=dtype)
        iy = torch.arange(self.ny, device=device, dtype=dtype)
        x = (ix + 0.5) * self.dx_um - 0.5 * self.Lx
        y = (iy + 0.5) * self.dy_um - 0.5 * self.Ly
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")  # both [ny, nx]
        self.X = self.X.contiguous()
        self.Y = self.Y.contiguous()

        # Source: fixed thick line at the trained YBranch convention. Trained
        # data has port_x_left = nonpml_left + 0.8, with the source 0.5 µm
        # upstream of that, so x_src = nonpml_left + 0.3 = x_left + 0.3.
        self.x_src = self.x_left + 0.3
        # Distance to extend the stem and output straights *past* the cell
        # edges so eps is uniform-core at the boundary (matching training,
        # where waveguides extend through the PML before cropping). Without
        # this, the soft-edge sigmoid leaves a fading transition right at
        # the grid boundary that the model has never seen.
        self._extend_um = 1.0

    # ---- helpers --------------------------------------------------------
    def _sig(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z / self.edge_sigma_um)

    def _strip_indicator(
        self,
        x_lo: torch.Tensor | float,
        x_hi: torch.Tensor | float,
        y_center: torch.Tensor | float,
        half_width: torch.Tensor | float,
    ) -> torch.Tensor:
        in_x = self._sig(self.X - x_lo) * self._sig(x_hi - self.X)
        in_y = self._sig(half_width - torch.abs(self.Y - y_center))
        return in_x * in_y

    @staticmethod
    def _soft_union(masks: list[torch.Tensor]) -> torch.Tensor:
        out = torch.zeros_like(masks[0])
        for m in masks:
            out = out + m - out * m
        return out

    @staticmethod
    def _bend_offset(s: torch.Tensor, L: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Sine S-bend centerline offset: u = clamp(s/L, 0, 1);
        offset = u*h - h/(2π) * sin(2π u). Tensor-valued L and h carry grad."""
        L_safe = torch.clamp(L, min=1e-6)
        u = torch.clamp(s / L_safe, 0.0, 1.0)
        return (u * h) - (h / (2.0 * math.pi)) * torch.sin(2.0 * math.pi * u)

    # ---- render ---------------------------------------------------------
    def render(
        self,
        *,
        widths_top: torch.Tensor,    # [13] half-widths along upper junction edge (µm)
        widths_bot: torch.Tensor,    # [13] half-widths along lower junction edge (µm, positive)
        h_top: torch.Tensor,         # vertical offset of top arm at bend exit (µm, positive)
        h_bot: torch.Tensor,         # vertical offset of bottom arm at bend exit (µm, positive)
        l_bend_top: torch.Tensor,    # top S-bend length (µm)
        l_bend_bot: torch.Tensor,    # bottom S-bend length (µm)
        l_in: torch.Tensor,          # stem length (µm)
        l_junction: torch.Tensor,    # junction taper length (µm)
    ) -> dict[str, torch.Tensor]:
        if widths_top.numel() != 13 or widths_bot.numel() != 13:
            raise ValueError("widths_top and widths_bot must each have 13 entries")

        # Force continuity at junction inlet: both edges meet the input stem.
        inlet_half = torch.tensor(
            [0.5 * self.wg_width_um],
            device=widths_top.device, dtype=widths_top.dtype,
        )
        half_top = torch.cat([inlet_half, widths_top[1:]], dim=0)  # [13]
        half_bot = torch.cat([inlet_half, widths_bot[1:]], dim=0)  # [13]

        # X-axis layout, left-to-right.
        x_j0 = self.x_left + l_in
        x_j1 = x_j0 + l_junction
        x_b1_top = x_j1 + l_bend_top
        x_b1_bot = x_j1 + l_bend_bot

        masks: list[torch.Tensor] = []

        # 1) Input stem (symmetric). Extend leftward past the grid so the
        #    leftmost pixels of the rendered eps map are uniform-core,
        #    matching the trained-data convention where the waveguide ran
        #    through the PML.
        masks.append(self._strip_indicator(
            self.x_left - self._extend_um, x_j0, 0.0, 0.5 * self.wg_width_um,
        ))

        # 2) Asymmetric junction: 12 sub-segments, each linearly interpolated
        #    in x between successive (half_top, half_bot) anchors.
        Nw = 13
        dx_seg = (x_j1 - x_j0) / (Nw - 1)
        for i in range(Nw - 1):
            x0 = x_j0 + i * dx_seg
            x1 = x_j0 + (i + 1) * dx_seg
            t_local = (self.X - x0) / torch.clamp(x1 - x0, min=1e-9)
            t_local = torch.clamp(t_local, 0.0, 1.0)
            ht_local = (1.0 - t_local) * half_top[i] + t_local * half_top[i + 1]
            hb_local = (1.0 - t_local) * half_bot[i] + t_local * half_bot[i + 1]
            in_x = self._sig(self.X - x0) * self._sig(x1 - self.X)
            in_y = self._sig(ht_local - self.Y) * self._sig(self.Y + hb_local)
            masks.append(in_x * in_y)

        # 3) S-bends. Bend edges aligned with the asymmetric junction edges:
        #    top arm centerline at bend start = +half_top[-1] - wg_width/2
        #    bot arm centerline at bend start = -(half_bot[-1] - wg_width/2)
        bend_half_w = 0.5 * self.wg_width_um
        y_base_top = half_top[-1] - bend_half_w
        y_base_bot = half_bot[-1] - bend_half_w
        s = self.X - x_j1
        offset_top = self._bend_offset(s, l_bend_top, h_top)
        offset_bot = self._bend_offset(s, l_bend_bot, h_bot)
        y_top_arm = +y_base_top + offset_top
        y_bot_arm = -y_base_bot - offset_bot
        in_x_bend_top = self._sig(s) * self._sig(x_b1_top - self.X)
        in_x_bend_bot = self._sig(s) * self._sig(x_b1_bot - self.X)
        masks.append(in_x_bend_top * self._sig(bend_half_w - torch.abs(self.Y - y_top_arm)))
        masks.append(in_x_bend_bot * self._sig(bend_half_w - torch.abs(self.Y - y_bot_arm)))

        # 4) Output straights. Extend rightward past the grid so the
        #    rightmost pixels of the rendered eps map are uniform-core,
        #    matching the trained-data convention (output guides extend
        #    through the PML before cropping).
        y_top_out = +y_base_top + h_top
        y_bot_out = -y_base_bot - h_bot
        masks.append(self._strip_indicator(
            x_b1_top, self.x_right + self._extend_um, y_top_out, bend_half_w,
        ))
        masks.append(self._strip_indicator(
            x_b1_bot, self.x_right + self._extend_um, y_bot_out, bend_half_w,
        ))

        core_ind = self._soft_union(masks)
        eps_phys = self.eps_clad + (self.eps_core - self.eps_clad) * core_ind

        # Source mask (fixed). Thick vertical line at x_src, ±port_y/2 in y.
        port_y = self.wg_width_um + self.port_y_pad_um
        in_x_src = self._sig(self.X - (self.x_src - 1.5 * self.dx_um)) * self._sig(
            (self.x_src + 1.5 * self.dx_um) - self.X
        )
        in_y_src = self._sig(0.5 * port_y - torch.abs(self.Y))
        src_mask = in_x_src * in_y_src

        # Input port mask: midway between source and junction inlet, always
        # inside the input stem regardless of l_in. Used to anchor the loss
        # to actual injected power so the optimizer can't just collapse the
        # field amplitude.
        x_in_port = 0.5 * (self.x_src + x_j0)
        in_x_in_port = self._sig(self.X - (x_in_port - 1.5 * self.dx_um)) * self._sig(
            (x_in_port + 1.5 * self.dx_um) - self.X
        )
        port_mask_in = in_x_in_port * self._sig(0.5 * port_y - torch.abs(self.Y))

        # Output port masks track each arm's actual y-position so optimization
        # changes where we measure the output.
        x_port = self.x_right - 0.5
        in_x_port = self._sig(self.X - (x_port - 1.5 * self.dx_um)) * self._sig(
            (x_port + 1.5 * self.dx_um) - self.X
        )
        port_mask_top = in_x_port * self._sig(0.5 * port_y - torch.abs(self.Y - y_top_out))
        port_mask_bot = in_x_port * self._sig(0.5 * port_y - torch.abs(self.Y - y_bot_out))

        return dict(
            eps_phys=eps_phys,
            src_mask=src_mask,
            port_mask_in=port_mask_in,
            port_mask_top=port_mask_top,
            port_mask_bot=port_mask_bot,
            half_top=half_top,
            half_bot=half_bot,
            y_top_out=y_top_out.detach() if isinstance(y_top_out, torch.Tensor) else y_top_out,
            y_bot_out=y_bot_out.detach() if isinstance(y_bot_out, torch.Tensor) else y_bot_out,
        )


# ---------------------------------------------------------------------------
# Conditioning maps
# ---------------------------------------------------------------------------
def build_cond_maps_diff(
    eps_phys: torch.Tensor, src_mask: torch.Tensor,
    *, stats: dict[str, Any], ckpt_args: Any,
    dx_um: float, dy_um: float,
) -> torch.Tensor:
    """Differentiable replacement for predict_parametric_device._build_cond_maps.

    eps_norm and src_mask channels carry gradients (the optimization signal).
    When the trained model expects extra SDF channels (x_channels > 4), they
    are recomputed every iteration from a *detached* numpy copy of eps_phys
    via scipy.ndimage.distance_transform_edt and contribute zero gradient.
    The SDF stays consistent with the current geometry but the optimizer is
    driven only by eps_norm / src_mask gradients — which is the dominant
    signal anyway.
    """
    normalize_eps = bool(_ckpt_get(ckpt_args, "normalize_eps", True))
    if normalize_eps:
        eps_norm = (eps_phys - float(stats["eps_mean"])) / float(stats["eps_std"])
    else:
        eps_norm = eps_phys

    # Hard-binarize the source mask before feeding to the model. The trained
    # data uses {0, 1} masks (predict_parametric_device does the same `> 0`
    # binarization), so a soft 0..1 sigmoid mask is OOD and can suppress the
    # injected field. The src position is fixed; we don't need its gradient.
    src_bin = (src_mask.detach() > 0.5).to(eps_phys.dtype)
    chans = [
        eps_norm.unsqueeze(0).unsqueeze(0),                  # [1, 1, ny, nx]
        src_bin.unsqueeze(0).unsqueeze(0),                   # [1, 1, ny, nx]
    ]

    x_channels = int(stats.get("x_channels", 4))
    if x_channels > 4:
        # Non-differentiable SDF channel(s) from current eps. Same code path as
        # predict_parametric_device._build_cond_maps but on a detached numpy copy.
        eps_np = eps_phys.detach().cpu().numpy()
        thr = float(stats.get("sdf_thr_eps", _ckpt_get(ckpt_args, "sdf_thr_eps", 3.0)))
        phi_nm = _sdf_nm_from_eps(eps_np, dx_um=float(dx_um), dy_um=float(dy_um), thr_eps=thr)
        feature = str(stats.get("sdf_feature", _ckpt_get(ckpt_args, "sdf_feature", "raw")))
        sigma_nm = float(stats.get("sdf_sigma_nm", _ckpt_get(ckpt_args, "sdf_sigma_nm", 100.0)))
        phi_feat = _sdf_feature(phi_nm, feature=feature, sigma_nm=sigma_nm)

        if bool(_ckpt_get(ckpt_args, "normalize_sdf", True)):
            for c in range(phi_feat.shape[0]):
                if feature == "raw":
                    mu = float(stats.get("sdf_nm_mean", 0.0))
                    sd = float(stats.get("sdf_nm_std", 1.0))
                else:
                    mu = float(stats.get(f"sdf_feat{c}_mean", stats.get("sdf_feat_mean", 0.0)))
                    sd = float(stats.get(f"sdf_feat{c}_std", stats.get("sdf_feat_std", 1.0)))
                phi_feat[c] = (phi_feat[c] - mu) / max(sd, 1e-8)

        sdf_tensor = torch.from_numpy(phi_feat.astype(np.float32, copy=False))[None].to(
            device=eps_phys.device, dtype=eps_phys.dtype,
        )
        chans.append(sdf_tensor)

    return torch.cat(chans, dim=1)


# ---------------------------------------------------------------------------
# Per-step Heun sampler with gradient checkpointing
# ---------------------------------------------------------------------------
def fm_sample_checkpointed(
    model: torch.nn.Module,
    x0: torch.Tensor,
    *,
    cond_maps: torch.Tensor,
    cond: torch.Tensor | None,
    lambda_um: torch.Tensor,
    num_steps: int,
    time_grid: str = "linear",
    progress: bool = False,
) -> torch.Tensor:
    """Heun integrator with torch.utils.checkpoint around each model call so
    activations are not retained across steps. Equivalent to
    flow_matching.sample but uses checkpointing for memory.
    """
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]

    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if time_grid == "quadratic":
        time_steps = base ** 2
    else:
        time_steps = base

    x = x0
    n_ch = x.shape[1]
    iter_range = range(num_steps)
    if progress:
        try:
            from tqdm.auto import trange
            iter_range = trange(num_steps, desc="FM (ckpt)", leave=False)
        except ModuleNotFoundError:
            pass

    def call_model(x_in, t_vec):
        if cond is None:
            return model(x_in, t_vec, lambda_um=lambda_um)
        return model(x_in, t_vec, cond=cond, lambda_um=lambda_um)

    for k in iter_range:
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        dt = t1 - t0
        t_vec0 = t0.expand(B)
        t_vec1 = t1.expand(B)

        x_in0 = torch.cat([x, cond_maps], dim=1)
        v0 = ckpt_fn(call_model, x_in0, t_vec0, use_reentrant=False)[:, :n_ch]
        x_eu = x + dt * v0

        x_in1 = torch.cat([x_eu, cond_maps], dim=1)
        v1 = ckpt_fn(call_model, x_in1, t_vec1, use_reentrant=False)[:, :n_ch]

        x = x + 0.5 * dt * (v0 + v1)

    return x


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def split_loss(
    ezr: torch.Tensor, ezi: torch.Tensor,
    mask_in: torch.Tensor,
    mask_top: torch.Tensor, mask_bot: torch.Tensor,
    target_top: float = 0.7,
    eps_pin: float = 1e-6,
    lossless_weight: float = 0.5,
) -> tuple[torch.Tensor, float, float, float]:
    """Transmission-anchored power-split loss with explicit lossless term.

        T_k  = P_k / P_in,   P_x = sum(mask_x * (Re^2 + Im^2))
        L    = (T_top - target_top)^2 + (T_bot - target_bot)^2
                   + lossless_weight * (1 - (T_top + T_bot))^2

    The lossless term penalizes designs where the output ports don't carry
    the input power, eliminating the degenerate "collapse outputs to zero"
    minimum (which would otherwise score `target_top^2 + target_bot^2`
    plus `lossless_weight`, far worse than the true optimum at 0). At the
    target operating point T_top = target_top, T_bot = target_bot,
    T_top + T_bot = 1 the loss is exactly zero.

    Returns (loss, P_in, P_top, P_bot).
    """
    if not (0.0 < float(target_top) < 1.0):
        raise ValueError(f"target_top must be in (0, 1); got {target_top}")
    target_bot = 1.0 - float(target_top)
    energy = ezr ** 2 + ezi ** 2  # [ny, nx]
    p_in = (mask_in * energy).sum()
    p_top = (mask_top * energy).sum()
    p_bot = (mask_bot * energy).sum()
    p_in_safe = p_in + float(eps_pin)
    t_top = p_top / p_in_safe
    t_bot = p_bot / p_in_safe
    ratio_err = (t_top - float(target_top)) ** 2 + (t_bot - target_bot) ** 2
    lossless_err = (1.0 - (t_top + t_bot)) ** 2
    loss = ratio_err + float(lossless_weight) * lossless_err
    return (
        loss,
        float(p_in.detach()),
        float(p_top.detach()),
        float(p_bot.detach()),
    )


# ---------------------------------------------------------------------------
# FDTD verification (Meep): mirrors DifferentiableYBranch.render() in real
# Meep primitives so we can ground-truth the converged geometry.
# ---------------------------------------------------------------------------
class AsymmetricYBranchMeep:
    """Asymmetric Y-branch built with Meep primitives, matching the geometry
    produced by `DifferentiableYBranch.render()` 1:1. Inherits from
    `Device2DBase` lazily inside `__init__` so this script doesn't fail to
    import when Meep isn't available."""

    def __init__(
        self,
        *,
        wg_width_um: float,
        wavelength_um: float,
        widths_top: list[float],     # 13 half-widths along upper junction edge (µm)
        widths_bot: list[float],     # 13 half-widths along lower junction edge (µm)
        h_top: float, h_bot: float,
        l_bend_top: float, l_bend_bot: float,
        l_in: float, l_junction: float,
        cell_x: float, cell_y: float,
        dpml: float, resolution: int,
        n_core: float | None = None, n_clad: float = 1.444,
        bend_n_segments: int = 80,
        port_y_pad_um: float = 0.4, source_shift_um: float = 0.5,
    ):
        import meep as mp
        from devices_base import Device2DBase

        # Stick a Device2DBase mixin onto self so super().cell etc. work.
        Device2DBase.__init__(self, cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution)

        self.wg_width_um = float(wg_width_um)
        self.wavelength_um = float(wavelength_um)
        self.widths_top = [float(w) for w in widths_top]
        self.widths_bot = [float(w) for w in widths_bot]
        self.h_top = float(h_top); self.h_bot = float(h_bot)
        self.l_bend_top = float(l_bend_top); self.l_bend_bot = float(l_bend_bot)
        self.l_in = float(l_in); self.l_junction = float(l_junction)
        self.bend_n_segments = int(bend_n_segments)
        self.port_y_pad_um = float(port_y_pad_um)
        self.source_shift_um = float(source_shift_um)

        self.n_core = (
            float(n_core) if n_core is not None
            else float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
        )
        self.n_clad = float(n_clad)
        self.core_medium = mp.Medium(index=self.n_core)
        self.clad_medium = mp.Medium(index=self.n_clad)
        self.full_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(self.cell_x, self.cell_y, 0),
        )

        self.geometry = None
        self.port_1 = self.port_2 = self.port_3 = None
        self.src_1 = None
        self.y_top_out = None
        self.y_bot_out = None

        self.build_geometry()

    # --- helpers (same shape as CascadedY1x3 in tools/make_ood_inference_triptychs.py) ---
    def _straight(self, x0, x1, y_center, w):
        import meep as mp
        if x1 <= x0:
            return []
        L = x1 - x0
        return [mp.Block(
            size=mp.Vector3(L, w, mp.inf),
            center=mp.Vector3(0.5 * (x0 + x1), y_center, 0),
            material=self.core_medium,
        )]

    def _sine_sbend_blocks(self, x0, x1, y_center, h, w, n_seg=80):
        import meep as mp
        L = x1 - x0
        if L <= 0:
            return []
        seg_len = L / n_seg
        blocks = []
        for i in range(n_seg):
            u = (i + 0.5) / n_seg
            x = x0 + u * L
            s = u * L
            y = y_center + (s * h / L) - (h * np.sin(2.0 * np.pi * s / L) / (2.0 * np.pi))
            blocks.append(mp.Block(
                size=mp.Vector3(seg_len * 1.05, w, mp.inf),
                center=mp.Vector3(float(x), float(y), 0),
                material=self.core_medium,
            ))
        return blocks

    def build_geometry(self):
        import meep as mp

        half_x = 0.5 * self.cell_x
        nonpml_left = -half_x + self.dpml
        nonpml_right = +half_x - self.dpml

        # Renderer's x_left = -inner_x/2 = nonpml_left, so the layout aligns.
        x_left = nonpml_left
        x_j0 = x_left + self.l_in
        x_j1 = x_j0 + self.l_junction
        x_b1_top = x_j1 + self.l_bend_top
        x_b1_bot = x_j1 + self.l_bend_bot

        if max(x_b1_top, x_b1_bot) > nonpml_right:
            raise ValueError(
                f"AsymmetricYBranchMeep does not fit: bend ends "
                f"{x_b1_top:.3g}/{x_b1_bot:.3g} µm vs avail right edge {nonpml_right:.3g} µm"
            )

        # Inlet continuity (override widths[0] to wg_width/2).
        widths_top = list(self.widths_top); widths_top[0] = 0.5 * self.wg_width_um
        widths_bot = list(self.widths_bot); widths_bot[0] = 0.5 * self.wg_width_um

        geometry = []

        # 1) Input stem (extends into left PML for clean source injection).
        geometry += self._straight(-half_x, x_j0, 0.0, self.wg_width_um)

        # 2) Asymmetric junction Prism (or block fallback).
        Nw = 13
        x_top_arr = np.linspace(x_j0, x_j1, Nw)
        verts = []
        for i in range(Nw):
            verts.append(mp.Vector3(float(x_top_arr[i]), float(widths_top[i]), 0))
        for i in range(Nw - 1, -1, -1):
            verts.append(mp.Vector3(float(x_top_arr[i]), float(-widths_bot[i]), 0))
        try:
            geometry.append(mp.Prism(
                vertices=verts, height=mp.inf, axis=mp.Z, material=self.core_medium,
            ))
        except Exception:
            dx_seg = (x_j1 - x_j0) / (Nw - 1)
            for i in range(Nw - 1):
                xc = 0.5 * (x_top_arr[i] + x_top_arr[i + 1])
                top_avg = 0.5 * (widths_top[i] + widths_top[i + 1])
                bot_avg = 0.5 * (widths_bot[i] + widths_bot[i + 1])
                yc_avg = 0.5 * (top_avg - bot_avg)
                w_avg = top_avg + bot_avg
                geometry.append(mp.Block(
                    size=mp.Vector3(dx_seg * 1.05, w_avg, mp.inf),
                    center=mp.Vector3(float(xc), float(yc_avg), 0),
                    material=self.core_medium,
                ))

        # 3) S-bends (two arms, independent length and offset).
        bend_half_w = 0.5 * self.wg_width_um
        y_base_top = widths_top[-1] - bend_half_w
        y_base_bot = widths_bot[-1] - bend_half_w
        geometry += self._sine_sbend_blocks(
            x_j1, x_b1_top, +y_base_top, +self.h_top, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )
        geometry += self._sine_sbend_blocks(
            x_j1, x_b1_bot, -y_base_bot, -self.h_bot, self.wg_width_um,
            n_seg=self.bend_n_segments,
        )

        # 4) Output straights to right PML.
        y_top_out = +y_base_top + self.h_top
        y_bot_out = -y_base_bot - self.h_bot
        geometry += self._straight(x_b1_top, half_x, y_top_out, self.wg_width_um)
        geometry += self._straight(x_b1_bot, half_x, y_bot_out, self.wg_width_um)

        self.geometry = geometry
        self.y_top_out = float(y_top_out)
        self.y_bot_out = float(y_bot_out)

        # Ports + source (matches the trained YBranch convention).
        port_span = self.wg_width_um + self.port_y_pad_um
        port_size = mp.Vector3(0, port_span, 0)
        port_x_left = nonpml_left + 0.8
        port_x_right = nonpml_right - 0.5

        self.port_1 = mp.Volume(center=mp.Vector3(port_x_left, 0.0, 0), size=port_size)
        self.port_2 = mp.Volume(center=mp.Vector3(port_x_right, y_top_out, 0), size=port_size)
        self.port_3 = mp.Volume(center=mp.Vector3(port_x_right, y_bot_out, 0), size=port_size)
        src_x = max(port_x_left - self.source_shift_um, nonpml_left + 0.1)
        self.src_1 = mp.Volume(center=mp.Vector3(src_x, 0.0, 0), size=port_size)

    def simulate(self, decay_tol: float = 1e-5, df_frac: float = 0.1):
        import meep as mp
        from utils import get_mode_alpha_2dir, pick_in_out_from_alpha

        fcen = 1.0 / self.wavelength_um
        fwidth = float(df_frac) * fcen
        sources = [mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=fwidth),
            volume=self.src_1, eig_band=1, eig_parity=mp.NO_PARITY,
            eig_match_freq=True, eig_kpoint=mp.Vector3(+1, 0, 0),
        )]
        sim = mp.Simulation(
            cell_size=self.cell, resolution=int(self.resolution),
            boundary_layers=[mp.PML(float(self.dpml))],
            geometry=self.geometry, default_material=self.clad_medium,
            sources=sources,
        )
        m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_1))
        m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_2))
        m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=self.port_3))
        dft = sim.add_dft_fields(
            [mp.Ez], fcen, 0, 1,
            center=self.full_plane.center, size=self.full_plane.size,
        )
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=float(decay_tol)))

        eps_full = sim.get_epsilon().T.astype(np.float32)
        Ez_full = sim.get_dft_array(dft, mp.Ez, 0).T.astype(np.complex64)

        a1 = get_mode_alpha_2dir(sim, m1, band=1, eig_parity=mp.NO_PARITY)
        a2 = get_mode_alpha_2dir(sim, m2, band=1, eig_parity=mp.NO_PARITY)
        a3 = get_mode_alpha_2dir(sim, m3, band=1, eig_parity=mp.NO_PARITY)
        a1_in, b1_out = pick_in_out_from_alpha(a1, +1, dir_plus=0, dir_minus=1)
        _, b2_out = pick_in_out_from_alpha(a2, -1, dir_plus=0, dir_minus=1)
        _, b3_out = pick_in_out_from_alpha(a3, -1, dir_plus=0, dir_minus=1)
        S = {
            "S11": complex((b1_out / a1_in).item()),
            "S21": complex((b2_out / a1_in).item()),
            "S31": complex((b3_out / a1_in).item()),
        }
        sim.reset_meep()
        return eps_full, Ez_full, S


def _verify_fdtd_at_final(
    *, args: argparse.Namespace, renderer: "DifferentiableYBranch",
    widths_top: torch.Tensor, widths_bot: torch.Tensor,
    h_top: torch.Tensor, h_bot: torch.Tensor,
    l_bend_top: torch.Tensor, l_bend_bot: torch.Tensor,
    l_in: torch.Tensor, l_junction: torch.Tensor,
) -> dict[str, Any]:
    """Run Meep on the converged geometry and report realized split."""
    print("[inv-design] running FDTD verification (Meep) ...")
    t0 = time.perf_counter()

    crop_x_px = int(args.crop_x_px); crop_y_px = int(args.crop_y_px)
    resolution = int(args.resolution); dpml = float(getattr(args, "dpml", 1.0))
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * dpml
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * dpml

    dev = AsymmetricYBranchMeep(
        wg_width_um=renderer.wg_width_um,
        wavelength_um=float(args.wavelength_um),
        widths_top=widths_top.detach().cpu().numpy().tolist(),
        widths_bot=widths_bot.detach().cpu().numpy().tolist(),
        h_top=float(h_top.detach()), h_bot=float(h_bot.detach()),
        l_bend_top=float(l_bend_top.detach()), l_bend_bot=float(l_bend_bot.detach()),
        l_in=float(l_in.detach()), l_junction=float(l_junction.detach()),
        cell_x=cell_x, cell_y=cell_y, dpml=dpml, resolution=resolution,
    )
    eps_full, Ez_full, S = dev.simulate(decay_tol=1e-5)

    pml_px = int(round(dpml * resolution))
    if pml_px > 0:
        eps_fdtd = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez_fdtd = Ez_full[pml_px:-pml_px, pml_px:-pml_px]
    else:
        eps_fdtd = eps_full; Ez_fdtd = Ez_full

    # Phase-anchor the FDTD field on the source ROI so amplitude is in the
    # same frame as the trained data.
    from dataset import phase_anchor_mask
    rendered = renderer.render(
        widths_top=widths_top, widths_bot=widths_bot,
        h_top=h_top, h_bot=h_bot,
        l_bend_top=l_bend_top, l_bend_bot=l_bend_bot,
        l_in=l_in, l_junction=l_junction,
    )
    src_bin = (rendered["src_mask"].detach().cpu().numpy() > 0.5)
    ezr_anc, ezi_anc, _ = phase_anchor_mask(
        Ez_fdtd.real.copy(), Ez_fdtd.imag.copy(), src_bin,
        eps_r=eps_fdtd, thr_eps=3.0,
    )

    # Mask-based powers, using the SAME masks the optimizer used.
    mask_in_np = rendered["port_mask_in"].detach().cpu().numpy()
    mask_top_np = rendered["port_mask_top"].detach().cpu().numpy()
    mask_bot_np = rendered["port_mask_bot"].detach().cpu().numpy()
    energy_fdtd = ezr_anc ** 2 + ezi_anc ** 2
    P_in_f = float((mask_in_np * energy_fdtd).sum())
    P_top_f = float((mask_top_np * energy_fdtd).sum())
    P_bot_f = float((mask_bot_np * energy_fdtd).sum())
    p_in_safe = max(P_in_f, 1e-12)

    # Modal S-parameters (from Meep's eigenmode coefficients).
    S21_sq = float(abs(S["S21"]) ** 2)
    S31_sq = float(abs(S["S31"]) ** 2)
    S11_sq = float(abs(S["S11"]) ** 2)

    out = dict(
        eps_fdtd=eps_fdtd.astype(np.float32),
        Ez_real_fdtd=ezr_anc.astype(np.float32),
        Ez_imag_fdtd=ezi_anc.astype(np.float32),
        P_in=P_in_f, P_top=P_top_f, P_bot=P_bot_f,
        T_top_mask=P_top_f / p_in_safe,
        T_bot_mask=P_bot_f / p_in_safe,
        T_total_mask=(P_top_f + P_bot_f) / p_in_safe,
        S21_sq=S21_sq, S31_sq=S31_sq, S11_sq=S11_sq,
        T_top_modal=S21_sq, T_bot_modal=S31_sq,
        T_total_modal=S21_sq + S31_sq,
        sparams_complex=S,
        elapsed_s=time.perf_counter() - t0,
    )
    print(
        f"[inv-design] FDTD done in {out['elapsed_s']:.1f}s. "
        f"target T_top={float(args.target_top):.2f} | "
        f"mask: T_top={out['T_top_mask']:.3f}, T_bot={out['T_bot_mask']:.3f}, "
        f"T_tot={out['T_total_mask']:.3f} | "
        f"modal: |S21|²={S21_sq:.3f}, |S31|²={S31_sq:.3f}, sum={S21_sq+S31_sq:.3f}"
    )
    return out


def _save_verification_plot(
    *,
    eps_model: np.ndarray, ezr_model: np.ndarray, ezi_model: np.ndarray,
    trans_model: dict[str, float],
    fdtd: dict[str, Any],
    extent: tuple[float, float, float, float],
    target_top: float,
    out_path: Path,
) -> None:
    """2x2 side-by-side: model eps + |Ez| (top), FDTD eps + |Ez| (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mag_m = np.sqrt(ezr_model ** 2 + ezi_model ** 2)
    mag_f = np.sqrt(fdtd["Ez_real_fdtd"] ** 2 + fdtd["Ez_imag_fdtd"] ** 2)
    eps_f = fdtd["eps_fdtd"]
    vmax = float(np.percentile(np.concatenate([mag_m.ravel(), mag_f.ravel()]), 99.5))
    vmax = vmax if vmax > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Row 0: model
    axes[0, 0].imshow(eps_model, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[0, 0].set_title(r"Model $\varepsilon_r$")
    axes[0, 0].set_ylabel(r"$y$ ($\mu$m)")
    im_mm = axes[0, 1].imshow(mag_m, origin="lower", extent=extent, cmap="magma",
                              aspect="equal", vmin=0, vmax=vmax)
    axes[0, 1].contour(eps_model, levels=[0.5 * (eps_model.min() + eps_model.max())],
                       colors=["white"], linewidths=0.4, origin="lower", extent=extent)
    axes[0, 1].set_title(
        rf"Model $|E_z|$  $T_{{\rm top}}$={trans_model['T_top']:.2f}, "
        rf"$T_{{\rm bot}}$={trans_model['T_bot']:.2f}, $T_{{\rm tot}}$={trans_model['T_total']:.2f}"
    )
    fig.colorbar(im_mm, ax=axes[0, 1], fraction=0.04, pad=0.02)

    # Row 1: FDTD
    axes[1, 0].imshow(eps_f, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[1, 0].set_title(r"FDTD $\varepsilon_r$")
    axes[1, 0].set_xlabel(r"$x$ ($\mu$m)")
    axes[1, 0].set_ylabel(r"$y$ ($\mu$m)")
    im_ff = axes[1, 1].imshow(mag_f, origin="lower", extent=extent, cmap="magma",
                              aspect="equal", vmin=0, vmax=vmax)
    axes[1, 1].contour(eps_f, levels=[0.5 * (eps_f.min() + eps_f.max())],
                       colors=["white"], linewidths=0.4, origin="lower", extent=extent)
    axes[1, 1].set_title(
        rf"FDTD $|E_z|$  $|S_{{21}}|^2$={fdtd['S21_sq']:.2f}, "
        rf"$|S_{{31}}|^2$={fdtd['S31_sq']:.2f}, sum={fdtd['S21_sq']+fdtd['S31_sq']:.2f}"
    )
    axes[1, 1].set_xlabel(r"$x$ ($\mu$m)")
    fig.colorbar(im_ff, ax=axes[1, 1], fraction=0.04, pad=0.02)

    fig.suptitle(
        rf"Inverse-design verification (target $T_{{\rm top}}$ = {target_top:.2f})",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _save_state(state_dict: dict[str, Any], out_path: Path) -> None:
    """Save plot of eps + |Ez| + asymmetric junction half-width profiles + scalars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = state_dict["eps_phys"]
    ezr = state_dict["ezr"]
    ezi = state_dict["ezi"]
    mask_in = state_dict.get("mask_in")
    mask_top = state_dict["mask_top"]
    mask_bot = state_dict["mask_bot"]
    src = state_dict["src_mask"]
    widths_top = state_dict["widths_top"]
    widths_bot = state_dict["widths_bot"]
    scalars = state_dict["scalars"]
    trans = state_dict.get("transmissions", {})

    mag = np.sqrt(ezr ** 2 + ezi ** 2)
    ext = state_dict["extent"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), constrained_layout=True)
    fig.patch.set_facecolor("white")

    im = axes[0].imshow(eps, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    axes[0].contour(src, levels=[0.5], colors=["lime"], linewidths=1.0,
                    origin="lower", extent=ext)
    if mask_in is not None:
        axes[0].contour(mask_in, levels=[0.5], colors=["yellow"], linewidths=0.8,
                        origin="lower", extent=ext)
    axes[0].contour(mask_top, levels=[0.5], colors=["cyan"], linewidths=0.7,
                    origin="lower", extent=ext)
    axes[0].contour(mask_bot, levels=[0.5], colors=["cyan"], linewidths=0.7,
                    origin="lower", extent=ext)
    axes[0].set_title(r"$\varepsilon_r$ + ports")
    axes[0].set_xlabel(r"$x$ ($\mu$m)")
    axes[0].set_ylabel(r"$y$ ($\mu$m)")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

    im2 = axes[1].imshow(mag, origin="lower", extent=ext, cmap="magma", aspect="equal")
    axes[1].contour(eps, levels=[0.5 * (eps.min() + eps.max())], colors=["white"],
                    linewidths=0.4, origin="lower", extent=ext)
    axes[1].set_title(r"Predicted $|E_z|$")
    axes[1].set_xlabel(r"$x$ ($\mu$m)")
    fig.colorbar(im2, ax=axes[1], fraction=0.04, pad=0.02)

    idx = np.arange(13)
    axes[2].plot(idx, widths_top, "o-", color="C0", label="top half-width")
    axes[2].plot(idx, widths_bot, "s-", color="C3", label="bot half-width")
    axes[2].set_title("Asymmetric junction profile ($\\mu$m)")
    axes[2].set_xlabel("segment index")
    axes[2].set_ylabel(r"half-width ($\mu$m)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)
    scalar_text = (
        f"h = {scalars['h_top']:.2f}/{scalars['h_bot']:.2f}\n"
        f"$\\ell_b$ = {scalars['l_bend_top']:.2f}/{scalars['l_bend_bot']:.2f}\n"
        f"$\\ell_{{in}}$ = {scalars['l_in']:.2f},  $\\ell_{{j}}$ = {scalars['l_junction']:.2f}"
    )
    if trans:
        scalar_text += (
            f"\n$T_\\mathrm{{top}}$ = {trans['T_top']:.2f},  "
            f"$T_\\mathrm{{bot}}$ = {trans['T_bot']:.2f}"
            f"\n$T_\\mathrm{{tot}}$ = {trans['T_total']:.2f}  "
            f"(target $T_\\mathrm{{top}}$ = {trans['target_top']:.2f})"
        )
    axes[2].text(
        0.02, 0.98, scalar_text, transform=axes[2].transAxes,
        ha="left", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=2.5),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse-design a 50/50 Y-branch via FM-surrogate backprop.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "inverse_design_ybranch_5050"))
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--crop-x-px", type=int, default=DEFAULT_CROP_X_PX)
    parser.add_argument("--crop-y-px", type=int, default=DEFAULT_CROP_Y_PX)
    parser.add_argument("--num-steps", type=int, default=10, help="FM sampler steps (Heun => 2x model calls).")
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="linear")
    parser.add_argument("--iters", type=int, default=30,
                        help="Optimization iterations. Loss typically converges "
                             "in 15-25 iters; default 30 leaves a small buffer.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Adam learning rate. Lowered from 2e-2 because the "
                             "model's surrogate landscape has steep cliffs in some "
                             "OOD regions and large steps fall off them.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the noise sample x_0.")
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--target-top", type=float, default=0.7,
                        help="Target power fraction in the top output (0,1). "
                             "0.5 = balanced; 0.7 = 70/30 split; etc.")
    parser.add_argument("--break-symmetry", type=float, default=0.05,
                        help="Initial fractional perturbation between top/bot params, "
                             "to break the symmetric-init local minimum (default 0.05 = 5%%).")
    parser.add_argument("--verify-fdtd", action=argparse.BooleanOptionalAction, default=False,
                        help="After optimization, run a single Meep FDTD simulation on "
                             "the converged geometry and emit verification.png + numerical "
                             "comparison in final.json. Adds ~1-3 min on V100; needs Meep.")
    parser.add_argument("--dpml", type=float, default=1.0,
                        help="PML thickness used for the FDTD verification cell.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False,
                        help="Show tqdm bar inside each FM sampler call (verbose).")
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inv-design] checkpoint: {args.ckpt}")
    print(f"[inv-design] runtime_device: {runtime_device}")
    print(f"[inv-design] num_steps={args.num_steps}, iters={args.iters}, lr={args.lr}")

    # Load checkpoint + model.
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    print(f"[inv-design] loaded weights: {state_key}")

    # Renderer.
    dx_um = 1.0 / float(args.resolution)
    renderer = DifferentiableYBranch(
        nx=int(args.crop_x_px), ny=int(args.crop_y_px),
        dx_um=dx_um, dy_um=dx_um,
        wavelength_um=float(args.wavelength_um),
        device=runtime_device, dtype=torch.float32,
    )

    # Initialize learnable widths to the Tidy3D default half-width profile,
    # scaled by wg_width/0.5. widths_top[0] / widths_bot[0] are forced to
    # wg_width/2 inside render() for inlet continuity (their gradients are
    # zeroed every iteration so optimization can't drift them).
    init_half = (TIDY3D_BASE_WIDTHS * (renderer.wg_width_um / 0.5)) * 0.5
    init_half[0] = 0.5 * renderer.wg_width_um

    # Tiny symmetric -> asymmetric perturbation so the optimizer has a gradient
    # signal away from the symmetric local minimum (loss is exactly stationary
    # there for a y-symmetric architecture).
    eps_break = float(args.break_symmetry)
    init_half_top_np = init_half * (1.0 + eps_break)
    init_half_bot_np = init_half * (1.0 - eps_break)
    init_half_top_np[0] = 0.5 * renderer.wg_width_um
    init_half_bot_np[0] = 0.5 * renderer.wg_width_um

    def _param(value: float, *, requires_grad: bool = True) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.tensor(float(value), dtype=torch.float32, device=runtime_device),
            requires_grad=requires_grad,
        )

    widths_top = torch.nn.Parameter(
        torch.tensor(init_half_top_np, dtype=torch.float32, device=runtime_device)
    )
    widths_bot = torch.nn.Parameter(
        torch.tensor(init_half_bot_np, dtype=torch.float32, device=runtime_device)
    )
    h_top = _param(INIT_PARAMS["h_top_um"] * (1.0 + eps_break))
    h_bot = _param(INIT_PARAMS["h_bot_um"] * (1.0 - eps_break))
    l_bend_top = _param(INIT_PARAMS["l_bend_top_um"])
    l_bend_bot = _param(INIT_PARAMS["l_bend_bot_um"])
    l_in = _param(INIT_PARAMS["l_in_um"])
    l_junction = _param(INIT_PARAMS["l_junction_um"])

    learnables = [widths_top, widths_bot, h_top, h_bot,
                  l_bend_top, l_bend_bot, l_in, l_junction]
    optimizer = torch.optim.Adam(learnables, lr=float(args.lr))

    # Conditioning vector (wavelength normalization).
    cond_vec = _build_cond_vector(float(args.wavelength_um), stats, device=runtime_device)
    lambda_um_t = torch.tensor([[float(args.wavelength_um)]],
                                device=runtime_device, dtype=torch.float32)

    # Fix the noise so each iteration is comparable.
    gen = torch.Generator(device=runtime_device)
    gen.manual_seed(int(args.seed))
    x0 = torch.randn((1, 2, int(args.crop_y_px), int(args.crop_x_px)),
                     device=runtime_device, dtype=torch.float32, generator=gen)

    extent = (
        -0.5 * renderer.Lx, 0.5 * renderer.Lx,
        -0.5 * renderer.Ly, 0.5 * renderer.Ly,
    )

    log_path = out_dir / "trajectory.jsonl"
    with open(log_path, "w") as f:
        f.write("")  # truncate

    def _box_penalty(value: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (F.relu(lo - value) ** 2 + F.relu(value - hi) ** 2).sum()

    hw_lo, hw_hi = PARAM_BOUNDS["half_width_um"]
    h_lo, h_hi = PARAM_BOUNDS["h_um"]
    lb_lo, lb_hi = PARAM_BOUNDS["l_bend_um"]
    li_lo, li_hi = PARAM_BOUNDS["l_in_um"]
    lj_lo, lj_hi = PARAM_BOUNDS["l_junction_um"]

    t_start = time.perf_counter()
    for it in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)

        rendered = renderer.render(
            widths_top=widths_top, widths_bot=widths_bot,
            h_top=h_top, h_bot=h_bot,
            l_bend_top=l_bend_top, l_bend_bot=l_bend_bot,
            l_in=l_in, l_junction=l_junction,
        )
        eps_phys = rendered["eps_phys"]
        src_mask = rendered["src_mask"]
        mask_in = rendered["port_mask_in"]
        mask_top = rendered["port_mask_top"]
        mask_bot = rendered["port_mask_bot"]

        cond_maps = build_cond_maps_diff(
            eps_phys, src_mask, stats=stats, ckpt_args=ckpt_args,
            dx_um=renderer.dx_um, dy_um=renderer.dy_um,
        )

        x_pred = fm_sample_checkpointed(
            model, x0,
            cond_maps=cond_maps, cond=cond_vec, lambda_um=lambda_um_t,
            num_steps=int(args.num_steps), time_grid=str(args.time_grid),
            progress=bool(args.progress),
        )
        ezr = x_pred[0, 0] * float(stats["ez_real_std"]) + float(stats["ez_real_mean"])
        ezi = x_pred[0, 1] * float(stats["ez_imag_std"]) + float(stats["ez_imag_mean"])

        loss, p_in, p_top, p_bot = split_loss(
            ezr, ezi, mask_in, mask_top, mask_bot,
            target_top=float(args.target_top),
        )

        # Soft box constraints on every learnable parameter.
        box_loss = (
            _box_penalty(widths_top, hw_lo, hw_hi)
            + _box_penalty(widths_bot, hw_lo, hw_hi)
            + _box_penalty(h_top, h_lo, h_hi)
            + _box_penalty(h_bot, h_lo, h_hi)
            + _box_penalty(l_bend_top, lb_lo, lb_hi)
            + _box_penalty(l_bend_bot, lb_lo, lb_hi)
            + _box_penalty(l_in, li_lo, li_hi)
            + _box_penalty(l_junction, lj_lo, lj_hi)
        )
        total_loss = loss + 1.0 * box_loss

        total_loss.backward()
        # Inlet half-widths are pinned to wg_width/2 every iteration.
        with torch.no_grad():
            if widths_top.grad is not None:
                widths_top.grad[0] = 0.0
            if widths_bot.grad is not None:
                widths_bot.grad[0] = 0.0
        optimizer.step()

        # Transmissions (T_k = P_k / P_in). Optimizer drives T_top -> target_top
        # and T_bot -> 1 - target_top, with T_top + T_bot ~ 1 (lossless ideal).
        p_in_safe = max(p_in, 1e-12)
        t_top = p_top / p_in_safe
        t_bot = p_bot / p_in_safe
        t_total = t_top + t_bot
        ratio_top_at_output = p_top / max(p_top + p_bot, 1e-12)
        msg = {
            "iter": it,
            "loss": float(loss.detach()),
            "box_loss": float(box_loss.detach()),
            "p_in": p_in, "p_top": p_top, "p_bot": p_bot,
            "T_top": t_top, "T_bot": t_bot, "T_total": t_total,
            "ratio_top_at_output": ratio_top_at_output,
            "target_top": float(args.target_top),
            "widths_top": [float(w) for w in widths_top.detach().cpu().numpy()],
            "widths_bot": [float(w) for w in widths_bot.detach().cpu().numpy()],
            "h_top": float(h_top.detach()),
            "h_bot": float(h_bot.detach()),
            "l_bend_top": float(l_bend_top.detach()),
            "l_bend_bot": float(l_bend_bot.detach()),
            "l_in": float(l_in.detach()),
            "l_junction": float(l_junction.detach()),
            "elapsed_s": time.perf_counter() - t_start,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        print(
            f"[inv-design] iter {it:03d}  loss={float(loss.detach()):.4e}  "
            f"P_in={p_in:.3f}  T_top={t_top:.3f}  T_bot={t_bot:.3f}  "
            f"T_tot={t_total:.3f}  (target T_top={float(args.target_top):.2f})  "
            f"h={float(h_top.detach()):.2f}/{float(h_bot.detach()):.2f}  "
            f"lb={float(l_bend_top.detach()):.2f}/{float(l_bend_bot.detach()):.2f}"
        )

        if (it % int(args.save_every)) == 0 or it == int(args.iters) - 1:
            state_dict = {
                "eps_phys": eps_phys.detach().cpu().numpy(),
                "ezr": ezr.detach().cpu().numpy(),
                "ezi": ezi.detach().cpu().numpy(),
                "mask_in": mask_in.detach().cpu().numpy(),
                "mask_top": mask_top.detach().cpu().numpy(),
                "mask_bot": mask_bot.detach().cpu().numpy(),
                "src_mask": src_mask.detach().cpu().numpy(),
                "widths_top": widths_top.detach().cpu().numpy(),
                "widths_bot": widths_bot.detach().cpu().numpy(),
                "scalars": {
                    "h_top": float(h_top.detach()),
                    "h_bot": float(h_bot.detach()),
                    "l_bend_top": float(l_bend_top.detach()),
                    "l_bend_bot": float(l_bend_bot.detach()),
                    "l_in": float(l_in.detach()),
                    "l_junction": float(l_junction.detach()),
                },
                "transmissions": {
                    "T_top": float(t_top), "T_bot": float(t_bot),
                    "T_total": float(t_total), "P_in": float(p_in),
                    "target_top": float(args.target_top),
                },
                "extent": extent,
            }
            _save_state(state_dict, out_dir / f"iter_{it:04d}.png")

    # Save final parameters + summary.
    summary = {
        "ckpt": str(ckpt_path),
        "wg_width_um": float(renderer.wg_width_um),
        "wavelength_um": float(args.wavelength_um),
        "num_steps": int(args.num_steps),
        "iters": int(args.iters),
        "target_top": float(args.target_top),
        "param_bounds": PARAM_BOUNDS,
        "final": {
            "widths_top_um": widths_top.detach().cpu().numpy().tolist(),
            "widths_bot_um": widths_bot.detach().cpu().numpy().tolist(),
            "h_top_um": float(h_top.detach()),
            "h_bot_um": float(h_bot.detach()),
            "l_bend_top_um": float(l_bend_top.detach()),
            "l_bend_bot_um": float(l_bend_bot.detach()),
            "l_in_um": float(l_in.detach()),
            "l_junction_um": float(l_junction.detach()),
            "P_in": float(p_in),
            "P_top": float(p_top),
            "P_bot": float(p_bot),
            "T_top": float(t_top),
            "T_bot": float(t_bot),
            "T_total": float(t_total),
        },
        "elapsed_s": time.perf_counter() - t_start,
    }

    # Optional FDTD verification on the converged geometry.
    if bool(args.verify_fdtd):
        try:
            fdtd = _verify_fdtd_at_final(
                args=args, renderer=renderer,
                widths_top=widths_top, widths_bot=widths_bot,
                h_top=h_top, h_bot=h_bot,
                l_bend_top=l_bend_top, l_bend_bot=l_bend_bot,
                l_in=l_in, l_junction=l_junction,
            )
            summary["fdtd_verification"] = {
                "P_in": float(fdtd["P_in"]),
                "P_top": float(fdtd["P_top"]), "P_bot": float(fdtd["P_bot"]),
                "T_top_mask": float(fdtd["T_top_mask"]),
                "T_bot_mask": float(fdtd["T_bot_mask"]),
                "T_total_mask": float(fdtd["T_total_mask"]),
                "S11_sq": float(fdtd["S11_sq"]),
                "S21_sq": float(fdtd["S21_sq"]),
                "S31_sq": float(fdtd["S31_sq"]),
                "T_top_modal": float(fdtd["T_top_modal"]),
                "T_bot_modal": float(fdtd["T_bot_modal"]),
                "T_total_modal": float(fdtd["T_total_modal"]),
                "elapsed_s": float(fdtd["elapsed_s"]),
                "target_top": float(args.target_top),
            }
            _save_verification_plot(
                eps_model=eps_phys.detach().cpu().numpy(),
                ezr_model=ezr.detach().cpu().numpy(),
                ezi_model=ezi.detach().cpu().numpy(),
                trans_model=dict(
                    T_top=float(t_top), T_bot=float(t_bot), T_total=float(t_total),
                ),
                fdtd=fdtd,
                extent=extent,
                target_top=float(args.target_top),
                out_path=out_dir / "verification.png",
            )
            np.savez_compressed(
                out_dir / "verification_fields.npz",
                eps_model=eps_phys.detach().cpu().numpy(),
                Ez_real_model=ezr.detach().cpu().numpy(),
                Ez_imag_model=ezi.detach().cpu().numpy(),
                eps_fdtd=fdtd["eps_fdtd"],
                Ez_real_fdtd=fdtd["Ez_real_fdtd"],
                Ez_imag_fdtd=fdtd["Ez_imag_fdtd"],
                target_top=np.float32(float(args.target_top)),
            )
            print(f"[inv-design] saved: {out_dir / 'verification.png'}")
            print(f"[inv-design] saved: {out_dir / 'verification_fields.npz'}")
        except Exception as e:
            print(f"[inv-design] FDTD verification FAILED: {e!r}")
            summary["fdtd_verification_error"] = repr(e)

    with open(out_dir / "final.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[inv-design] done in {summary['elapsed_s']:.1f}s; final widths -> {out_dir / 'final.json'}")


if __name__ == "__main__":
    main()
