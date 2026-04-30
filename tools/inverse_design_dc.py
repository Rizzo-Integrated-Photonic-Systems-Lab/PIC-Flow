#!/usr/bin/env python3
"""
Gradient-based inverse design of a directional-coupler power splitter.

Why a DC instead of a Y-branch for non-50/50 design:
  * The DC sweep covers gap in [0.10, 0.35] um and wg_length in [5.0, 8.0] um;
    coupled-mode theory gives the cross fraction as sin^2(kappa * L), which
    spans every split ratio across the trained design space. The model has
    been directly supervised on 70/30 (and every other) DC.
  * Just two essential degrees of freedom (gap, wg_length) instead of the
    32 dof we had on the asymmetric Y-branch. Smaller, in-distribution,
    physically interpretable.

Loss (transmission-anchored + lossless penalty), same form as the YBranch
script:
    T_top = P_top / P_in,  T_bot = P_bot / P_in
    L = (T_top - target_top)^2 + (T_bot - target_bot)^2
        + lossless_weight * (1 - (T_top + T_bot))^2

Port convention (DC source on port 1):
    port_in  = port 1   (top-left input)
    port_top = port 3   (top-right, BAR — "stay" channel)
    port_bot = port 4   (bot-right, CROSS — "coupled" channel)
So target_top = 0.7 means a 70/30 BAR/CROSS splitter.

FDTD verification at the end builds the trained DirectionalCoupler2D Meep
class with the converged (gap, wg_length) pair and reports modal |S31|^2,
|S41|^2 alongside the model's prediction.
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
)
from utils import neff_siwire_from_tables  # noqa: E402

# Reuse the heavy helpers from the YBranch script (model-call wrappers and
# cond-maps builder). The split-loss formulation here is different — see
# split_loss_decoupled below — because the YBranch's coupled (ratio + lossless
# both anchored on P_in) loss can drive ratio drift when the model produces
# fields with T_tot >> 1 (which can happen even on in-distribution DCs due
# to mask-vs-modal differences).
from inverse_design_ybranch_5050 import (  # noqa: E402
    build_cond_maps_diff,
    fm_sample_checkpointed,
)


def split_loss_decoupled(
    ezr: torch.Tensor, ezi: torch.Tensor,
    mask_in: torch.Tensor,
    mask_top: torch.Tensor, mask_bot: torch.Tensor,
    target_top: float = 0.7,
    eps_pin: float = 1e-6,
    lossless_weight: float = 0.01,
) -> tuple[torch.Tensor, float, float, float]:
    """Decoupled split loss for inverse design when mask-vs-modal scaling
    means T_tot != 1 even on in-distribution geometries.

        ratio_loss     = (P_top/(P_top+P_bot) - target_top)^2
                       + (P_bot/(P_top+P_bot) - (1 - target_top))^2
        lossless_loss  = ((P_top + P_bot) / P_in - 1)^2
        L              = ratio_loss + lossless_weight * lossless_loss

    Ratio term is normalized by the *output* total — scale-invariant in
    field amplitude, so the optimizer can match the split without fighting
    the model on absolute magnitudes. Lossless term is normalized by *input*
    port power and pushes the design toward energy conservation.

    Default lossless_weight is intentionally very small (0.01): for a DC
    in the trained sweep the model produces fields with mask T_tot >> 1
    even though modal |S31|^2 + |S41|^2 ~ 1, so a strong lossless term
    drags the ratio off-target. A small weight is enough to discourage
    degenerate "collapse all outputs to zero" minima without dominating
    the ratio objective.

    Returns (loss, P_in, P_top, P_bot).
    """
    if not (0.0 < float(target_top) < 1.0):
        raise ValueError(f"target_top must be in (0, 1); got {target_top}")
    target_bot = 1.0 - float(target_top)
    energy = ezr ** 2 + ezi ** 2  # [ny, nx]
    p_in = (mask_in * energy).sum()
    p_top = (mask_top * energy).sum()
    p_bot = (mask_bot * energy).sum()

    p_total_out_safe = p_top + p_bot + float(eps_pin)
    p_in_safe = p_in + float(eps_pin)

    f_top = p_top / p_total_out_safe
    f_bot = p_bot / p_total_out_safe
    t_total = (p_top + p_bot) / p_in_safe

    ratio_loss = (f_top - float(target_top)) ** 2 + (f_bot - target_bot) ** 2
    lossless_loss = (t_total - 1.0) ** 2
    loss = ratio_loss + float(lossless_weight) * lossless_loss
    return (
        loss,
        float(p_in.detach()),
        float(p_top.detach()),
        float(p_bot.detach()),
    )


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

# Fixed shell parameters (kept at trained-sweep midpoints; only gap and
# wg_length are learnable, since those are the two knobs that set the split).
SHELL = dict(
    wg_width_um=0.45,
    bend_length_um=5.0,
    lead_extra_gap_um=1.4,
)

# Initial values for the two learnable parameters and their soft bounds.
INIT_PARAMS = dict(
    gap_um=0.20,         # midpoint of trained [0.10, 0.35]
    wg_length_um=6.5,    # midpoint of trained [5.0, 8.0]
)
PARAM_BOUNDS = dict(
    gap_um=(0.10, 0.35),
    wg_length_um=(5.0, 8.0),
)


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


# ---------------------------------------------------------------------------
# Differentiable directional-coupler renderer
# ---------------------------------------------------------------------------
class DifferentiableDirectionalCoupler:
    """Renders a 4-port DC eps map (and source / port masks) as a differentiable
    function of (gap, wg_length). Layout:

       lead (top)      ___________________________
            \\ S-bend /  coupling region (top guide) \\ S-bend /  lead (top)
             ----o----                                 ----o----
                                gap
             ----o----                                 ----o----
            / S-bend \\  coupling region (bot guide) / S-bend \\  lead (bot)
                       ___________________________

    Coupling region centered at x=0; both bends use the same sine profile as
    DirectionalCoupler2D in FDTD/directional_coupler/. wg_width, bend_length,
    and lead_extra_gap are fixed at construction time.
    """

    def __init__(
        self,
        *,
        nx: int, ny: int, dx_um: float, dy_um: float, wavelength_um: float,
        wg_width_um: float = SHELL["wg_width_um"],
        bend_length_um: float = SHELL["bend_length_um"],
        lead_extra_gap_um: float = SHELL["lead_extra_gap_um"],
        port_y_pad_um: float = 0.4,
        source_shift_um: float = 0.5,
        edge_sigma_um: float = 0.04,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.nx = int(nx); self.ny = int(ny)
        self.dx_um = float(dx_um); self.dy_um = float(dy_um)
        self.wavelength_um = float(wavelength_um)
        self.wg_width_um = float(wg_width_um)
        self.bend_length_um = float(bend_length_um)
        self.lead_extra_gap_um = float(lead_extra_gap_um)
        self.port_y_pad_um = float(port_y_pad_um)
        self.source_shift_um = float(source_shift_um)
        self.edge_sigma_um = float(edge_sigma_um)

        self.Lx = self.nx * self.dx_um
        self.Ly = self.ny * self.dy_um
        self.x_left = -0.5 * self.Lx
        self.x_right = +0.5 * self.Lx

        # Extension past grid edges so eps is uniform-core at the cell boundary
        # (matches the trained DC where input/output leads run through the PML).
        self._extend_um = 1.0

        # Index materials.
        n_core = float(neff_siwire_from_tables(self.wg_width_um, self.wavelength_um))
        self.eps_core = float(n_core ** 2)
        self.eps_clad = float(N_CLAD ** 2)

        # Pixel-center coordinate grids (µm), [ny, nx] convention.
        ix = torch.arange(self.nx, device=device, dtype=dtype)
        iy = torch.arange(self.ny, device=device, dtype=dtype)
        x = (ix + 0.5) * self.dx_um - 0.5 * self.Lx
        y = (iy + 0.5) * self.dy_um - 0.5 * self.Ly
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")
        self.X = self.X.contiguous()
        self.Y = self.Y.contiguous()

    # ---- helpers --------------------------------------------------------
    def _sig(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z / self.edge_sigma_um)

    def _strip_indicator(self, x_lo, x_hi, y_center, half_width) -> torch.Tensor:
        in_x = self._sig(self.X - x_lo) * self._sig(x_hi - self.X)
        in_y = self._sig(half_width - torch.abs(self.Y - y_center))
        return in_x * in_y

    def _sine_sbend_indicator(
        self, x0, x1, y0, y1, half_w,
    ) -> torch.Tensor:
        """Sine-profile S-bend strip from (x0, y0) to (x1, y1), constant width.
        y(x) = y0 + h * (u - sin(2pi u)/(2pi))  where u = (X - x0) / L,
        L = x1 - x0, h = y1 - y0.  Matches DirectionalCoupler2D's s_bend_blocks.
        """
        L = x1 - x0
        L_safe = torch.clamp(L, min=1e-6) if isinstance(L, torch.Tensor) else max(L, 1e-6)
        h = y1 - y0
        s = self.X - x0
        u = torch.clamp(s / L_safe, 0.0, 1.0)
        y_centerline = y0 + h * (u - torch.sin(2.0 * math.pi * u) / (2.0 * math.pi))
        in_x = self._sig(self.X - x0) * self._sig(x1 - self.X)
        in_y = self._sig(half_w - torch.abs(self.Y - y_centerline))
        return in_x * in_y

    @staticmethod
    def _soft_union(masks: list[torch.Tensor]) -> torch.Tensor:
        out = torch.zeros_like(masks[0])
        for m in masks:
            out = out + m - out * m
        return out

    # ---- render ---------------------------------------------------------
    def render(
        self, *, gap: torch.Tensor, wg_length: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Geometric quantities (gradients flow through gap and wg_length).
        center_off = 0.5 * (self.wg_width_um + gap)             # |y| of coupling cores
        lead_off = center_off + 0.5 * self.lead_extra_gap_um    # |y| of input/output leads
        bend_half_w = 0.5 * self.wg_width_um

        x_couple_lo = -0.5 * wg_length
        x_couple_hi = +0.5 * wg_length
        x_left_bend_lo = x_couple_lo - self.bend_length_um
        x_right_bend_hi = x_couple_hi + self.bend_length_um

        masks: list[torch.Tensor] = []

        # 1) Input leads (top + bot), extending past the left grid edge.
        x_lead_left_lo = self.x_left - self._extend_um
        masks.append(self._strip_indicator(x_lead_left_lo, x_left_bend_lo, +lead_off, bend_half_w))
        masks.append(self._strip_indicator(x_lead_left_lo, x_left_bend_lo, -lead_off, bend_half_w))

        # 2) Left S-bends: lead-y -> coupling-y, sine profile.
        masks.append(self._sine_sbend_indicator(
            x_left_bend_lo, x_couple_lo, +lead_off, +center_off, bend_half_w,
        ))
        masks.append(self._sine_sbend_indicator(
            x_left_bend_lo, x_couple_lo, -lead_off, -center_off, bend_half_w,
        ))

        # 3) Coupling region (two parallel cores).
        masks.append(self._strip_indicator(x_couple_lo, x_couple_hi, +center_off, bend_half_w))
        masks.append(self._strip_indicator(x_couple_lo, x_couple_hi, -center_off, bend_half_w))

        # 4) Right S-bends: coupling-y -> lead-y.
        masks.append(self._sine_sbend_indicator(
            x_couple_hi, x_right_bend_hi, +center_off, +lead_off, bend_half_w,
        ))
        masks.append(self._sine_sbend_indicator(
            x_couple_hi, x_right_bend_hi, -center_off, -lead_off, bend_half_w,
        ))

        # 5) Output leads (top + bot), extending past the right grid edge.
        x_lead_right_hi = self.x_right + self._extend_um
        masks.append(self._strip_indicator(x_right_bend_hi, x_lead_right_hi, +lead_off, bend_half_w))
        masks.append(self._strip_indicator(x_right_bend_hi, x_lead_right_hi, -lead_off, bend_half_w))

        core_ind = self._soft_union(masks)
        eps_phys = self.eps_clad + (self.eps_core - self.eps_clad) * core_ind

        # Source and port masks. Trained DC convention (DirectionalCoupler2D
        # in FDTD/directional_coupler/) puts the input port at
        # nonpml_left + fit_margin = -11.5 (with fit_margin = 0.5), and the
        # source 0.5 µm upstream — i.e. *at the cell edge*. This is different
        # from the YBranch convention (which uses port_margin = 0.8), so we
        # use 0.5 here to match what the model was supervised on for DCs.
        port_x_left = float(self.x_left + 0.5)
        port_x_right = float(self.x_right - 0.5)

        port_y_span = self.wg_width_um + self.port_y_pad_um
        in_x_src = self._sig(self.X - (port_x_left - self.source_shift_um - 1.5 * self.dx_um)) * \
                   self._sig((port_x_left - self.source_shift_um + 1.5 * self.dx_um) - self.X)
        in_y_src = self._sig(0.5 * port_y_span - torch.abs(self.Y - lead_off))
        src_mask = in_x_src * in_y_src

        # Input port mask (port 1)
        in_x_p1 = self._sig(self.X - (port_x_left - 1.5 * self.dx_um)) * \
                  self._sig((port_x_left + 1.5 * self.dx_um) - self.X)
        port_mask_in = in_x_p1 * self._sig(0.5 * port_y_span - torch.abs(self.Y - lead_off))

        # Output ports (port 3 / BAR top-right; port 4 / CROSS bot-right)
        in_x_p_right = self._sig(self.X - (port_x_right - 1.5 * self.dx_um)) * \
                       self._sig((port_x_right + 1.5 * self.dx_um) - self.X)
        port_mask_top = in_x_p_right * self._sig(0.5 * port_y_span - torch.abs(self.Y - lead_off))
        port_mask_bot = in_x_p_right * self._sig(0.5 * port_y_span - torch.abs(self.Y + lead_off))

        return dict(
            eps_phys=eps_phys,
            src_mask=src_mask,
            port_mask_in=port_mask_in,
            port_mask_top=port_mask_top,
            port_mask_bot=port_mask_bot,
            lead_off=lead_off.detach() if isinstance(lead_off, torch.Tensor) else lead_off,
            center_off=center_off.detach() if isinstance(center_off, torch.Tensor) else center_off,
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save_state(state_dict: dict[str, Any], out_path: Path) -> None:
    """3-panel snapshot: eps + ports overlay, predicted |Ez|, scalar readout."""
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
    scalars = state_dict["scalars"]
    trans = state_dict.get("transmissions", {})

    mag = np.sqrt(ezr ** 2 + ezi ** 2)
    ext = state_dict["extent"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), constrained_layout=True)
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
    scalar_text = (
        f"gap = {scalars['gap_um']:.3f} $\\mu$m\n"
        f"$L_c$ = {scalars['wg_length_um']:.3f} $\\mu$m"
    )
    if trans:
        scalar_text += (
            f"\n$T_\\mathrm{{top}}$ = {trans['T_top']:.2f},  "
            f"$T_\\mathrm{{bot}}$ = {trans['T_bot']:.2f}"
            f"\n$T_\\mathrm{{tot}}$ = {trans['T_total']:.2f}  "
            f"(target $T_\\mathrm{{top}}$ = {trans['target_top']:.2f})"
        )
    axes[1].text(
        0.02, 0.98, scalar_text, transform=axes[1].transAxes,
        ha="left", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=2.5),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FDTD verification: just instantiate the existing trained DC class
# ---------------------------------------------------------------------------
def _verify_fdtd_at_final(
    *, args: argparse.Namespace, renderer: DifferentiableDirectionalCoupler,
    gap: torch.Tensor, wg_length: torch.Tensor,
) -> dict[str, Any]:
    print("[inv-design-dc] running FDTD verification (Meep) ...")
    t0 = time.perf_counter()

    from unified_sweep import build_device, run_sim_and_extract

    crop_x_px = int(args.crop_x_px); crop_y_px = int(args.crop_y_px)
    resolution = int(args.resolution); dpml = float(args.dpml)
    cell_x = float(crop_x_px) / float(resolution) + 2.0 * dpml
    cell_y = float(crop_y_px) / float(resolution) + 2.0 * dpml

    params = {
        "wg_width_um": float(renderer.wg_width_um),
        "gap_um": float(gap.detach()),
        "wg_length_um": float(wg_length.detach()),
        "bend_length_um": float(renderer.bend_length_um),
        "lead_extra_gap_um": float(renderer.lead_extra_gap_um),
    }
    dev = build_device(
        "directional_coupler", params,
        wavelength_um=float(args.wavelength_um),
        resolution=resolution, dpml=dpml,
        cell_x=cell_x, cell_y=cell_y,
        crop_x_px=crop_x_px, crop_y_px=crop_y_px,
    )
    eps_full, Ez_full, S, cell_size = run_sim_and_extract(
        dev, "directional_coupler", input_port=1, decay_tol=1e-5,
    )

    pml_px = int(round(dpml * resolution))
    if pml_px > 0:
        eps_fdtd = eps_full[pml_px:-pml_px, pml_px:-pml_px]
        Ez_fdtd = Ez_full[pml_px:-pml_px, pml_px:-pml_px]
    else:
        eps_fdtd = eps_full; Ez_fdtd = Ez_full

    # Phase-anchor on source ROI for amplitude consistency.
    from dataset import phase_anchor_mask
    rendered = renderer.render(gap=gap, wg_length=wg_length)
    src_bin = (rendered["src_mask"].detach().cpu().numpy() > 0.5)
    ezr_anc, ezi_anc, _ = phase_anchor_mask(
        Ez_fdtd.real.copy(), Ez_fdtd.imag.copy(), src_bin,
        eps_r=eps_fdtd, thr_eps=3.0,
    )

    # Mask-based powers (same masks the optimizer used).
    mask_in_np = rendered["port_mask_in"].detach().cpu().numpy()
    mask_top_np = rendered["port_mask_top"].detach().cpu().numpy()
    mask_bot_np = rendered["port_mask_bot"].detach().cpu().numpy()
    energy_fdtd = ezr_anc ** 2 + ezi_anc ** 2
    P_in_f = float((mask_in_np * energy_fdtd).sum())
    P_top_f = float((mask_top_np * energy_fdtd).sum())
    P_bot_f = float((mask_bot_np * energy_fdtd).sum())
    p_in_safe = max(P_in_f, 1e-12)

    # Modal S-parameters from Meep mode monitors. Source on port 1, so
    # |S31|^2 = BAR (top output), |S41|^2 = CROSS (bot output).
    S11_sq = float(abs(S.get("S11", 0.0 + 0j)) ** 2)
    S21_sq = float(abs(S.get("S21", 0.0 + 0j)) ** 2)
    S31_sq = float(abs(S.get("S31", 0.0 + 0j)) ** 2)
    S41_sq = float(abs(S.get("S41", 0.0 + 0j)) ** 2)

    elapsed = time.perf_counter() - t0
    print(
        f"[inv-design-dc] FDTD done in {elapsed:.1f}s.  "
        f"target T_top={float(args.target_top):.2f} | "
        f"mask: T_top={P_top_f/p_in_safe:.3f}, T_bot={P_bot_f/p_in_safe:.3f}, "
        f"T_tot={(P_top_f+P_bot_f)/p_in_safe:.3f} | "
        f"modal: |S31|²={S31_sq:.3f}, |S41|²={S41_sq:.3f}, sum={S31_sq+S41_sq:.3f}"
    )
    return dict(
        eps_fdtd=eps_fdtd.astype(np.float32),
        Ez_real_fdtd=ezr_anc.astype(np.float32),
        Ez_imag_fdtd=ezi_anc.astype(np.float32),
        P_in=P_in_f, P_top=P_top_f, P_bot=P_bot_f,
        T_top_mask=P_top_f / p_in_safe,
        T_bot_mask=P_bot_f / p_in_safe,
        T_total_mask=(P_top_f + P_bot_f) / p_in_safe,
        S11_sq=S11_sq, S21_sq=S21_sq, S31_sq=S31_sq, S41_sq=S41_sq,
        T_top_modal=S31_sq, T_bot_modal=S41_sq,
        T_total_modal=S31_sq + S41_sq,
        sparams_complex={k: complex(v) for k, v in S.items()},
        elapsed_s=elapsed,
    )


def _save_verification_plot(
    *, eps_model, ezr_model, ezi_model, trans_model,
    fdtd, extent, target_top: float, out_path: Path,
) -> None:
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

    axes[1, 0].imshow(eps_f, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[1, 0].set_title(r"FDTD $\varepsilon_r$")
    axes[1, 0].set_xlabel(r"$x$ ($\mu$m)")
    axes[1, 0].set_ylabel(r"$y$ ($\mu$m)")
    im_ff = axes[1, 1].imshow(mag_f, origin="lower", extent=extent, cmap="magma",
                              aspect="equal", vmin=0, vmax=vmax)
    axes[1, 1].contour(eps_f, levels=[0.5 * (eps_f.min() + eps_f.max())],
                       colors=["white"], linewidths=0.4, origin="lower", extent=extent)
    axes[1, 1].set_title(
        rf"FDTD $|E_z|$  $|S_{{31}}|^2$={fdtd['S31_sq']:.2f}, "
        rf"$|S_{{41}}|^2$={fdtd['S41_sq']:.2f}, sum={fdtd['S31_sq']+fdtd['S41_sq']:.2f}"
    )
    axes[1, 1].set_xlabel(r"$x$ ($\mu$m)")
    fig.colorbar(im_ff, ax=axes[1, 1], fraction=0.04, pad=0.02)

    fig.suptitle(
        rf"DC inverse-design verification (target $T_{{\rm top}}$ = {target_top:.2f})",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverse-design a directional-coupler power splitter via FM-surrogate backprop."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "inverse_design_dc"))
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--dpml", type=float, default=DEFAULT_DPML)
    parser.add_argument("--crop-x-px", type=int, default=DEFAULT_CROP_X_PX)
    parser.add_argument("--crop-y-px", type=int, default=DEFAULT_CROP_Y_PX)
    parser.add_argument("--num-steps", type=int, default=30,
                        help="FM sampler steps per opt iter. Heun => 2x model calls.")
    parser.add_argument("--time-grid", choices=("checkpoint", "linear", "quadratic"), default="linear")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Adam learning rate. With only 2 dofs, can use larger steps "
                             "than the YBranch script.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the noise sample x_0.")
    parser.add_argument("--device-runtime", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--target-top", type=float, default=0.7,
                        help="Target T_top = P_BAR / P_in (top-right output, port 3). "
                             "0.5 = 3dB coupler, 0.7 = 70/30 BAR/CROSS, 0.3 = 30/70.")
    parser.add_argument("--init-gap-um", type=float, default=INIT_PARAMS["gap_um"])
    parser.add_argument("--init-wg-length-um", type=float, default=INIT_PARAMS["wg_length_um"])
    parser.add_argument("--lossless-weight", type=float, default=0.01,
                        help="Weight on the lossless ((P_top+P_bot)/P_in - 1)^2 term. "
                             "Very small by default because the trained model produces "
                             "fields with mask T_tot >> 1 even in-distribution; raising "
                             "this can drag the ratio off-target.")
    parser.add_argument("--verify-fdtd", action=argparse.BooleanOptionalAction, default=False,
                        help="Run a final Meep verification on the converged geometry.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    runtime_device = _select_device(args.device_runtime)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inv-design-dc] checkpoint: {args.ckpt}")
    print(f"[inv-design-dc] runtime_device: {runtime_device}")
    print(f"[inv-design-dc] num_steps={args.num_steps}, iters={args.iters}, lr={args.lr}")
    print(f"[inv-design-dc] target T_top (BAR fraction) = {args.target_top}")

    # Load checkpoint + model.
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=runtime_device, weights_only=False)
    stats = ckpt["stats"]
    ckpt_args = ckpt.get("args")
    model = _build_model_from_checkpoint(ckpt, device=runtime_device)
    state_key, state = _checkpoint_state_dict(ckpt, use_ema=not bool(args.no_ema))
    model.load_state_dict(state, strict=True)
    print(f"[inv-design-dc] loaded weights: {state_key}")

    # Renderer.
    dx_um = 1.0 / float(args.resolution)
    renderer = DifferentiableDirectionalCoupler(
        nx=int(args.crop_x_px), ny=int(args.crop_y_px),
        dx_um=dx_um, dy_um=dx_um,
        wavelength_um=float(args.wavelength_um),
        device=runtime_device, dtype=torch.float32,
    )

    # Two learnable scalars.
    gap = torch.nn.Parameter(torch.tensor(
        float(args.init_gap_um), dtype=torch.float32, device=runtime_device,
    ))
    wg_length = torch.nn.Parameter(torch.tensor(
        float(args.init_wg_length_um), dtype=torch.float32, device=runtime_device,
    ))
    optimizer = torch.optim.Adam([gap, wg_length], lr=float(args.lr))

    # Conditioning vector (wavelength normalization).
    cond_vec = _build_cond_vector(float(args.wavelength_um), stats, device=runtime_device)
    lambda_um_t = torch.tensor([[float(args.wavelength_um)]],
                                device=runtime_device, dtype=torch.float32)

    # Fix the noise so iterations are comparable.
    gen = torch.Generator(device=runtime_device)
    gen.manual_seed(int(args.seed))
    x0 = torch.randn(
        (1, 2, int(args.crop_y_px), int(args.crop_x_px)),
        device=runtime_device, dtype=torch.float32, generator=gen,
    )

    extent = (
        -0.5 * renderer.Lx, 0.5 * renderer.Lx,
        -0.5 * renderer.Ly, 0.5 * renderer.Ly,
    )
    log_path = out_dir / "trajectory.jsonl"
    with open(log_path, "w") as f:
        f.write("")

    def _box_penalty(value: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (F.relu(lo - value) ** 2 + F.relu(value - hi) ** 2).sum()

    g_lo, g_hi = PARAM_BOUNDS["gap_um"]
    L_lo, L_hi = PARAM_BOUNDS["wg_length_um"]

    t_start = time.perf_counter()
    for it in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)

        rendered = renderer.render(gap=gap, wg_length=wg_length)
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

        loss, p_in, p_top, p_bot = split_loss_decoupled(
            ezr, ezi, mask_in, mask_top, mask_bot,
            target_top=float(args.target_top),
            lossless_weight=float(args.lossless_weight),
        )
        box_loss = _box_penalty(gap, g_lo, g_hi) + _box_penalty(wg_length, L_lo, L_hi)
        total_loss = loss + 1.0 * box_loss

        total_loss.backward()
        optimizer.step()

        p_in_safe = max(p_in, 1e-12)
        t_top = p_top / p_in_safe
        t_bot = p_bot / p_in_safe
        t_total = t_top + t_bot

        msg = {
            "iter": it,
            "loss": float(loss.detach()),
            "box_loss": float(box_loss.detach()),
            "p_in": p_in, "p_top": p_top, "p_bot": p_bot,
            "T_top": t_top, "T_bot": t_bot, "T_total": t_total,
            "target_top": float(args.target_top),
            "gap_um": float(gap.detach()),
            "wg_length_um": float(wg_length.detach()),
            "elapsed_s": time.perf_counter() - t_start,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        print(
            f"[inv-design-dc] iter {it:03d}  loss={float(loss.detach()):.4e}  "
            f"P_in={p_in:.2f}  T_top={t_top:.3f}  T_bot={t_bot:.3f}  T_tot={t_total:.3f}  "
            f"gap={float(gap.detach()):.3f}  Lc={float(wg_length.detach()):.3f}"
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
                "scalars": {
                    "gap_um": float(gap.detach()),
                    "wg_length_um": float(wg_length.detach()),
                },
                "transmissions": {
                    "T_top": float(t_top), "T_bot": float(t_bot),
                    "T_total": float(t_total),
                    "target_top": float(args.target_top),
                },
                "extent": extent,
            }
            _save_state(state_dict, out_dir / f"iter_{it:04d}.png")

    summary = {
        "ckpt": str(ckpt_path),
        "shell": SHELL,
        "wavelength_um": float(args.wavelength_um),
        "num_steps": int(args.num_steps),
        "iters": int(args.iters),
        "target_top": float(args.target_top),
        "param_bounds": PARAM_BOUNDS,
        "final": {
            "gap_um": float(gap.detach()),
            "wg_length_um": float(wg_length.detach()),
            "P_in": float(p_in),
            "P_top": float(p_top),
            "P_bot": float(p_bot),
            "T_top": float(t_top),
            "T_bot": float(t_bot),
            "T_total": float(t_total),
        },
        "elapsed_s": time.perf_counter() - t_start,
    }

    if bool(args.verify_fdtd):
        try:
            fdtd = _verify_fdtd_at_final(args=args, renderer=renderer, gap=gap, wg_length=wg_length)
            summary["fdtd_verification"] = {
                "P_in": float(fdtd["P_in"]),
                "P_top": float(fdtd["P_top"]), "P_bot": float(fdtd["P_bot"]),
                "T_top_mask": float(fdtd["T_top_mask"]),
                "T_bot_mask": float(fdtd["T_bot_mask"]),
                "T_total_mask": float(fdtd["T_total_mask"]),
                "S11_sq": float(fdtd["S11_sq"]), "S21_sq": float(fdtd["S21_sq"]),
                "S31_sq": float(fdtd["S31_sq"]), "S41_sq": float(fdtd["S41_sq"]),
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
                trans_model=dict(T_top=float(t_top), T_bot=float(t_bot), T_total=float(t_total)),
                fdtd=fdtd, extent=extent,
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
            print(f"[inv-design-dc] saved: {out_dir / 'verification.png'}")
            print(f"[inv-design-dc] saved: {out_dir / 'verification_fields.npz'}")
        except Exception as e:
            print(f"[inv-design-dc] FDTD verification FAILED: {e!r}")
            summary["fdtd_verification_error"] = repr(e)

    with open(out_dir / "final.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[inv-design-dc] done in {summary['elapsed_s']:.1f}s; "
          f"final -> {out_dir / 'final.json'}")


if __name__ == "__main__":
    main()
