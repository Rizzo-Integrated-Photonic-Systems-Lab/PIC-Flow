# flow_matching.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from grad_utils import generalized_image_to_b_xy_c  # kept for compatibility (may be used elsewhere)
from sparams_loss import sparam_loss, extract_sparams
from modal_sparams import extract_sparams_modal, sparam_loss_modal

# -----------------------------------------------------------------------------
# Joint training mask modes
# -----------------------------------------------------------------------------
MASK_MODE_FORWARD = "forward"    # eps fixed, fields noised (standard forward sim)
MASK_MODE_INVERSE = "inverse"    # fields fixed, eps noised (inverse design)
MASK_MODE_JOINT = "joint"        # both noised (joint generation)


def sample_mask_mode(forward_ratio: float = 0.5, inverse_ratio: float = 0.3) -> str:
    """Sample a mask mode for joint training."""
    r = torch.rand(1).item()
    if r < forward_ratio:
        return MASK_MODE_FORWARD
    if r < forward_ratio + inverse_ratio:
        return MASK_MODE_INVERSE
    return MASK_MODE_JOINT


def binarization_loss(eps_phys: torch.Tensor, eps_core: float = 12.25, eps_clad: float = 2.07) -> torch.Tensor:
    """
    Penalty that pushes generated eps toward binary (core or cladding).
    eps_norm in [0,1], penalty = eps_norm * (1 - eps_norm), maximized at 0.5.
    """
    eps_norm = ((eps_phys - eps_clad) / (eps_core - eps_clad + 1e-8)).clamp(0, 1)
    return (eps_norm * (1.0 - eps_norm)).mean()


# -----------------------------------------------------------------------------
# Flow-matching path parameters
# -----------------------------------------------------------------------------
SIG_MIN: float = 0.0  # if >0, endpoint at t=1 is x1 + SIG_MIN * x0 (noise floor)


# -----------------------------------------------------------------------------
# Path: x_t = (1 - (1 - s) t) x0 + t x1,  where s = sig_min
# Velocity: u_t = d/dt x_t = x1 - (1 - s) x0  (constant in t)
#
# IMPORTANT identities (used for training/sampling consistency):
#   x1 = (1 - s) x_t + (1 - (1 - s) t) u_t
#   At t=1: x(1) = x1 + s x0, and x1 = (1 - s) x(1) + s u(1)
# -----------------------------------------------------------------------------

def psi_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sig_min: float = SIG_MIN,
) -> torch.Tensor:
    """
    Interpolation path between x0 and x1:
        x_t = (1 - (1 - sig_min) * t) * x0 + t * x1
    """
    return (1.0 - (1.0 - sig_min) * t) * x_0 + t * x_1


def u_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    sig_min: float = SIG_MIN,
) -> torch.Tensor:
    """
    Target flow / velocity field for the above path:
        u_t = x1 - (1 - sig_min) * x0
    """
    return x_1 - (1.0 - sig_min) * x_0


def sample_t(
    x_1: torch.Tensor,
    mode: str = "uniform",
    loc: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Time sampling for FM training.
    Returns shape [B, 1, 1, 1] (broadcastable to images).

    Modes:
        "uniform"       — U[0, 1]  (original default)
        "logit_normal"  — t = sigmoid(loc + scale * z),  z ~ N(0,1)
                          Concentrates samples around sigmoid(loc) ≈ 0.5
                          and avoids the extremes, giving a bell-shaped
                          density on (0, 1).  scale controls spread.
        "mixture"       — 50% uniform + 50% logit_normal.
                          Ensures good coverage at t near 0 and 1 (where
                          quadratic ODE grids spend most steps) while still
                          concentrating training near t=0.5 for hard samples.
    """
    B = x_1.shape[0]
    if mode == "logit_normal":
        z = torch.randn((B,), device=x_1.device)
        t = torch.sigmoid(loc + scale * z)
    elif mode == "mixture":
        t_uniform = torch.rand((B,), device=x_1.device)
        z = torch.randn((B,), device=x_1.device)
        t_logit = torch.sigmoid(loc + scale * z)
        mask = torch.rand((B,), device=x_1.device) < 0.5
        t = torch.where(mask, t_uniform, t_logit)
    else:  # uniform
        t = torch.rand((B,), device=x_1.device)
    return t.view(B, 1, 1, 1)


# -----------------------------------------------------------------------------
# Mask / reduction utilities (robust to [B,H,W] and [B,1,H,W])
# -----------------------------------------------------------------------------

def _spatial_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=(-2, -1), keepdim=True)


def masked_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x and m must be broadcastable. Reduces over spatial dims only.
    Works for x: [B,H,W], [B,1,H,W], [B,C,H,W].
    """
    num = _spatial_sum(x * m)
    den = _spatial_sum(m).clamp_min(1.0)
    return num / den.clamp_min(eps)


def masked_normalize(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize x by its masked spatial mean magnitude.
    Returns x / mean(x under mask).
    """
    mu = masked_mean(x, m, eps=eps)
    return x / mu.clamp_min(eps)


def pml_mask_like(x: torch.Tensor, pml_cells: int = 0, margin: int = 4) -> torch.Tensor:
    """
    Create a [B,1,H,W] mask that is 1 in interior and 0 near borders (PML band + margin).
    If pml_cells == 0 -> all ones.
    """
    B = x.shape[0]
    H, W = x.shape[-2], x.shape[-1]
    pml_cells = int(pml_cells)

    if pml_cells <= 0:
        return torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype, requires_grad=False)

    p2 = min(pml_cells + margin, max(0, H // 2 - 1), max(0, W // 2 - 1))
    if p2 <= 0:
        return torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype, requires_grad=False)

    y = torch.arange(H, device=x.device).view(1, 1, H, 1)
    xg = torch.arange(W, device=x.device).view(1, 1, 1, W)
    m = ((y >= p2) & (y < H - p2) & (xg >= p2) & (xg < W - p2)).to(dtype=x.dtype)
    return m.expand(B, 1, H, W)


def device_mask_from_eps(eps_phys: torch.Tensor, thr: float = 3.0, dilate: int = 9) -> torch.Tensor:
    """
    eps_phys: [B,1,H,W] in physical (not normalized) eps units.
    Returns [B,1,H,W] mask of device/core (eps > thr), optionally dilated.
    """
    m = (eps_phys > float(thr)).to(eps_phys.dtype)
    if dilate and int(dilate) > 1:
        k = int(dilate)
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    return m


def amp_gate(E_true: torch.Tensor, m_focus: torch.Tensor, tau: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    """
    E_true: complex [B,H,W]
    m_focus: real [B,H,W]
    Returns real [B,H,W] mask that suppresses low-amplitude regions within m_focus.
    """
    abs_true = torch.abs(E_true).clamp_min(eps)  # [B,H,W]
    m = m_focus.to(dtype=abs_true.dtype)         # [B,H,W]
    thr = float(tau) * masked_mean(abs_true, m, eps=eps)  # [B,1,1]
    m_amp = (abs_true > thr).to(dtype=m.dtype)            # [B,H,W]
    return m_focus * m_amp


def phase_grad_loss(
    E_pred: torch.Tensor,
    E_true: torch.Tensor,
    m_focus: torch.Tensor,
    eps: float = 1e-8,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    Stable local phase-advance loss.
    Uses E_true amplitude in denominators to avoid blow-ups when E_pred is small.

    E_pred, E_true: complex [B,H,W]
    m_focus: real [B,H,W]
    """
    m = amp_gate(E_true, m_focus.to(E_true.real.dtype), tau=float(tau), eps=eps)  # [B,H,W]
    abs_true = torch.abs(E_true).clamp_min(eps)                                   # [B,H,W]

    # weights: |E_true|^2 normalized over masked region
    w = (abs_true ** 2).to(E_true.real.dtype)  # [B,H,W]
    w = masked_normalize(w, m, eps=eps)        # [B,H,W]

    # forward diffs
    dE_dx_pred = E_pred[:, :, 1:] - E_pred[:, :, :-1]
    dE_dx_true = E_true[:, :, 1:] - E_true[:, :, :-1]
    dE_dy_pred = E_pred[:, 1:, :] - E_pred[:, :-1, :]
    dE_dy_true = E_true[:, 1:, :] - E_true[:, :-1, :]

    # TRUE amplitude denom (left/top point)
    A2_true_x = (torch.abs(E_true[:, :, :-1]) ** 2).clamp_min(eps)
    A2_true_y = (torch.abs(E_true[:, :-1, :]) ** 2).clamp_min(eps)

    # phase-advance estimates (using TRUE reference is most stable)
    kx_pred = torch.imag(torch.conj(E_true[:, :, :-1]) * dE_dx_pred) / A2_true_x
    kx_true = torch.imag(torch.conj(E_true[:, :, :-1]) * dE_dx_true) / A2_true_x
    ky_pred = torch.imag(torch.conj(E_true[:, :-1, :]) * dE_dy_pred) / A2_true_y
    ky_true = torch.imag(torch.conj(E_true[:, :-1, :]) * dE_dy_true) / A2_true_y

    # slice masks/weights
    mx = m[:, :, :-1]; wx = w[:, :, :-1]
    my = m[:, :-1, :]; wy = w[:, :-1, :]

    loss_x = (wx * mx * (kx_pred - kx_true) ** 2).sum(dim=(1, 2)) / (wx * mx).sum(dim=(1, 2)).clamp_min(1.0)
    loss_y = (wy * my * (ky_pred - ky_true) ** 2).sum(dim=(1, 2)) / (wy * my).sum(dim=(1, 2)).clamp_min(1.0)

    return (loss_x + loss_y).mean()


def _get_pml_cells(model) -> int:
    m = model.module if hasattr(model, "module") else model
    p = getattr(m, "pml_cells", None)
    if p is not None:
        try:
            return int(p)
        except Exception:
            pass
    helm = getattr(m, "helmholtz", None)
    if helm is not None and hasattr(helm, "pml_cells"):
        try:
            return int(helm.pml_cells)
        except Exception:
            pass
    return 0


def _stats_float(stats: dict, key: str, default: float) -> float:
    v = stats.get(key, default)
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def cfm_loss_residual(
    model,
    x_t,                 # [B, 3+K, H, W] = [Re_t, Im_t, eps, (optional extra maps like src)]
    t,                   # [B,1,1,1] or [B] time
    v_t_fields,          # [B,2,H,W] target FM velocity in field-space (normalized)
    helmholtz_op,        # HelmholtzResidual2D instance (expects physical units)
    stats: dict,         # normalization stats
    eps_normalized: bool,
    fields_1,            # [B,2,H,W] ground truth final fields (normalized)
    cond=None,
    lambda_um=None,      # [B,1] physical wavelength (um)
    aux=None,            # dict from dataset (port_masks, sparams_true, etc.)
    compute_sparam: bool = False,
    device_focus: float = 0.0,
    w_min: float = 0.1,
    eps_thr: float = 6.0,
    dilate: int = 9,
    weight_residual: bool = True,
    compute_endpoint: bool = True,
    compute_phase_grad: bool = True,
    compute_phase: bool = True,
    phys_gate: float = 1.0,
    phase_gate: float = 1.0,
    compute_residual: bool = True,
    sig_min: float = SIG_MIN,
    # AMP: run only the model forward in fp16, keep all physics/loss math in fp32 for stability.
    amp_enabled: bool = False,
    amp_device_type: str = "cuda",
    amp_dtype: torch.dtype = torch.float16,
    sparam_mode: str = "project",
    # PBFM temporal unrolling
    unroll_steps: int = 0,
    unroll_phase: bool = False,
    phase_amp_tau: float = 0.2,
    # Joint training parameters
    joint_training: bool = False,
    mask_mode: str = "forward",
    v_t_eps: torch.Tensor = None,  # [B,1,H,W] target eps velocity
    eps_core: float = 12.25,
    eps_clad: float = 2.07,
    lambda_binarize: float = 0.0,
    eps_1: torch.Tensor = None,  # [B,1,H,W] GT clean eps (normalized) for geometry loss
    lambda_geom: float = 0.0,
    t_physics_min: float = 0.0,  # minimum t for physics losses (residual/phase/sparam); FM always at all t
    residual_t_power: float = 0.0,  # if >0, weight per-sample residual by t^p (upweights high-t = cleaner reconstructions)
    interface_thr: float = 0.0,  # if >0, mask out pixels where |∇ε| > thr from residual (removes interface artifacts)
):
    """
    Conditional FM loss + physics residual + phase loss + endpoint loss (+ optional sparam loss).
    x_t may include extra conditioning maps appended after eps (e.g. src mask).
    """
    B, C, H, W = x_t.shape
    device = x_t.device

    # Normalize t into [B] for model; keep view [B,1,1,1] for algebra
    if t.dim() == 1:
        t_vec = t
        t_view = t.view(B, 1, 1, 1)
    else:
        t_view = t.view(B, 1, 1, 1)
        t_vec = t_view.view(B)

    # Time-gating: only compute physics losses for samples where t >= threshold.
    # The x1 reconstruction is noisy at low t, producing uninformative physics gradients.
    if t_physics_min > 0.0:
        physics_mask = (t_vec >= t_physics_min).to(dtype=torch.float32)  # [B]
        physics_frac = physics_mask.mean().clamp_min(1e-8)  # scalar
    else:
        physics_mask = None
        physics_frac = None

    # wavelength
    lambda_um_model = None
    if lambda_um is not None:
        lambda_um_model = lambda_um.view(B, 1).to(device=device, dtype=x_t.dtype)

    # -----------------------
    # 1) Model prediction u_t
    # -----------------------
    with torch.autocast(amp_device_type, enabled=bool(amp_enabled), dtype=amp_dtype):
        want_sparam_head = (sparam_mode == "head") and bool(compute_sparam) and (aux is not None)

        if cond is None:
            if want_sparam_head:
                u_t_pred, S_pred_head = model(
                    x_t,
                    t_vec,
                    lambda_um=lambda_um_model,
                    phys_gate=phys_gate,
                    phase_gate=phase_gate,
                    return_sparams=True,
                    aux=aux,
                    sig_min=sig_min,
                )
            else:
                u_t_pred = model(
                    x_t,
                    t_vec,
                    lambda_um=lambda_um_model,
                    phys_gate=phys_gate,
                    phase_gate=phase_gate,
                    sig_min=sig_min,
                )
        else:
            if want_sparam_head:
                u_t_pred, S_pred_head = model(
                    x_t,
                    t_vec,
                    cond=cond,
                    lambda_um=lambda_um_model,
                    phys_gate=phys_gate,
                    phase_gate=phase_gate,
                    return_sparams=True,
                    aux=aux,
                    sig_min=sig_min,
                )
            else:
                u_t_pred = model(
                    x_t,
                    t_vec,
                    cond=cond,
                    lambda_um=lambda_um_model,
                    phys_gate=phys_gate,
                    phase_gate=phase_gate,
                    sig_min=sig_min,
                )

    # -----------------------
    # 2) Cast to fp32 for loss/physics stability
    # -----------------------
    x_t_f = x_t.to(dtype=torch.float32)
    u_t_pred_full = u_t_pred.to(dtype=torch.float32)
    v_t_fields_f = v_t_fields.to(dtype=torch.float32)
    fields_1_f = fields_1.to(dtype=torch.float32)

    # Split model output for joint training (model may output 2 or 3 channels)
    if joint_training and u_t_pred_full.shape[1] == 3:
        u_t_pred_f = u_t_pred_full[:, :2]       # field velocity [B,2,H,W]
        u_t_pred_eps_f = u_t_pred_full[:, 2:3]  # eps velocity [B,1,H,W]
    else:
        u_t_pred_f = u_t_pred_full[:, :2]
        u_t_pred_eps_f = None

    diff_fm = u_t_pred_f - v_t_fields_f  # [B,2,H,W]

    fields_t = x_t_f[:, 0:2]   # [B,2,H,W]
    eps_in   = x_t_f[:, 2:3]   # [B,1,H,W] (normalized or physical, depending on eps_normalized)

    # -----------------------
    # 3) Estimate clean x1 fields from (x_t, u_t_pred)
    #     x1 = (1-s) x_t + (1 - (1-s) t) u
    # -----------------------
    s = float(sig_min)
    a = (1.0 - s)                        # scalar
    b = (1.0 - a * t_view.to(torch.float32))  # [B,1,1,1]
    x1_fields_pred = a * fields_t + b * u_t_pred_f  # [B,2,H,W] (normalized)

    # -----------------------
    # 4) De-normalize to physical units for physics + phase comparison
    # -----------------------
    ez_real_mean = _stats_float(stats, "ez_real_mean", 0.0)
    ez_real_std  = _stats_float(stats, "ez_real_std", 1.0)
    ez_imag_mean = _stats_float(stats, "ez_imag_mean", 0.0)
    ez_imag_std  = _stats_float(stats, "ez_imag_std", 1.0)
    eps_mean     = _stats_float(stats, "eps_mean", 0.0)
    eps_std      = _stats_float(stats, "eps_std", 1.0)

    ez_real_mean_t = torch.tensor(ez_real_mean, device=device, dtype=torch.float32)
    ez_real_std_t  = torch.tensor(ez_real_std,  device=device, dtype=torch.float32)
    ez_imag_mean_t = torch.tensor(ez_imag_mean, device=device, dtype=torch.float32)
    ez_imag_std_t  = torch.tensor(ez_imag_std,  device=device, dtype=torch.float32)
    eps_mean_t     = torch.tensor(eps_mean,     device=device, dtype=torch.float32)
    eps_std_t      = torch.tensor(eps_std,      device=device, dtype=torch.float32)

    # eps is a conditioning map; do NOT reconstruct it from x1 algebra.
    eps_phys_only = eps_in * eps_std_t + eps_mean_t if bool(eps_normalized) else eps_in  # [B,1,H,W]

    # physical complex fields
    Er_pred_phys = x1_fields_pred[:, 0] * ez_real_std_t + ez_real_mean_t
    Ei_pred_phys = x1_fields_pred[:, 1] * ez_imag_std_t + ez_imag_mean_t
    Er_true_phys = fields_1_f[:, 0] * ez_real_std_t + ez_real_mean_t
    Ei_true_phys = fields_1_f[:, 1] * ez_imag_std_t + ez_imag_mean_t

    E_pred_phys_raw = torch.complex(Er_pred_phys, Ei_pred_phys)  # [B,H,W]
    E_true_phys     = torch.complex(Er_true_phys, Ei_true_phys)  # [B,H,W]

    # -----------------------
    # 5) Optional temporal unrolling (PBFM): integrate dx/dt = u_theta(x,t) from t -> 1
    # -----------------------
    x1_fields_pred_for_residual = x1_fields_pred
    x1_fields_phys_for_residual = None

    unroll_steps_i = int(unroll_steps) if unroll_steps is not None else 0
    if unroll_steps_i > 0:
        cond_maps = x_t_f[:, 2:]  # eps (+ optional src/extra maps); held fixed
        t0 = t_vec.to(dtype=torch.float32).clamp(0.0, 1.0)  # [B]

        base = torch.linspace(0.0, 1.0, unroll_steps_i + 1, device=device, dtype=torch.float32)
        tau = base ** 2
        t_grid = t0.view(B, 1) + (1.0 - t0).view(B, 1) * tau.view(1, -1)  # [B,N+1]

        x_new = fields_t  # [B,2,H,W] (normalized, fp32)

        def _unroll_model_call(x_fields, t_k):
            """Call model during unrolling, returning only the velocity tensor."""
            x_in = torch.cat([x_fields, cond_maps], dim=1).to(dtype=x_t.dtype)
            kwargs = dict(lambda_um=lambda_um_model, phys_gate=phys_gate,
                          phase_gate=phase_gate, sig_min=sig_min)
            if cond is not None:
                kwargs["cond"] = cond
            if want_sparam_head:
                kwargs["return_sparams"] = True
                kwargs["aux"] = aux
                out, _ = model(x_in, t_k, **kwargs)
            else:
                out = model(x_in, t_k, **kwargs)
            return out.to(dtype=torch.float32)

        with torch.autocast(amp_device_type, enabled=bool(amp_enabled), dtype=amp_dtype):
            for k in range(unroll_steps_i):
                tk0 = t_grid[:, k]
                tk1 = t_grid[:, k + 1]
                dt = (tk1 - tk0).view(B, 1, 1, 1)

                v0 = _unroll_model_call(x_new, tk0)
                v0_fields = v0[:, :2] if v0.shape[1] > 2 else v0
                x_euler = x_new + dt * v0_fields

                v1 = _unroll_model_call(x_euler, tk1)
                v1_fields = v1[:, :2] if v1.shape[1] > 2 else v1
                x_new = x_new + 0.5 * dt * (v0_fields + v1_fields)

            # If sig_min > 0, unrolled x_new is x(1) = x1 + s x0. Debias:
            #   x1 = (1 - s) x(1) + s u(1)
            if s != 0.0:
                ones = torch.ones((B,), device=device, dtype=torch.float32)
                u1 = _unroll_model_call(x_new, ones)
                u1_fields = u1[:, :2] if u1.shape[1] > 2 else u1
                x_new = (1.0 - s) * x_new + s * u1_fields

        x1_fields_pred_for_residual = x_new  # normalized, fp32

        # denormalize unrolled fields to physical units
        x1_fields_phys_for_residual = torch.stack(
            [
                x1_fields_pred_for_residual[:, 0] * ez_real_std_t + ez_real_mean_t,
                x1_fields_pred_for_residual[:, 1] * ez_imag_std_t + ez_imag_mean_t,
            ],
            dim=1,
        )  # [B,2,H,W]

    # -----------------------
    # 6) Build spatial masks/weights (PML + device + source-free)
    # -----------------------
    pml_cells = _get_pml_cells(model)
    pml_m = pml_mask_like(x_t_f, pml_cells=pml_cells, margin=4)  # [B,1,H,W]

    dev_m = device_mask_from_eps(eps_phys_only, thr=float(eps_thr), dilate=int(dilate))  # [B,1,H,W]

    # Extract source mask (channel 3 if present) and create source-free mask
    # Helmholtz equation ∇²E + k₀²εE = 0 only valid in source-free regions
    if C > 3:
        src_m = x_t_f[:, 3:4]  # [B,1,H,W] source mask
        # Dilate source region slightly to avoid edge effects
        src_dilated = F.max_pool2d(src_m, kernel_size=5, stride=1, padding=2)
        src_free_m = (src_dilated < 0.5).to(dtype=torch.float32)  # [B,1,H,W]
    else:
        src_free_m = torch.ones((B, 1, H, W), device=device, dtype=torch.float32)

    df = torch.tensor(float(device_focus), device=device, dtype=torch.float32)
    w0 = torch.tensor(float(w_min), device=device, dtype=torch.float32)

    w_fm = (w0 + df * dev_m) * pml_m  # [B,1,H,W] fp32 (for FM/endpoint loss)
    w_residual = w_fm * src_free_m     # [B,1,H,W] fp32 (for Helmholtz - excludes sources)

    # Mask out dielectric interface pixels where the 5-point Laplacian stencil
    # spans a sharp ε discontinuity (Si/SiO₂), producing spurious residuals.
    if float(interface_thr) > 0.0 and helmholtz_op is not None:
        grad_eps_x = helmholtz_op.diff.diff_x(eps_phys_only)  # [B,1,H,W]
        grad_eps_y = helmholtz_op.diff.diff_y(eps_phys_only)  # [B,1,H,W]
        grad_eps_mag = (grad_eps_x ** 2 + grad_eps_y ** 2).sqrt()
        interface_free = (grad_eps_mag < float(interface_thr)).to(dtype=torch.float32)
        w_residual = w_residual * interface_free

    m_focus = (pml_m * dev_m).squeeze(1)  # [B,H,W]

    # -----------------------
    # 7) Global phase alignment for endpoint/phase losses (NOT for sparams)
    # -----------------------
    need_align = bool(compute_endpoint or compute_phase or compute_phase_grad)

    if need_align:
        abs_true = torch.abs(E_true_phys).clamp_min(1e-8)  # [B,H,W]
        w_align = (pml_m.squeeze(1) * (w0 + df * dev_m.squeeze(1)) * (abs_true ** 2)).to(torch.float32)  # [B,H,W]

        dot = (w_align * torch.conj(E_pred_phys_raw) * E_true_phys).sum(dim=(1, 2))  # [B]
        mag = torch.abs(dot)
        rot = torch.where(mag > 1e-8, dot / mag, torch.ones_like(dot))
        rot_hw = rot.view(B, 1, 1)

        E_pred_phys_aligned = E_pred_phys_raw * rot_hw
        x1_fields_phys_aligned = torch.stack([E_pred_phys_aligned.real, E_pred_phys_aligned.imag], dim=1)  # [B,2,H,W]

        # Align normalized pred the same way for endpoint loss
        E_pred_norm = torch.complex(x1_fields_pred[:, 0], x1_fields_pred[:, 1]) * rot_hw
        x1_fields_pred_aligned = torch.stack([E_pred_norm.real, E_pred_norm.imag], dim=1)  # [B,2,H,W]
    else:
        x1_fields_phys_aligned = torch.stack([Er_pred_phys, Ei_pred_phys], dim=1)
        x1_fields_pred_aligned = x1_fields_pred

    # -----------------------
    # 8) Weighted FM loss
    # -----------------------
    num = (w_fm * (diff_fm ** 2)).sum(dim=(2, 3))  # [B,2]
    den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)      # [B,1]
    fm_loss = (num / den).mean()

    # Joint training: eps FM loss
    if joint_training and mask_mode != MASK_MODE_FORWARD and v_t_eps is not None and u_t_pred_eps_f is not None:
        v_t_eps_f = v_t_eps.to(dtype=torch.float32)
        fm_loss_eps = (u_t_pred_eps_f - v_t_eps_f).pow(2).mean()
        fm_loss = fm_loss + fm_loss_eps

    # -----------------------
    # Joint training: skip physics losses when eps is noised (inverse/joint modes)
    # In INVERSE mode fields are clean (x_t_fields = fields_1), so phase losses
    # remain valid and provide regularization + shared feature gradients.
    # Residual/sparam/endpoint depend on eps which IS noised, so skip those.
    # In JOINT mode both fields and eps are noised, so skip everything.
    # -----------------------
    if joint_training and mask_mode in (MASK_MODE_INVERSE, MASK_MODE_JOINT):
        compute_endpoint = False
        compute_residual = False
        compute_sparam = False
        if mask_mode == MASK_MODE_JOINT:
            compute_phase = False
            compute_phase_grad = False

    # Time-gating early-out: if no samples have t >= t_physics_min, skip all physics
    if physics_mask is not None and physics_mask.sum() == 0:
        compute_residual = False
        compute_phase = False
        compute_phase_grad = False
        compute_sparam = False

    # -----------------------
    # 9) Endpoint loss (aligned + unaligned blend)
    #    The aligned component ignores global phase offset.
    #    The unaligned component forces the model to learn correct global phase.
    #    Blend: endpoint_loss = 0.5 * aligned + 0.5 * unaligned
    # -----------------------
    if bool(compute_endpoint):
        # Aligned endpoint loss
        diff_end_aligned = x1_fields_pred_aligned - fields_1_f
        end_num_a = (w_fm * (diff_end_aligned ** 2)).sum(dim=(2, 3))
        end_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)
        per_sample_end_a = (end_num_a / end_den).mean(dim=-1)  # [B]

        # Unaligned endpoint loss (raw prediction vs GT — penalizes global phase offset)
        diff_end_raw = x1_fields_pred - fields_1_f
        end_num_r = (w_fm * (diff_end_raw ** 2)).sum(dim=(2, 3))
        per_sample_end_r = (end_num_r / end_den).mean(dim=-1)  # [B]

        # Blend: half aligned, half unaligned
        per_sample_end = 0.5 * per_sample_end_a + 0.5 * per_sample_end_r

        if physics_mask is not None:
            endpoint_loss = (per_sample_end * physics_mask).sum() / physics_mask.sum().clamp_min(1.0)
        else:
            endpoint_loss = per_sample_end.mean()
    else:
        endpoint_loss = torch.zeros((), device=device, dtype=torch.float32)

    # -----------------------
    # 10) Phase losses (physical fields)
    # -----------------------
    if bool(compute_phase) or bool(compute_phase_grad):
        E_true = E_true_phys
        E_pred = torch.complex(x1_fields_phys_aligned[:, 0], x1_fields_phys_aligned[:, 1])

        # Optionally use unrolled prediction for phase losses (aligned to GT)
        if bool(unroll_phase) and (x1_fields_phys_for_residual is not None):
            E_pred_u = torch.complex(
                x1_fields_phys_for_residual[:, 0],
                x1_fields_phys_for_residual[:, 1],
            )
            abs_true = torch.abs(E_true).clamp_min(1e-8)
            w_align_u = (pml_m.squeeze(1) * (w0 + df * dev_m.squeeze(1)) * (abs_true ** 2)).to(torch.float32)

            dot_u = (w_align_u * torch.conj(E_pred_u) * E_true).sum(dim=(1, 2))
            mag_u = torch.abs(dot_u)
            rot_u = torch.where(mag_u > 1e-8, dot_u / mag_u, torch.ones_like(dot_u))
            E_pred = E_pred_u * rot_u.view(B, 1, 1)
    else:
        E_pred = None
        E_true = None

    if bool(compute_phase_grad):
        phase_grad = phase_grad_loss(E_pred, E_true, m_focus, tau=float(phase_amp_tau))
        if physics_frac is not None:
            phase_grad = phase_grad * physics_frac
    else:
        phase_grad = torch.zeros((), device=device, dtype=torch.float32)

    if bool(compute_phase):
        abs_pred = torch.abs(E_pred).clamp_min(1e-8)
        abs_true = torch.abs(E_true).clamp_min(1e-8)
        u_pred = E_pred / abs_pred
        u_true = E_true / abs_true

        phase_err = 1.0 - torch.real(u_pred * torch.conj(u_true))  # [B,H,W]

        m_phase = amp_gate(E_true, m_focus.to(torch.float32), tau=float(phase_amp_tau), eps=1e-8)  # [B,H,W]
        w_phase = (abs_true ** 2).to(torch.float32)
        w_phase = masked_normalize(w_phase, m_phase, eps=1e-8)

        phase_num = (phase_err * w_phase * m_phase).sum(dim=(1, 2))
        phase_den = (w_phase * m_phase).sum(dim=(1, 2)).clamp_min(1.0)
        per_sample_phase = phase_num / phase_den  # [B]
        if physics_mask is not None:
            phase_loss = (per_sample_phase * physics_mask).sum() / physics_mask.sum().clamp_min(1.0)
        else:
            phase_loss = per_sample_phase.mean()
    else:
        phase_loss = torch.zeros((), device=device, dtype=torch.float32)

    # -----------------------
    # 11) Physics residual (Helmholtz) on physical fields
    # -----------------------
    if bool(compute_residual):
        if lambda_um is not None:
            k0 = (2.0 * torch.pi) / lambda_um.view(B).to(dtype=torch.float32)
        else:
            k0 = None

        if x1_fields_phys_for_residual is not None:
            x_fields_phys = x1_fields_phys_for_residual  # [B,2,H,W]
        else:
            x_fields_phys = x1_fields_phys_aligned        # [B,2,H,W]

        R = helmholtz_op(x_fields_phys, eps_phys_only, k0=k0)  # expected [B,2,H,W]
        R2 = (R[:, 0:1] ** 2 + R[:, 1:2] ** 2)                 # [B,1,H,W]

        # Use w_residual which excludes source regions (Helmholtz only valid source-free)
        if bool(weight_residual):
            norm_mask = w_residual
            res_num = (w_residual * R2).sum(dim=(2, 3))
            res_den = w_residual.sum(dim=(2, 3)).clamp_min(1.0)
        else:
            # Still mask out sources even without device weighting
            norm_mask = src_free_m
            res_num = (src_free_m * R2).sum(dim=(2, 3))
            res_den = src_free_m.sum(dim=(2, 3)).clamp_min(1.0)

        # Per-sample residual, then time-gated (and optionally t-weighted) average.
        # When residual_t_power > 0, samples with higher t (cleaner x1 reconstruction)
        # contribute more to the loss, effectively decoupling the residual from
        # noisy low-t reconstructions without requiring a second forward pass.
        per_sample_res = (res_num / res_den).squeeze(-1)  # [B]
        if residual_t_power > 0.0:
            t_w = t_vec.detach().clamp(0.0, 1.0) ** residual_t_power  # [B]
            if physics_mask is not None:
                tw = t_w * physics_mask
            else:
                tw = t_w
            residual_loss = (per_sample_res * tw).sum() / tw.sum().clamp_min(1e-8)
        elif physics_mask is not None:
            residual_loss = (per_sample_res * physics_mask).sum() / physics_mask.sum().clamp_min(1.0)
        else:
            residual_loss = per_sample_res.mean()

        # Normalize by driving-term scale: weighted-mean of (k₀²εE_true)².
        # The Helmholtz equation is ∇²E + k₀²εE = 0; the dominant term scale
        # is |k₀²εE|.  Dividing R² by this makes the loss:
        #   ≈ 0  when physics is satisfied,
        #   ≈ 1  when the residual is as large as the driving term.
        # This is automatically grid-independent (no external dx^4 needed)
        # and field-magnitude-independent.
        if k0 is not None:
            k0_sq = (k0 ** 2).view(B, 1, 1, 1)                                   # [B,1,1,1]
            E_true_mag_sq = (Er_true_phys ** 2 + Ei_true_phys ** 2).unsqueeze(1)  # [B,1,H,W]
            driving_sq = (k0_sq * eps_phys_only) ** 2 * E_true_mag_sq             # [B,1,H,W]
            driving_scale = (
                (norm_mask * driving_sq).sum(dim=(2, 3))
                / norm_mask.sum(dim=(2, 3)).clamp_min(1.0)
            ).mean().clamp_min(1e-12)
            residual_loss = residual_loss / driving_scale
    else:
        residual_loss = torch.zeros((), device=device, dtype=torch.float32)

    # -----------------------
    # 12) S-parameter loss (optional)
    # -----------------------
    if bool(compute_sparam) and (aux is not None):
        pm = aux.get("port_masks", None)
        st = aux.get("sparams_true", None)

        if (pm is not None) and (st is not None):
            # Use RAW predicted field so the model must get global phase right for S-params.
            if sparam_mode == "head":
                if "S_pred_head" not in locals():
                    raise RuntimeError("sparam_mode=head but S_pred_head was not produced by model forward.")
                S_pred = S_pred_head
                sparam_loss_val = sparam_loss(
                    S_pred,
                    st,
                    port_valid=aux.get("port_valid", None),
                    in_port_idx=aux.get("in_port_idx", None),
                    port_ids=aux.get("port_ids", None),
                )
            elif sparam_mode == "modal":
                # MEEP-style modal decomposition for S-parameter extraction
                # Prefer per-sample grid spacing from dataset aux; fallback to helmholtz_op.
                dx_aux = aux.get("grid_dx_um", None)
                if dx_aux is not None:
                    dx_arr = dx_aux.reshape(-1)
                    dx_um = float(dx_arr[0].item())
                else:
                    dx_um = float(getattr(helmholtz_op, 'dx', 1.0 / 24.0))

                # Get per-sample wavelengths [B] for correct mode solving per sample
                if lambda_um is not None:
                    wl_um = lambda_um.reshape(-1)  # [B] tensor, per-sample
                else:
                    # Fallback to omega from helmholtz_op
                    omega = float(getattr(helmholtz_op, 'omega', 2.0 * 3.14159 / 1.55))
                    wl_um = 2.0 * 3.14159 / omega  # scalar fallback

                # Get eps for mode solving - squeeze to [B, H, W]
                eps_for_modal = eps_phys_only.squeeze(1)  # [B, H, W]

                # Use modal S-parameter loss which computes extraction internally
                sparam_loss_val = sparam_loss_modal(
                    E_pred_phys_raw,  # [B, H, W] complex
                    E_true_phys,      # [B, H, W] complex (for reference if S_true not provided)
                    pm,
                    eps_for_modal,
                    wl_um,
                    dx_um,
                    S_true=st,
                    in_port_idx=aux.get("in_port_idx", None),
                    port_ids=aux.get("port_ids", None),
                    port_valid=aux.get("port_valid", None),
                )
            else:
                # "project" mode: simple averaging
                S_pred = extract_sparams(
                    E_pred_phys_raw,
                    pm,
                    in_port_idx=aux.get("in_port_idx", None),
                    port_ids=aux.get("port_ids", None),
                )
                sparam_loss_val = sparam_loss(
                    S_pred,
                    st,
                    port_valid=aux.get("port_valid", None),
                    in_port_idx=aux.get("in_port_idx", None),
                    port_ids=aux.get("port_ids", None),
                )
        else:
            # DDP-safe dummy use of head outputs when supervision missing
            if (sparam_mode == "head") and ("S_pred_head" in locals()):
                z = (S_pred_head.real.sum() + S_pred_head.imag.sum())
                sparam_loss_val = torch.zeros((), device=device, dtype=torch.float32) + (0.0 * z.to(torch.float32))
            else:
                sparam_loss_val = torch.zeros((), device=device, dtype=torch.float32)
    else:
        sparam_loss_val = torch.zeros((), device=device, dtype=torch.float32)

    # Apply time-gating to S-param loss
    if physics_frac is not None:
        sparam_loss_val = sparam_loss_val * physics_frac

    # -----------------------
    # 13) Binarization loss (joint training: push generated eps toward binary)
    # -----------------------
    binarize_loss_val = torch.zeros((), device=device, dtype=torch.float32)
    if joint_training and mask_mode != MASK_MODE_FORWARD and lambda_binarize > 0.0:
        if u_t_pred_eps_f is not None:
            # Reconstruct predicted eps: x1_eps = a * x_t_eps + b * v_eps
            eps_t = x_t_f[:, 2:3]  # noised eps (normalized)
            x1_eps_pred = a * eps_t + b * u_t_pred_eps_f  # normalized
            # De-normalize to physical
            x1_eps_phys = x1_eps_pred * eps_std_t + eps_mean_t if bool(eps_normalized) else x1_eps_pred
            binarize_loss_val = binarization_loss(x1_eps_phys, eps_core=eps_core, eps_clad=eps_clad)

    # -----------------------
    # 14) Geometry loss (pixel-wise MSE on reconstructed eps vs GT eps)
    # -----------------------
    geom_loss_val = torch.zeros((), device=device, dtype=torch.float32)
    if joint_training and mask_mode != MASK_MODE_FORWARD and lambda_geom > 0.0 and eps_1 is not None:
        if u_t_pred_eps_f is not None:
            # Reconstruct predicted eps (reuse from binarization if already computed)
            eps_t = x_t_f[:, 2:3]  # noised eps (normalized)
            x1_eps_pred_g = a * eps_t + b * u_t_pred_eps_f  # normalized
            # De-normalize both to physical units
            x1_eps_phys_g = x1_eps_pred_g * eps_std_t + eps_mean_t if bool(eps_normalized) else x1_eps_pred_g
            eps_1_f = eps_1.to(dtype=torch.float32)
            eps_1_phys = eps_1_f * eps_std_t + eps_mean_t if bool(eps_normalized) else eps_1_f
            geom_loss_val = F.mse_loss(x1_eps_phys_g, eps_1_phys)

    return fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad, sparam_loss_val, binarize_loss_val, geom_loss_val


def sample(
    ema,
    x_0: torch.Tensor,           # noise over *fields only*: [B, 2, H, W]
    num_steps: int,
    use_stoc_samp: bool,         # kept for API compatibility (currently unused)
    cond_maps: torch.Tensor,     # fixed maps, e.g. [B,1,H,W]=[eps] or [B,2,H,W]=[eps,src]
    cond=None,
    lambda_um=None,
    phys_gate=1.0,
    phase_gate=1.0,
    sig_min: float = SIG_MIN,
    time_grid: str = "quadratic",
) -> torch.Tensor:
    """
    Flow-matching sampler for fields, conditioned on spatial maps (eps, and optionally src).
    Integrates dx/dt = u_theta(x,t) on t in [0,1] using RK2/Heun.

    time_grid:
        "quadratic" — base**2, allocates more steps near t=0 (original)
        "linear"    — uniform spacing, matches uniform/mixture training distribution

    If sig_min > 0, returns clean x1 via the exact debias identity:
        x1 = (1 - s) x(1) + s u(1).
    """
    assert cond_maps is not None, "cond_maps must be provided (at least eps)"
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]

    cond_maps = cond_maps.to(device=device, dtype=dtype)

    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if time_grid == "quadratic":
        time_steps = base ** 2  # allocate more steps near t=0
    else:
        time_steps = base  # linear spacing

    x_new = x_0.clone()

    lambda_um_model = None
    if lambda_um is not None:
        lambda_um_model = lambda_um.view(B, 1).to(device=device, dtype=dtype)

    for k in range(num_steps):
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        dt = (t1 - t0)

        t_vec0 = t0.expand(B)
        t_vec1 = t1.expand(B)

        x_in0 = torch.cat([x_new, cond_maps], dim=1)
        if cond is None:
            v0 = ema(
                x_in0, t_vec0,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )
        else:
            v0 = ema(
                x_in0, t_vec0,
                cond=cond,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )

        # Slice to field channels only (model may output 3 ch in joint mode)
        n_ch = x_new.shape[1]
        v0 = v0[:, :n_ch].clone()  # clone: CUDAGraphs reuses output buffer

        x_euler = x_new + dt * v0

        x_in1 = torch.cat([x_euler, cond_maps], dim=1)
        if cond is None:
            v1 = ema(
                x_in1, t_vec1,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )
        else:
            v1 = ema(
                x_in1, t_vec1,
                cond=cond,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )
        v1 = v1[:, :n_ch]

        x_new = x_new + 0.5 * dt * (v0 + v1)

    # If sig_min > 0, x_new is x(1) = x1 + s x0. Debias to clean x1:
    #   x1 = (1 - s) x(1) + s u(1)
    s = float(sig_min)
    if s != 0.0:
        t_vec = torch.ones((B,), device=device, dtype=dtype)
        x_in = torch.cat([x_new, cond_maps], dim=1)
        if cond is None:
            u1 = ema(
                x_in, t_vec,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )
        else:
            u1 = ema(
                x_in, t_vec,
                cond=cond,
                lambda_um=lambda_um_model,
                phys_gate=phys_gate,
                phase_gate=phase_gate,
                sig_min=sig_min,
            )
        u1 = u1[:, :n_ch]
        x_new = (1.0 - s) * x_new + s * u1

    return x_new


def _model_call(ema, x_in, t_vec, cond, lambda_um_model, phys_gate, phase_gate, sig_min):
    """Helper to call model with or without cond."""
    kwargs = dict(lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate, sig_min=sig_min)
    if cond is not None:
        kwargs["cond"] = cond
    return ema(x_in, t_vec, **kwargs)


def sample_joint(
    ema,
    x_0: torch.Tensor,           # noise: [B, 3, H, W] = [field_noise(2), eps_noise(1)]
    num_steps: int,
    src_mask: torch.Tensor,       # [B, 1, H, W] source mask (fixed conditioning)
    cond=None,                    # [B, cond_dim] conditioning vector (includes S-params)
    cond_uncond=None,             # [B, cond_dim] unconditional cond (S-param entries zeroed) for CFG
    lambda_um=None,
    phys_gate=1.0,
    phase_gate=1.0,
    sig_min: float = SIG_MIN,
    cfg_scale: float = 1.0,       # CFG scale (1.0 = no guidance)
    inpaint_fields: torch.Tensor = None,  # [B,2,H,W] clean fields to inpaint (inverse mode)
    inpaint_eps: torch.Tensor = None,     # [B,1,H,W] clean eps to inpaint (forward mode)
) -> torch.Tensor:
    """
    Joint flow-matching sampler for 3-channel generation (fields + eps).

    Supports:
    - Full joint generation (both from noise)
    - Inpainting: fix fields or eps channels at each ODE step
    - Classifier-free guidance (CFG) on S-param conditioning
    """
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]

    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2

    x_new = x_0.clone()  # [B,3,H,W]

    lambda_um_model = None
    if lambda_um is not None:
        lambda_um_model = lambda_um.view(B, 1).to(device=device, dtype=dtype)

    use_cfg = (cfg_scale != 1.0) and (cond_uncond is not None)

    for k in range(num_steps):
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        dt = (t1 - t0)

        t_vec0 = t0.expand(B)
        t_vec1 = t1.expand(B)

        # Inpainting: replace fixed channels with their deterministic interpolation at time t
        if inpaint_fields is not None:
            # x_t_fields = (1 - (1-s)*t) * noise + t * clean
            noise_fields = x_0[:, :2]
            x_new[:, :2] = psi_t(noise_fields, inpaint_fields, t0.view(1, 1, 1, 1))
        if inpaint_eps is not None:
            noise_eps = x_0[:, 2:3]
            x_new[:, 2:3] = psi_t(noise_eps, inpaint_eps, t0.view(1, 1, 1, 1))

        # Model input: [x_t_fields(2), x_t_eps(1), src(1)] = 4 channels
        x_in0 = torch.cat([x_new, src_mask], dim=1)

        if use_cfg:
            # Two forward passes for CFG
            v0_cond = _model_call(ema, x_in0, t_vec0, cond, lambda_um_model, phys_gate, phase_gate, sig_min)
            v0_uncond = _model_call(ema, x_in0, t_vec0, cond_uncond, lambda_um_model, phys_gate, phase_gate, sig_min)
            v0 = v0_uncond + cfg_scale * (v0_cond - v0_uncond)
        else:
            v0 = _model_call(ema, x_in0, t_vec0, cond, lambda_um_model, phys_gate, phase_gate, sig_min)

        x_euler = x_new + dt * v0

        # Inpainting at t1
        if inpaint_fields is not None:
            noise_fields = x_0[:, :2]
            x_euler[:, :2] = psi_t(noise_fields, inpaint_fields, t1.view(1, 1, 1, 1))
        if inpaint_eps is not None:
            noise_eps = x_0[:, 2:3]
            x_euler[:, 2:3] = psi_t(noise_eps, inpaint_eps, t1.view(1, 1, 1, 1))

        x_in1 = torch.cat([x_euler, src_mask], dim=1)

        if use_cfg:
            v1_cond = _model_call(ema, x_in1, t_vec1, cond, lambda_um_model, phys_gate, phase_gate, sig_min)
            v1_uncond = _model_call(ema, x_in1, t_vec1, cond_uncond, lambda_um_model, phys_gate, phase_gate, sig_min)
            v1 = v1_uncond + cfg_scale * (v1_cond - v1_uncond)
        else:
            v1 = _model_call(ema, x_in1, t_vec1, cond, lambda_um_model, phys_gate, phase_gate, sig_min)

        x_new = x_new + 0.5 * dt * (v0 + v1)

    # Final inpainting at t=1
    if inpaint_fields is not None:
        x_new[:, :2] = inpaint_fields
    if inpaint_eps is not None:
        x_new[:, 2:3] = inpaint_eps

    # Debias if sig_min > 0
    s = float(sig_min)
    if s != 0.0:
        t_vec = torch.ones((B,), device=device, dtype=dtype)
        x_in = torch.cat([x_new, src_mask], dim=1)
        u1 = _model_call(ema, x_in, t_vec, cond, lambda_um_model, phys_gate, phase_gate, sig_min)
        x_new = (1.0 - s) * x_new + s * u1

    return x_new  # [B,3,H,W]


def sample_inverse(
    ema,
    num_steps: int,
    src_mask: torch.Tensor,       # [B,1,H,W]
    cond: torch.Tensor,           # [B,cond_dim] with S-params filled
    lambda_um=None,
    cfg_scale: float = 3.0,
    sig_min: float = SIG_MIN,
    base_cond_dim: int = 4,       # first N entries are wavelength+geom (not S-params)
) -> torch.Tensor:
    """
    Convenience wrapper for inverse design: generate eps (and fields) from target S-params.

    Returns [B, 3, H, W] = [predicted_fields(2), predicted_eps(1)]
    """
    device = src_mask.device
    dtype = src_mask.dtype
    B = src_mask.shape[0]
    H, W = src_mask.shape[-2], src_mask.shape[-1]

    # Start from pure noise for all 3 channels
    x_0 = torch.randn(B, 3, H, W, device=device, dtype=dtype)

    # Build unconditional cond for CFG (zero only S-param Re/Im, keep port_valid)
    cond_uncond = cond.clone()
    n_sparam_reals = 2 * 4  # Re/Im for max 4 ports
    cond_uncond[:, base_cond_dim:base_cond_dim + n_sparam_reals] = 0.0

    return sample_joint(
        ema,
        x_0,
        num_steps=num_steps,
        src_mask=src_mask,
        cond=cond,
        cond_uncond=cond_uncond,
        lambda_um=lambda_um,
        phys_gate=1.0,
        phase_gate=1.0,
        sig_min=sig_min,
        cfg_scale=cfg_scale,
        inpaint_fields=None,  # generate everything
        inpaint_eps=None,
    )
