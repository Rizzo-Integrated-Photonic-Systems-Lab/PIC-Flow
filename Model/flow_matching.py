# flow_matching.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from grad_utils import generalized_image_to_b_xy_c  # kept for compatibility (may be used elsewhere)
from sparams_loss import sparam_loss, extract_sparams

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


def sample_t(x_1: torch.Tensor) -> torch.Tensor:
    """
    Time sampling for FM training.
    50% Uniform, 50% Beta(2,2)
    Returns shape [B, 1, 1, 1] (broadcastable to images).
    """
    B = x_1.shape[0]
    dev = x_1.device
    u = torch.rand((B,), device=dev)
    dist = torch.distributions.Beta(
        torch.tensor(2.0, device=dev),
        torch.tensor(2.0, device=dev),
    )
    b = dist.sample((B,))
    t = torch.where(torch.rand((B,), device=dev) < 0.5, u, b)
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
    u_t_pred_f = u_t_pred.to(dtype=torch.float32)
    v_t_fields_f = v_t_fields.to(dtype=torch.float32)
    fields_1_f = fields_1.to(dtype=torch.float32)

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

        with torch.autocast(amp_device_type, enabled=bool(amp_enabled), dtype=amp_dtype):
            for k in range(unroll_steps_i):
                tk0 = t_grid[:, k]
                tk1 = t_grid[:, k + 1]
                dt = (tk1 - tk0).view(B, 1, 1, 1)

                x_in0 = torch.cat([x_new, cond_maps], dim=1).to(dtype=x_t.dtype)
                if want_sparam_head:
                    if cond is None:
                        v0, _ = model(
                            x_in0, tk0,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                    else:
                        v0, _ = model(
                            x_in0, tk0,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                else:
                    if cond is None:
                        v0 = model(
                            x_in0, tk0,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )
                    else:
                        v0 = model(
                            x_in0, tk0,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )

                v0 = v0.to(dtype=torch.float32)
                x_euler = x_new + dt * v0

                x_in1 = torch.cat([x_euler, cond_maps], dim=1).to(dtype=x_t.dtype)
                if want_sparam_head:
                    if cond is None:
                        v1, _ = model(
                            x_in1, tk1,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                    else:
                        v1, _ = model(
                            x_in1, tk1,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                else:
                    if cond is None:
                        v1 = model(
                            x_in1, tk1,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )
                    else:
                        v1 = model(
                            x_in1, tk1,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )

                v1 = v1.to(dtype=torch.float32)
                x_new = x_new + 0.5 * dt * (v0 + v1)

            # If sig_min > 0, unrolled x_new is x(1) = x1 + s x0. Debias:
            #   x1 = (1 - s) x(1) + s u(1)
            if s != 0.0:
                ones = torch.ones((B,), device=device, dtype=torch.float32)
                x_in = torch.cat([x_new, cond_maps], dim=1).to(dtype=x_t.dtype)
                if want_sparam_head:
                    if cond is None:
                        u1, _ = model(
                            x_in, ones,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                    else:
                        u1, _ = model(
                            x_in, ones,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            return_sparams=True,
                            aux=aux,
                            sig_min=sig_min,
                        )
                else:
                    if cond is None:
                        u1 = model(
                            x_in, ones,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )
                    else:
                        u1 = model(
                            x_in, ones,
                            cond=cond,
                            lambda_um=lambda_um_model,
                            phys_gate=phys_gate,
                            phase_gate=phase_gate,
                            sig_min=sig_min,
                        )
                u1 = u1.to(dtype=torch.float32)
                x_new = (1.0 - s) * x_new + s * u1

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
    # 6) Build spatial masks/weights (PML + device)
    # -----------------------
    pml_cells = _get_pml_cells(model)
    pml_m = pml_mask_like(x_t_f, pml_cells=pml_cells, margin=4)  # [B,1,H,W]

    dev_m = device_mask_from_eps(eps_phys_only, thr=float(eps_thr), dilate=int(dilate))  # [B,1,H,W]

    df = torch.tensor(float(device_focus), device=device, dtype=torch.float32)
    w0 = torch.tensor(float(w_min), device=device, dtype=torch.float32)

    w_fm = (w0 + df * dev_m) * pml_m  # [B,1,H,W] fp32
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

    # -----------------------
    # 9) Endpoint loss (normalized, aligned)
    # -----------------------
    if bool(compute_endpoint):
        diff_end = x1_fields_pred_aligned - fields_1_f
        end_num = (w_fm * (diff_end ** 2)).sum(dim=(2, 3))
        end_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)
        endpoint_loss = (end_num / end_den).mean()
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
        phase_loss = (phase_num / phase_den).mean()
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

        if bool(weight_residual):
            res_num = (w_fm * R2).sum(dim=(2, 3))
            res_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)
            residual_loss = (res_num / res_den).mean()
        else:
            residual_loss = R2.mean()
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
            else:
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

    return fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad, sparam_loss_val


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
) -> torch.Tensor:
    """
    Flow-matching sampler for fields, conditioned on spatial maps (eps, and optionally src).
    Integrates dx/dt = u_theta(x,t) on t in [0,1] using RK2/Heun with a quadratic time grid.
    If sig_min > 0, returns clean x1 via the exact debias identity:
        x1 = (1 - s) x(1) + s u(1).
    """
    assert cond_maps is not None, "cond_maps must be provided (at least eps)"
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]

    cond_maps = cond_maps.to(device=device, dtype=dtype)

    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2  # allocate more steps near t=0

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
        x_new = (1.0 - s) * x_new + s * u1

    return x_new
