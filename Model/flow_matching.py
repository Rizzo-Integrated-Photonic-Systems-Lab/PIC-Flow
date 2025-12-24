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


def psi_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sig_min: float = SIG_MIN,
) -> torch.Tensor:
    """
    Interpolation path between x0 and x1.

    psi_t = (1 - (1 - sig_min)*t) * x0 + t * x1
    """
    return (1 - (1 - sig_min) * t) * x_0 + t * x_1


def u_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    sig_min: float = SIG_MIN,
) -> torch.Tensor:
    """
    Target flow / velocity field for this path.
    """
    return x_1 - (1 - sig_min) * x_0


def sample_t(x_1: torch.Tensor) -> torch.Tensor:
    """
    Time sampling for FM training.
    50% Uniform, 50% Beta(2,2)
    """
    B = x_1.shape[0]
    dev = x_1.device
    u = torch.rand((B,), device=dev)
    dist = torch.distributions.Beta(torch.tensor(2.0, device=dev), torch.tensor(2.0, device=dev))
    b = dist.sample((B,))
    t = torch.where(torch.rand((B,), device=dev) < 0.5, u, b)
    return t.view(B, *([1] * (x_1.dim() - 1)))


def logit_normal_map(t: torch.Tensor, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    eps = 1e-6
    t = torch.clamp(t, eps, 1 - eps)
    return t * torch.exp(-0.5 * torch.square((torch.log(t / (1 - t)) - m) / s)) / (
        s * torch.sqrt(2 * torch.tensor(torch.pi, device=t.device, dtype=t.dtype)) * t * (1 - t) ** 2
    )


def spatial_grid(x: torch.Tensor):
    """
    Simple finite-difference spatial gradients
    x: [B, C, H, W]

    Returns:
        dx, dy with shapes [B, C, H, W-1] and [B, C, H-1, W]
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def pml_mask_like(x: torch.Tensor, pml_cells: int = 30, margin: int = 4) -> torch.Tensor:
    # x: [B,*,H,W]
    B = x.shape[0]
    H, W = x.shape[-2], x.shape[-1]
    p2 = min(pml_cells + margin, max(0, H // 2 - 1), max(0, W // 2 - 1))
    m = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
    if p2 > 0:
        m[:, :, :p2, :] = 0
        m[:, :, -p2:, :] = 0
        m[:, :, :, :p2] = 0
        m[:, :, :, -p2:] = 0
    return m


def device_mask_from_eps(eps_phys: torch.Tensor, thr: float = 3.0, dilate: int = 9) -> torch.Tensor:
    m = (eps_phys > thr).to(eps_phys.dtype)
    if dilate and dilate > 1:
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate // 2)
    return m


def masked_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8):
    num = (x * m).sum(dim=(1, 2), keepdim=True)
    den = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return num / den.clamp_min(eps)


def masked_normalize(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8):
    mu = masked_mean(x, m, eps=eps)
    return x / mu.clamp_min(eps)


def amp_gate(E_true: torch.Tensor, m_focus: torch.Tensor, tau: float = 0.2, eps: float = 1e-8):
    # E_true: complex [B,H,W], m_focus: [B,H,W]
    abs_true = torch.abs(E_true).clamp_min(eps)
    thr = tau * abs_true.mean(dim=(1, 2), keepdim=True)
    m_amp = (abs_true > thr).to(m_focus.dtype)
    return m_focus * m_amp


def align_global_phase(E_pred: torch.Tensor, E_true: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    """
    Find per-sample complex rotation r (|r|=1) that best aligns E_pred to E_true under weights w,
    then return E_pred_aligned = r * E_pred.

    E_pred, E_true: complex tensors [B,H,W]
    w: real weights [B,H,W]
    """
    dot = (w * torch.conj(E_pred) * E_true).sum(dim=(1, 2))  # [B]
    rot = dot / torch.abs(dot).clamp_min(eps)                # [B]
    return E_pred * rot.view(-1, 1, 1)


def phase_grad_loss(
    E_pred: torch.Tensor,
    E_true: torch.Tensor,
    m_focus: torch.Tensor,
    eps: float = 1e-8,
    tau: float = 0.2,
):
    """
    Stable local phase-advance loss.
    Uses E_true amplitude in denominators to avoid blow-ups when E_pred is small.
    """
    m = amp_gate(E_true, m_focus.to(E_true.real.dtype), tau=tau, eps=eps)  # [B,H,W]

    abs_true = torch.abs(E_true).clamp_min(eps)

    # weights: |E_true|^2 normalized over masked region
    w = (abs_true ** 2).to(E_true.real.dtype)
    w = masked_normalize(w, m, eps=eps)

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
    aux=None,            # <--- NEW: dict from dataset (port_masks, sparams_true, etc.)
    compute_sparam: bool = False,  # <--- NEW
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
):
    """
    Conditional FM loss + physics residual + phase loss + endpoint loss (+ optional sparam loss).
    Works with x_t that has extra conditioning maps (e.g., src mask) appended after eps.
    """
    B, C, H, W = x_t.shape
    device = x_t.device
    dtype = x_t.dtype

    # -----------------------
    # 1) Model prediction u_t
    # -----------------------
    t_vec = t.view(B) if t.dim() > 1 else t  # [B]

    lambda_um_model = None
    if lambda_um is not None:
        lambda_um_model = lambda_um.view(B, 1).to(device=device, dtype=dtype)

    if cond is None:
        u_t_pred = model(x_t, t_vec, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)
    else:
        u_t_pred = model(x_t, t_vec, cond=cond, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)

    diff_fm = u_t_pred - v_t_fields  # [B,2,H,W]

    # -----------------------
    # 2) Estimate clean x1 fields from (x_t, u_t_pred)
    # -----------------------
    fields_t = x_t[:, 0:2]  # [B,2,H,W]
    eps      = x_t[:, 2:3]  # [B,1,H,W]
    t_view   = t.view(B, 1, 1, 1) if t.dim() > 1 else t.view(B, 1, 1, 1)

    a = (1.0 - float(sig_min))
    b = (1.0 - a * t_view)  # = 1 - (1-sig_min)*t
    x1_fields_pred = a * fields_t + b * u_t_pred  # [B,2,H,W] (normalized)

    x1_full = torch.cat([x1_fields_pred, eps], dim=1)  # [B,3,H,W]

    # -----------------------
    # 3) De-normalize to physical units for physics + phase comparison
    # -----------------------
    ez_real_mean = torch.as_tensor(stats["ez_real_mean"], device=device, dtype=dtype)
    ez_real_std  = torch.as_tensor(stats["ez_real_std"],  device=device, dtype=dtype)
    ez_imag_mean = torch.as_tensor(stats["ez_imag_mean"], device=device, dtype=dtype)
    ez_imag_std  = torch.as_tensor(stats["ez_imag_std"],  device=device, dtype=dtype)
    eps_mean     = torch.as_tensor(stats["eps_mean"],     device=device, dtype=dtype)
    eps_std      = torch.as_tensor(stats["eps_std"],      device=device, dtype=dtype)

    x1_phys = x1_full.clone()
    x1_phys[:, 0] = x1_full[:, 0] * ez_real_std + ez_real_mean
    x1_phys[:, 1] = x1_full[:, 1] * ez_imag_std + ez_imag_mean
    if eps_normalized:
        x1_phys[:, 2] = x1_full[:, 2] * eps_std + eps_mean
    else:
        x1_phys[:, 2] = x1_full[:, 2]

    fields1_phys = fields_1.clone()
    fields1_phys[:, 0] = fields1_phys[:, 0] * ez_real_std + ez_real_mean
    fields1_phys[:, 1] = fields1_phys[:, 1] * ez_imag_std + ez_imag_mean

    # -----------------------
    # 4) Build spatial masks/weights (PML + device)
    # -----------------------
    eps_phys_only = x1_phys[:, 2:3]  # [B,1,H,W]

    pml_m = pml_mask_like(x1_phys, pml_cells=30, margin=4)                   # [B,1,H,W]
    dev_m = device_mask_from_eps(eps_phys_only, thr=eps_thr, dilate=dilate)  # [B,1,H,W]

    df = torch.as_tensor(device_focus, device=device, dtype=dtype)
    w0 = torch.as_tensor(w_min, device=device, dtype=dtype)
    w_fm = (w0 + df * dev_m) * pml_m                                         # [B,1,H,W]

    m_focus = (pml_m * dev_m).squeeze(1)                                     # [B,H,W]

    # -----------------------
    # 4.5) Global phase alignment (per-sample) for phase/endpoint losses
    # -----------------------
    need_align = compute_endpoint or compute_phase or compute_phase_grad

    # RAW physical complex prediction (useful for sparams – no label-based alignment)
    E_pred_phys_raw = torch.complex(x1_phys[:, 0], x1_phys[:, 1])            # [B,H,W]
    E_true_phys     = torch.complex(fields1_phys[:, 0], fields1_phys[:, 1])  # [B,H,W]

    if need_align:
        abs_true = torch.abs(E_true_phys).clamp_min(1e-8)

        w_align = (pml_m.squeeze(1) * (w_min + device_focus * dev_m.squeeze(1)) * (abs_true ** 2)).to(dtype)

        dot = (w_align * torch.conj(E_pred_phys_raw) * E_true_phys).sum(dim=(1, 2))
        mag = torch.abs(dot)
        rot = torch.where(mag > 1e-8, dot / mag, torch.ones_like(dot))
        rot_hw = rot.view(B, 1, 1)

        E_pred_phys = E_pred_phys_raw * rot_hw
        x1_fields_phys_aligned = torch.stack([torch.real(E_pred_phys), torch.imag(E_pred_phys)], dim=1)

        # Apply same rotation to normalized pred for endpoint loss
        E_pred_norm = torch.complex(x1_fields_pred[:, 0], x1_fields_pred[:, 1]) * rot_hw
        x1_fields_pred_aligned = torch.stack([torch.real(E_pred_norm), torch.imag(E_pred_norm)], dim=1)
    else:
        x1_fields_phys_aligned = x1_phys[:, 0:2]
        x1_fields_pred_aligned = x1_fields_pred

    # -----------------------
    # 5) Weighted FM loss
    # -----------------------
    num = (w_fm * (diff_fm ** 2)).sum(dim=(2, 3))     # [B,2]
    den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)         # [B,1]
    fm_loss = (num / den).mean()

    # -----------------------
    # 5.5) Endpoint loss
    # -----------------------
    if compute_endpoint:
        diff_end = x1_fields_pred_aligned - fields_1  # both normalized, pred aligned
        end_num = (w_fm * (diff_end ** 2)).sum(dim=(2, 3))
        end_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)
        endpoint_loss = (end_num / end_den).mean()
    else:
        endpoint_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------
    # 6) Phase losses (physical fields)
    # -----------------------
    if compute_phase or compute_phase_grad:
        E_pred = torch.complex(x1_fields_phys_aligned[:, 0], x1_fields_phys_aligned[:, 1])
        E_true = E_true_phys
    else:
        E_pred = None
        E_true = None

    if compute_phase_grad:
        phase_grad = phase_grad_loss(E_pred, E_true, m_focus)
    else:
        phase_grad = torch.tensor(0.0, device=device, dtype=dtype)

    if compute_phase:
        abs_pred = torch.abs(E_pred).clamp_min(1e-8)
        abs_true = torch.abs(E_true).clamp_min(1e-8)
        u_pred = E_pred / abs_pred
        u_true = E_true / abs_true

        phase_err = 1.0 - torch.real(u_pred * torch.conj(u_true))  # [B,H,W]

        m_phase = amp_gate(E_true, m_focus.to(dtype), tau=0.2, eps=1e-8)  # [B,H,W]
        w_phase = (abs_true ** 2).to(dtype)
        w_phase = masked_normalize(w_phase, m_phase, eps=1e-8)

        phase_num = (phase_err * w_phase * m_phase).sum(dim=(1, 2))
        phase_den = (w_phase * m_phase).sum(dim=(1, 2)).clamp_min(1.0)
        phase_loss = (phase_num / phase_den).mean()
    else:
        phase_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------
    # 7) Physics residual (Helmholtz) on aligned physical fields
    # -----------------------
    if compute_residual:
        if lambda_um is not None:
            k0 = (2.0 * torch.pi) / lambda_um.view(B)
        else:
            k0 = None

        x_fields_phys = x1_fields_phys_aligned
        R = helmholtz_op(x_fields_phys, eps_phys_only, k0=k0)
        R2 = (R[:, 0:1] ** 2 + R[:, 1:2] ** 2)

        if weight_residual:
            res_num = (w_fm * R2).sum(dim=(2, 3))
            res_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)
            residual_loss = (res_num / res_den).mean()
        else:
            residual_loss = R2.mean()
    else:
        residual_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------
    # 8) S-parameter loss (optional)
    # -----------------------
    if compute_sparam and (aux is not None):
        pm = aux.get("port_masks", None)
        st = aux.get("sparams_true", None)

        if (pm is not None) and (st is not None):
            # IMPORTANT CHOICE:
            # Use RAW predicted field so the model is forced to get global phase right for S-params.
            # (If you instead use aligned fields, you’re letting the loss “cheat” using labels.)
            S_pred = extract_sparams(
                E_pred_phys_raw,
                pm,
                in_port_idx=aux.get("in_port_idx", None),
                port_ids=aux.get("port_ids", None),
            )
            sparam_loss_val = sparam_loss(
                S_pred,
                st,
                in_port_idx=aux.get("in_port_idx", None),
                port_ids=aux.get("port_ids", None),
            )
        else:
            sparam_loss_val = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        sparam_loss_val = torch.tensor(0.0, device=device, dtype=dtype)

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
    sig_min: float = SIG_MIN,    # <--- NEW
) -> torch.Tensor:
    """
    Flow-matching sampler for fields, conditioned on spatial maps (eps, and optionally src).
    Returns x1 (clean field) even if sig_min != 0 (via final de-bias step).
    """
    assert cond_maps is not None, "cond_maps must be provided (at least eps)"
    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]

    cond_maps = cond_maps.to(device=device, dtype=dtype)

    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2
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
            v0 = ema(x_in0, t_vec0, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)
        else:
            v0 = ema(x_in0, t_vec0, cond=cond, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)

        x_euler = x_new + dt * v0

        x_in1 = torch.cat([x_euler, cond_maps], dim=1)
        if cond is None:
            v1 = ema(x_in1, t_vec1, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)
        else:
            v1 = ema(x_in1, t_vec1, cond=cond, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)

        x_new = x_new + 0.5 * dt * (v0 + v1)

    # If sig_min > 0, x_new is x(t=1) = x1 + sig_min * x0.
    # Convert to clean x1 using x1 = (1-s)*x_t + s*u_t evaluated at t=1.
    s = float(sig_min)
    if s != 0.0:
        t_vec = torch.ones((B,), device=device, dtype=dtype)
        x_in = torch.cat([x_new, cond_maps], dim=1)
        if cond is None:
            u1 = ema(x_in, t_vec, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)
        else:
            u1 = ema(x_in, t_vec, cond=cond, lambda_um=lambda_um_model, phys_gate=phys_gate, phase_gate=phase_gate)
        a = (1.0 - s)
        x_new = a * x_new + s * u1

    return x_new
