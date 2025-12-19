import torch
import torch.nn.functional as F
from grad_utils import generalized_image_to_b_xy_c

sig_min = 0.0


def psi_t(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return (1 - (1 - sig_min) * t) * x_0 + t * x_1


def u_t(x_0: torch.Tensor, x_1: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return x_1 - (1 - sig_min) * x_0


def sample_t(x_1):
    B = x_1.shape[0]
    dev = x_1.device
    # 50% uniform, 50% Beta(2,2)
    u = torch.rand((B,), device=dev)
    dist = torch.distributions.Beta(torch.tensor(2., device=dev), torch.tensor(2., device=dev))
    b = dist.sample((B,))
    t = torch.where(torch.rand((B,), device=dev) < 0.5, u, b)
    return t.view(B, *([1]*(x_1.dim()-1)))


def logit_normal_map(t: torch.Tensor, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    eps = 1e-6
    t = torch.clamp(t, eps, 1 - eps)
    return t * torch.exp(-0.5 * torch.square((torch.log(t / (1 - t)) - m) / s)) / (
        s * torch.sqrt(2 * torch.tensor(torch.pi)) * t * (1 - t) ** 2
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
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate//2)
    return m

def align_global_phase(E_pred: torch.Tensor, E_true: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    """
    Find per-sample complex rotation r (|r|=1) that best aligns E_pred to E_true under weights w,
    then return E_pred_aligned = r * E_pred.

    E_pred, E_true: complex tensors [B,H,W]
    w: real weights [B,H,W] (mask * amplitude weights etc.)
    """
    # dot = sum w * conj(E_pred) * E_true
    dot = (w * torch.conj(E_pred) * E_true).sum(dim=(1, 2))  # [B]
    rot = dot / torch.abs(dot).clamp_min(eps)                # [B], unit magnitude
    return E_pred * rot.view(-1, 1, 1)

def phase_grad_loss(E_pred: torch.Tensor, E_true: torch.Tensor, m_focus: torch.Tensor, eps: float = 1e-8):
    """
    E_pred, E_true: complex [B,H,W] (PHYSICAL units preferred), already globally phase-aligned.
    m_focus: real mask [B,H,W] (device * non-PML)
    Returns: scalar loss enforcing local phase advance match (reduces phase drift).
    """
    # amplitude weights (focus where field matters)
    w = m_focus * (torch.abs(E_true) ** 2)
    w = w / (w.mean(dim=(1,2), keepdim=True).clamp_min(eps))

    # forward diffs (grid units)
    dE_dx_pred = E_pred[:, :, 1:] - E_pred[:, :, :-1]   # [B,H,W-1]
    dE_dx_true = E_true[:, :, 1:] - E_true[:, :, :-1]
    dE_dy_pred = E_pred[:, 1:, :] - E_pred[:, :-1, :]   # [B,H-1,W]
    dE_dy_true = E_true[:, 1:, :] - E_true[:, :-1, :]

    # use left/top point for |E|^2 normalization (consistent with diff)
    A2_pred_x = (torch.abs(E_pred[:, :, :-1]) ** 2).clamp_min(eps)
    A2_true_x = (torch.abs(E_true[:, :, :-1]) ** 2).clamp_min(eps)
    A2_pred_y = (torch.abs(E_pred[:, :-1, :]) ** 2).clamp_min(eps)
    A2_true_y = (torch.abs(E_true[:, :-1, :]) ** 2).clamp_min(eps)

    # kx, ky ~ dphi (grid units)
    kx_pred = torch.imag(torch.conj(E_pred[:, :, :-1]) * dE_dx_pred) / A2_pred_x
    kx_true = torch.imag(torch.conj(E_true[:, :, :-1]) * dE_dx_true) / A2_true_x
    ky_pred = torch.imag(torch.conj(E_pred[:, :-1, :]) * dE_dy_pred) / A2_pred_y
    ky_true = torch.imag(torch.conj(E_true[:, :-1, :]) * dE_dy_true) / A2_true_y

    # masks/weights sliced to match shapes
    wx = w[:, :, :-1]
    mx = m_focus[:, :, :-1]
    wy = w[:, :-1, :]
    my = m_focus[:, :-1, :]

    # weighted MSE
    loss_x = (wx * mx * (kx_pred - kx_true) ** 2).sum(dim=(1,2)) / mx.sum(dim=(1,2)).clamp_min(1.0)
    loss_y = (wy * my * (ky_pred - ky_true) ** 2).sum(dim=(1,2)) / my.sum(dim=(1,2)).clamp_min(1.0)
    return (loss_x + loss_y).mean()


def cfm_loss_residual(
    model,
    x_t,                 # [B,3,H,W] = [Re_t, Im_t, eps] (normalized)
    t,                   # [B,1,1,1] or [B] time
    v_t_fields,          # [B,2,H,W] target FM velocity in field-space (normalized)
    helmholtz_op,        # HelmholtzResidual2D instance (expects physical units)
    stats: dict,         # normalization stats
    eps_normalized: bool,
    fields_1,            # [B,2,H,W] ground truth final fields (normalized)
    cond=None,
    lambda_um=None,      # [B,1] physical wavelength (um)
    device_focus: float = 0.0,   # 1.0 = strongly focus device, 0.0 = no extra focus
    w_min: float = 0.1,          # baseline weight everywhere
    eps_thr: float = 6.0,        # threshold in *physical* eps units
    dilate: int = 9,             # dilation kernel size (odd int recommended)
    weight_residual: bool = True # if True, weight residual with same spatial weights
):
    """
    Conditional FM loss + physics residual + phase loss + endpoint loss.

    Key features:
      - Spatial weighting: device-focused mask from eps + PML mask.
      - Phase loss: local phase via unit phasor cosine distance, masked to device.
      - Physics residual: Helmholtz residual on predicted physical fields.
      - Global phase alignment (per-sample): aligns predicted field to GT under a robust
        weight w_align (device mask * |E_true|^2). This makes endpoint + phase losses
        gauge-invariant to global phase.

    Expected:
      x_t:      [B,3,H,W] normalized [Ez_re, Ez_im, eps]
      fields_1: [B,2,H,W] normalized GT fields
      lambda_um:[B,1] physical wavelength in um (optional)
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
        u_t_pred = model(x_t, t_vec, lambda_um=lambda_um_model)             # [B,2,H,W]
    else:
        u_t_pred = model(x_t, t_vec, cond=cond, lambda_um=lambda_um_model)  # [B,2,H,W]

    diff_fm = u_t_pred - v_t_fields  # [B,2,H,W]

    # -----------------------
    # 2) Estimate clean x1 fields from (x_t, u_t_pred)
    # -----------------------
    fields_t = x_t[:, 0:2]  # [B,2,H,W]
    eps      = x_t[:, 2:3]  # [B,1,H,W]
    t_view   = t.view(B, 1, 1, 1) if t.dim() > 1 else t.view(B, 1, 1, 1)
    x1_fields_pred = fields_t + (1.0 - t_view) * u_t_pred  # [B,2,H,W]
    x1_full        = torch.cat([x1_fields_pred, eps], dim=1)  # [B,3,H,W]

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

    pml_m = pml_mask_like(x1_phys, pml_cells=30, margin=4)  # [B,1,H,W]
    dev_m = device_mask_from_eps(eps_phys_only, thr=eps_thr, dilate=dilate)  # [B,1,H,W]

    df = torch.as_tensor(device_focus, device=device, dtype=dtype)
    w0 = torch.as_tensor(w_min,         device=device, dtype=dtype)
    w_fm = (w0 + df * dev_m) * pml_m  # [B,1,H,W]

    # mask for phase/alignment focus (no baseline)
    m_focus = (pml_m * dev_m).squeeze(1)  # [B,H,W]

    # -----------------------
    # 4.5) Global phase alignment (per-sample)
    # -----------------------
    # Build physical complex fields
    E_pred_phys_raw = torch.complex(x1_phys[:, 0], x1_phys[:, 1])            # [B,H,W]
    E_true_phys     = torch.complex(fields1_phys[:, 0], fields1_phys[:, 1])  # [B,H,W]
    abs_true = torch.abs(E_true_phys).clamp_min(1e-8)

    # alignment weights: focus region * |E_true|^2
    w_align = (m_focus * (abs_true ** 2)).to(dtype)  # [B,H,W]

    # dot = sum w * conj(E_pred) * E_true
    dot = (w_align * torch.conj(E_pred_phys_raw) * E_true_phys).sum(dim=(1, 2))  # [B] complex
    mag = torch.abs(dot)

    # if dot ~ 0, use identity rotation (avoid NaNs / rot=0)
    rot = torch.where(mag > 1e-8, dot / mag, torch.ones_like(dot))  # [B] complex, |rot|=1
    rot_hw = rot.view(B, 1, 1)

    
    E_pred_phys = E_pred_phys_raw * rot_hw
    x1_fields_phys_aligned = torch.stack([torch.real(E_pred_phys), torch.imag(E_pred_phys)], dim=1)  # [B,2,H,W]

    E_pred_norm = torch.complex(x1_fields_pred[:, 0], x1_fields_pred[:, 1]) * rot_hw
    x1_fields_pred_aligned = torch.stack([torch.real(E_pred_norm), torch.imag(E_pred_norm)], dim=1)  # [B,2,H,W]


    # -----------------------
    # 5) Weighted FM loss
    # -----------------------
    num = (w_fm * (diff_fm ** 2)).sum(dim=(2, 3))  # [B,2]
    den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)      # [B,1]
    fm_loss = (num / den).mean()

    # -----------------------
    # 5.5) Endpoint loss (now gauge-invariant due to alignment)
    # -----------------------
    diff_end = x1_fields_pred_aligned - fields_1  # both normalized, pred aligned
    end_num = (w_fm * (diff_end ** 2)).sum(dim=(2, 3))  # [B,2]
    end_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)       # [B,1]
    endpoint_loss = (end_num / end_den).mean()

    # -----------------------
    # 6) Phase (phasor) loss (local phase), amplitude-weighted, device-focused
    # -----------------------
    E_pred = torch.complex(x1_fields_phys_aligned[:, 0], x1_fields_phys_aligned[:, 1])
    E_true = torch.complex(fields1_phys[:, 0], fields1_phys[:, 1])

    phase_grad = phase_grad_loss(E_pred, E_true, m_focus)

    abs_pred = torch.abs(E_pred).clamp_min(1e-8)
    abs_true = torch.abs(E_true).clamp_min(1e-8)

    u_pred = E_pred / abs_pred
    u_true = E_true / abs_true

    cos_sim = torch.real(u_pred * torch.conj(u_true))  # [B,H,W]
    phase_err = 1.0 - cos_sim

    w_phase = (abs_true ** 2)
    w_phase = w_phase / (w_phase.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8))

    phase_num = (phase_err * w_phase * m_focus).sum(dim=(1, 2))
    phase_den = m_focus.sum(dim=(1, 2)).clamp_min(1.0)
    phase_loss = (phase_num / phase_den).mean()

    

    # -----------------------
    # 7) Physics residual (Helmholtz)
    # -----------------------
    if lambda_um is not None:
        k0 = (2.0 * torch.pi) / lambda_um.view(B)  # [B]
    else:
        k0 = None

    x_fields_phys = x1_fields_phys_aligned  # [B,2,H,W] aligned
    R = helmholtz_op(x_fields_phys, eps_phys_only, k0=k0)          # [B,2,H,W] (PML masked inside)
    R2 = (R[:, 0:1] ** 2 + R[:, 1:2] ** 2)                         # [B,1,H,W]

    if weight_residual:
        res_num = (w_fm * R2).sum(dim=(2, 3))           # [B,1]
        res_den = w_fm.sum(dim=(2, 3)).clamp_min(1.0)  # [B,1]
        residual_loss = (res_num / res_den).mean()
    else:
        residual_loss = R2.mean()

    return fm_loss, residual_loss, phase_loss, endpoint_loss, phase_grad


def sample(
    ema,
    x_0: torch.Tensor,          # noise over *fields only*: [B, 2, H, W]
    num_steps: int,
    use_stoc_samp: bool,
    cond_eps: torch.Tensor,     # fixed eps: [B, 1, H, W]
    cond=None,                  # optional conditioning [B, 1]
    lambda_um=None,             # optional physical wavelength [B, 1]
) -> torch.Tensor:
    """
    Flow-matching sampler for fields, conditioned on eps (and optionally λ).

    ema       : EMA model (same API as DiT/PhysicsUNet)
    x_0       : initial Gaussian noise for [Re, Im] fields, shape [B, 2, H, W]
    cond_eps  : fixed geometry eps, shape [B, 1, H, W]
    cond      : conditioning vector, e.g. wavelength, shape [B, 1] or [B, d]
    returns   : final fields at t=1, shape [B, 2, H, W]
    """
    assert cond_eps is not None, "cond_eps (eps geometry) must be provided"

    device = x_0.device
    dtype = x_0.dtype
    B = x_0.shape[0]

    # concentrates steps near 0
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    time_steps = base ** 2   # concentrates steps near 0
    x_new = x_0.clone()

    for k in range(num_steps):
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        dt = (t1 - t0)
        t_vec0 = t0.expand(B)
        t_vec1 = t1.expand(B)

        x_in0 = torch.cat([x_new, cond_eps], dim=1)

        # v0 at (x_new, t0)
        if cond is None:
            v0 = ema(x_in0, t_vec0, lambda_um=lambda_um)
        else:
            v0 = ema(x_in0, t_vec0, cond=cond, lambda_um=lambda_um)

        # predictor (Euler)
        x_euler = x_new + dt * v0

        # v1 at (x_euler, t1)
        x_in1 = torch.cat([x_euler, cond_eps], dim=1)
        if cond is None:
            v1 = ema(x_in1, t_vec1, lambda_um=lambda_um)
        else:
            v1 = ema(x_in1, t_vec1, cond=cond, lambda_um=lambda_um)

        # corrector (Heun / RK2)
        x_new = x_new + 0.5 * dt * (v0 + v1)

    return x_new
