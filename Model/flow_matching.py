import torch
import torch.nn.functional as F
from grad_utils import generalized_image_to_b_xy_c

sig_min = 0.0


def psi_t(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return (1 - (1 - sig_min) * t) * x_0 + t * x_1


def u_t(x_0: torch.Tensor, x_1: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return x_1 - (1 - sig_min) * x_0


def sample_t(x_1: torch.tensor) -> torch.tensor:
    # returns [B, 1, 1, 1]
    return torch.rand([x_1.shape[0]] + [1] * (x_1.dim() - 1), device=x_1.device)


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


def cfm_loss_residual(
    model,
    x_t,                 # [B,3,H,W] = [Re_t, Im_t, eps]
    t,                   # [B,1,1,1]
    v_t_fields,          # [B,2,H,W] target velocity for fields
    grad_helper,
    use_dignorm: bool,
    residual_dt: float,
    stats: dict,
    eps_normalized: bool,
    fields_1,
    cond=None,           # [B,1] normalized λ (for the UNet)
    lambda_um=None,      # [B,1] physical λ (for PDE residual)
):
    """
    Conditional FM loss + physics residual, WITHOUT calling the sampler.

    - model input:  x_t = [fields_t (2ch), eps (1ch)]
    - model output: u_t_pred_fields (2ch)
    - v_t_fields:   target velocity for fields only
    - eps is kept fixed and only used as conditioning.
    - cond:         e.g. normalized wavelength passed into PhysicsUNet
    - lambda_um:    physical wavelength (um) for PDE residual; per-sample
    """
    B, C, H, W = x_t.shape
    device = x_t.device
    dtype = x_t.dtype

    # -----------------------
    # 1) FM loss on fields
    # -----------------------
    # model expects t as [B]
    t_vec = t.view(B)

    if cond is None:
        u_t_pred = model(x_t, t_vec)              # [B,2,H,W]
    else:
        # PhysicsUNet: forward(x, t, cond)
        u_t_pred = model(x_t, t_vec, cond=cond)   # [B,2,H,W]

    fm_loss = F.mse_loss(u_t_pred, v_t_fields)

    # -----------------------
    # 2) Estimate clean fields x1 from (x_t, u_t_pred)
    # -----------------------
    # split fields vs eps
    fields_t = x_t[:, 0:2]      # [B,2,H,W]
    eps      = x_t[:, 2:3]      # [B,1,H,W]

    # straight-path flow matching (σ_min = 0):
    #   x_1 = x_t + (1 - t) * u_t(x_t, t)
    t_view = t.view(B, 1, 1, 1)                     # [B,1,1,1]
    x1_fields_pred = fields_t + (1.0 - t_view) * u_t_pred

    # full state [Re, Im, eps] for residual computation
    x1_full = torch.cat([x1_fields_pred, eps], dim=1)  # [B,3,H,W]

    # -----------------------
    # 3) De-normalize to physical units
    # -----------------------
    ez_real_mean = torch.tensor(stats["ez_real_mean"], device=device, dtype=dtype)
    ez_real_std  = torch.tensor(stats["ez_real_std"],  device=device, dtype=dtype)
    ez_imag_mean = torch.tensor(stats["ez_imag_mean"], device=device, dtype=dtype)
    ez_imag_std  = torch.tensor(stats["ez_imag_std"],  device=device, dtype=dtype)
    eps_mean     = torch.tensor(stats["eps_mean"],     device=device, dtype=dtype)
    eps_std      = torch.tensor(stats["eps_std"],      device=device, dtype=dtype)

    x1_phys = x1_full.clone()
    x1_phys[:, 0] = x1_full[:, 0] * ez_real_std + ez_real_mean
    x1_phys[:, 1] = x1_full[:, 1] * ez_imag_std + ez_imag_mean
    if eps_normalized:
        x1_phys[:, 2] = x1_full[:, 2] * eps_std + eps_mean
    else:
        x1_phys[:, 2] = x1_full[:, 2]

    # -----------------------
    # 4) Physics residual (Helmholtz), with per-sample λ if available
    # -----------------------
    # GradientsHelper.compute_residual is assumed to accept optional wavelength_um
    if lambda_um is not None:
        # lambda_um: [B,1] → [B,1,1,1] for broadcasting
        lambda_um_broadcast = lambda_um.view(B, 1, 1, 1).to(device=device, dtype=dtype)
        residual_sq = grad_helper.compute_residual(
            x1_phys,
            wavelength_um=lambda_um_broadcast,
        )["residual_sq"]   # [B, H*W, 1]
    else:
        # fallback to the helper's default wavelength (set at construction)
        residual_sq = grad_helper.compute_residual(x1_phys)["residual_sq"]

    # mean squared residual over the spatial domain (and batch)
    residual_loss = residual_sq.mean()

    # `use_dignorm` and `residual_dt` are kept in the signature for future tricks
    # (e.g. dignorm or time-weighted residuals), but are not used here.

    return fm_loss, residual_loss


def sample(
    ema,
    x_0: torch.Tensor,          # noise over *fields only*: [B, 2, H, W]
    num_steps: int,
    use_stoc_samp: bool,
    cond_eps: torch.Tensor,     # fixed eps: [B, 1, H, W]
    cond=None,                  # optional conditioning [B, 1]
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

    time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    x_new = torch.clone(x_0)

    for k in range(num_steps):
        t0 = time_steps[k]
        t1 = time_steps[k + 1]
        # make a [B]-shaped time tensor for the timestep embedder
        t_vec = t0.expand(B)

        # concatenate current fields with fixed eps → 3-channel input
        x_in = torch.cat([x_new, cond_eps], dim=1)   # [B, 3, H, W]

        if cond is None:
            net = lambda x: ema(x, t_vec)
        else:
            net = lambda x: ema(x, t_vec, cond=cond)

        if (t0 < 0.2) and use_stoc_samp:
            # predictor step
            v_t = net(x_in)         # [B, 2, H, W]
            x_new = x_new + (1.0 - t0) * v_t

            # stochastic refresh of the fields only (eps stays untouched)
            noise = torch.randn_like(x_new)
            x_new = (1.0 - t1) * noise + t1 * x_new
        else:
            v_t = net(x_in)         # [B, 2, H, W]
            x_new = x_new + (t1 - t0) * v_t

    return x_new
