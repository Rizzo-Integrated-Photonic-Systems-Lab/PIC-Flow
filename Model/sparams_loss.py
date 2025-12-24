import torch

def port_project(E: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    E: complex [B,H,W]
    mask: real  [B,H,W] or [H,W] selecting a port window (ideally thin cross-section region)
    returns: complex [B]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).expand(E.shape[0], -1, -1)
    w = mask.to(E.real.dtype)

    # Normalize the projection so amplitude isn't scale-dependent on mask area
    den = w.sum(dim=(1,2)).clamp_min(1.0)
    a = (w * E).sum(dim=(1,2)) / den
    return a

def _resolve_in_idx(
    in_port_idx,
    port_ids=None,
) -> int | torch.Tensor:
    """
    Resolve a 0-based input-port index.

    in_port_idx: optional int-like (0-based). If provided, used directly.
    port_ids: optional tensor/array [P] of port numbers (e.g. [1,2,3,4]).
             If in_port_idx is None, we fall back to the index of port_id==1 if present.
    """
    if in_port_idx is not None:
        # Allow per-sample indices (e.g., tensor [B]) for mixed batches.
        if torch.is_tensor(in_port_idx):
            idx = in_port_idx.detach().flatten()
            if idx.numel() == 1:
                return int(idx.item())
            return idx.to(dtype=torch.long)
        # list/tuple/ndarray
        try:
            t = torch.as_tensor(in_port_idx).flatten()
            if t.numel() == 1:
                return int(t.item())
            return t.to(dtype=torch.long)
        except Exception:
            return int(in_port_idx)
    if port_ids is not None:
        try:
            pid = port_ids
            if torch.is_tensor(pid):
                pid_t = pid.detach()
            else:
                pid_t = torch.as_tensor(pid)

            # pid_t can be [P] or [B,P]
            if pid_t.dim() == 1:
                hits = (pid_t == 1).nonzero(as_tuple=False)
            if hits.numel() > 0:
                return int(hits[0].item())
            elif pid_t.dim() == 2:
                # per-sample: choose first match per row, fallback to 0
                B, P = pid_t.shape
                out = torch.zeros((B,), dtype=torch.long, device=pid_t.device)
                hits = (pid_t == 1).nonzero(as_tuple=False)  # [K,2] with (b,p)
                if hits.numel() > 0:
                    # first hit per batch item
                    seen = set()
                    for b, p in hits.detach().cpu().tolist():
                        if b in seen:
                            continue
                        out[b] = int(p)
                        seen.add(b)
                return out
        except Exception:
            pass
    return 0


def extract_sparams(
    E: torch.Tensor,
    port_masks: torch.Tensor,
    in_port_idx=None,
    port_ids=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    E: complex [B,H,W]
    port_masks: real [B,P,H,W] or [P,H,W]
    in_port_idx: 0-based index of excited input port (preferred).
    port_ids: optional [P] port-number labels to help resolve input when in_port_idx is None.
    returns: complex [B,P] with S_p = a_p / a_in
    """
    if port_masks.dim() == 3:
        port_masks = port_masks.unsqueeze(0).expand(E.shape[0], -1, -1, -1)  # [B,P,H,W]

    P = port_masks.shape[1]
    amps = []
    for p in range(P):
        amps.append(port_project(E, port_masks[:, p], eps=eps))
    a = torch.stack(amps, dim=1)  # [B,P]

    in_idx = _resolve_in_idx(in_port_idx, port_ids=port_ids)
    # complex clamp doesn't exist; handle via magnitude:
    if torch.is_tensor(in_idx):
        idx = in_idx.to(device=a.device).view(-1, 1)  # [B,1]
        a_in = a.gather(1, idx).squeeze(1)            # [B]
    else:
        a_in = a[:, int(in_idx)]
    a_in = a_in / torch.abs(a_in).clamp_min(eps) * torch.abs(a_in).clamp_min(eps)

    S = a / a_in.unsqueeze(1)
    return S

def sparam_loss(S_pred: torch.Tensor,
                S_true: torch.Tensor,
                mag_w: float = 1.0,
                phase_w: float = 1.0,
                tau: float = 1e-3,
                eps: float = 1e-8,
                in_port_idx=None,
                port_ids=None) -> torch.Tensor:
    """
    S_pred, S_true: complex [B,P] (or [B,P,Q] if you do full matrix)
    Returns scalar.
    """
    mag_p = torch.abs(S_pred).clamp_min(eps)
    mag_t = torch.abs(S_true).clamp_min(eps)

    # magnitude loss (you can switch to log-mag if dynamic range is huge)
    L_mag = (mag_p - mag_t) ** 2

    # phase loss, wrap-safe
    u_p = S_pred / mag_p
    u_t = S_true / mag_t
    L_phase = 1.0 - torch.real(u_p * torch.conj(u_t))

    gate = (mag_t > tau).to(L_mag.dtype)  # don't chase phase when transmission is ~0
    L = (mag_w * L_mag + phase_w * L_phase) * gate
    return L.mean()

