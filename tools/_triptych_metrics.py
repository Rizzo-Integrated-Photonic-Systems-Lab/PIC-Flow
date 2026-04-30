"""Per-sample physics metrics for triptych plots.

Computes the Helmholtz residual ratio (predicted / FDTD), matching the
masking convention used during training: PML margin excluded, source
region dilated then excluded, dielectric-interface pixels excluded.

R = ∇²E + k0² ε E,  with E = Re(Ez) + i·Im(Ez), k0 = 2π/λ.
Mask defaults match Model/flow_matching.py + the *.slurm INTERFACE_THR=1.0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _laplacian_5pt(field: np.ndarray, dx_um: float, dy_um: float) -> np.ndarray:
    """5-point finite-difference Laplacian. Edge pixels left as 0."""
    inv_dx2 = 1.0 / (dx_um * dx_um)
    inv_dy2 = 1.0 / (dy_um * dy_um)
    lap = np.zeros_like(field, dtype=np.float64)
    f = field.astype(np.float64, copy=False)
    lap[1:-1, 1:-1] = (
        inv_dx2 * (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2])
        + inv_dy2 * (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1])
    )
    return lap


def _helmholtz_residual_squared(
    ezr: np.ndarray, ezi: np.ndarray, eps: np.ndarray,
    *, dx_um: float, dy_um: float, wavelength_um: float,
) -> np.ndarray:
    """|R|² with R = ∇²E + k0² ε E (E complex via Re/Im)."""
    k0_sq = (2.0 * math.pi / float(wavelength_um)) ** 2
    lap_r = _laplacian_5pt(ezr, dx_um, dy_um)
    lap_i = _laplacian_5pt(ezi, dx_um, dy_um)
    R_r = lap_r + k0_sq * eps.astype(np.float64) * ezr.astype(np.float64)
    R_i = lap_i + k0_sq * eps.astype(np.float64) * ezi.astype(np.float64)
    return R_r * R_r + R_i * R_i


def _box_dilate_bool(mask: np.ndarray, kp: int) -> np.ndarray:
    """Square-kernel dilation of a boolean mask by ±kp pixels."""
    if kp <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    ny, nx = mask.shape
    for j in range(-kp, kp + 1):
        j0_dst = max(0, j); j1_dst = ny - max(0, -j)
        j0_src = max(0, -j); j1_src = ny - max(0, j)
        for i in range(-kp, kp + 1):
            i0_dst = max(0, i); i1_dst = nx - max(0, -i)
            i0_src = max(0, -i); i1_src = nx - max(0, i)
            out[j0_dst:j1_dst, i0_dst:i1_dst] |= mask[j0_src:j1_src, i0_src:i1_src]
    return out


def _residual_valid_mask(
    eps: np.ndarray, src_mask: np.ndarray,
    *, dx_um: float, dy_um: float,
    interface_thr: float = 1.0,
    src_dilate_px: int = 2,
    margin_px: int = 2,
) -> np.ndarray:
    """Pixels where Helmholtz residual is meaningful: not PML margin, not
    inside dilated source region, and not on a dielectric interface."""
    valid = np.ones_like(eps, dtype=bool)
    if margin_px > 0:
        valid[:margin_px, :] = False
        valid[-margin_px:, :] = False
        valid[:, :margin_px] = False
        valid[:, -margin_px:] = False

    src_bin = src_mask > 0.5
    src_dilated = _box_dilate_bool(src_bin, int(src_dilate_px))
    valid &= ~src_dilated

    grad_x = np.zeros_like(eps, dtype=np.float64)
    grad_y = np.zeros_like(eps, dtype=np.float64)
    e = eps.astype(np.float64, copy=False)
    grad_x[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2.0 * dx_um)
    grad_y[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2.0 * dy_um)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    valid &= grad_mag < float(interface_thr)
    return valid


def residual_metrics(
    eps: np.ndarray, src_mask: np.ndarray,
    fdtd_ezr: np.ndarray, fdtd_ezi: np.ndarray,
    pred_ezr: np.ndarray, pred_ezi: np.ndarray,
    *, dx_um: float, dy_um: float, wavelength_um: float,
    interface_thr: float = 1.0,
    src_dilate_px: int = 2,
    margin_px: int = 2,
) -> dict[str, float]:
    """Helmholtz-residual metrics for a single sample.

    Reported numbers
    ----------------
    L_res_pred / L_res_fdtd : dimensionless residuals (paper Eq. 8 form),
        L_res = mean(|R|² · w) / mean(|k₀²εE_FDTD|² · w).
        L_res_pred ≈ paper's "Val. residual" column.
    eps_R : sqrt(L_res_pred) in **percent** — interpretable as "model
        residual relative to local driving term magnitude". Used for the
        plot annotation.
    ratio : raw mean(|R_pred|²)/mean(|R_FDTD|²); same number as before but
        kept here only for diagnostic purposes — well-converged FDTD has
        numerical-noise residual, so this multiplier is misleading and
        not displayed by default.
    """
    valid = _residual_valid_mask(
        eps, src_mask, dx_um=dx_um, dy_um=dy_um,
        interface_thr=interface_thr,
        src_dilate_px=src_dilate_px,
        margin_px=margin_px,
    )
    R2_fdtd = _helmholtz_residual_squared(
        fdtd_ezr, fdtd_ezi, eps,
        dx_um=dx_um, dy_um=dy_um, wavelength_um=wavelength_um,
    )
    R2_pred = _helmholtz_residual_squared(
        pred_ezr, pred_ezi, eps,
        dx_um=dx_um, dy_um=dy_um, wavelength_um=wavelength_um,
    )
    n_valid = int(valid.sum())
    if n_valid == 0:
        nan = float("nan")
        return dict(
            mean_R2_fdtd=nan, mean_R2_pred=nan,
            L_res_fdtd=nan, L_res_pred=nan,
            eps_R_pct=nan, ratio=nan, n_valid=0,
        )
    mean_R2_fdtd = float(R2_fdtd[valid].mean())
    mean_R2_pred = float(R2_pred[valid].mean())

    # Driving term: |k₀² ε E_FDTD|² (paper Eq. 8 uses E_true).
    k0_sq = (2.0 * math.pi / float(wavelength_um)) ** 2
    f_mag2 = (fdtd_ezr.astype(np.float64) ** 2
              + fdtd_ezi.astype(np.float64) ** 2)
    drive_sq = (k0_sq * eps.astype(np.float64)) ** 2 * f_mag2
    mean_drive = float(drive_sq[valid].mean())
    mean_drive = max(mean_drive, 1e-30)

    L_res_fdtd = mean_R2_fdtd / mean_drive
    L_res_pred = mean_R2_pred / mean_drive
    eps_R_pct = float(math.sqrt(max(L_res_pred, 0.0)) * 100.0)

    return dict(
        mean_R2_fdtd=mean_R2_fdtd,
        mean_R2_pred=mean_R2_pred,
        L_res_fdtd=L_res_fdtd,
        L_res_pred=L_res_pred,
        eps_R_pct=eps_R_pct,
        ratio=mean_R2_pred / max(mean_R2_fdtd, 1e-30),
        n_valid=n_valid,
    )


def format_metric_label(m: dict[str, Any]) -> str:
    """One-line label for plotting. Returns the model's compliance error
    as a percentage of the driving term, e.g. '2.7%' (in-dataset) or
    '12.1%' (heavy OOD). Smaller = more physics-compliant prediction."""
    pct = m.get("eps_R_pct", float("nan"))
    if not np.isfinite(pct):
        return "n/a"
    if pct < 10:
        return rf"$\varepsilon_R$ = {pct:.1f}%"
    return rf"$\varepsilon_R$ = {pct:.0f}%"
