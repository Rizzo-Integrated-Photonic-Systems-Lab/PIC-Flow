# conditioning_masks.py
import numpy as np

def xy_to_ij(x_um: float, y_um: float, Lx_um: float, Ly_um: float, ny: int, nx: int):
    dx = Lx_um / nx
    dy = Ly_um / ny

    j = int(np.round((x_um + 0.5 * Lx_um) / dx - 0.5))
    i = int(np.round((y_um + 0.5 * Ly_um) / dy - 0.5))

    j = max(0, min(nx - 1, j))
    i = max(0, min(ny - 1, i))
    return i, j, dy


def make_port_stripe_mask(
    x_port_um: float,
    y_port_um: float,
    y_span_um: float,
    Lx_um: float,
    Ly_um: float,
    ny: int,
    nx: int,
    stripe_thickness_px: int = 3,
    soft_sigma_px: float = 1.0,
):
    mask = np.zeros((ny, nx), dtype=np.float32)

    i0, j0, dy = xy_to_ij(x_port_um, y_port_um, Lx_um, Ly_um, ny, nx)

    half_span_px = max(1, int(np.round(0.5 * y_span_um / dy)))
    i_lo = max(0, i0 - half_span_px)
    i_hi = min(ny, i0 + half_span_px + 1)

    half_thick = max(0, stripe_thickness_px // 2)
    j_lo = max(0, j0 - half_thick)
    j_hi = min(nx, j0 + half_thick + 1)

    mask[i_lo:i_hi, j_lo:j_hi] = 1.0

    if soft_sigma_px is not None and soft_sigma_px > 0:
        r = int(np.ceil(3 * soft_sigma_px))
        xs = np.arange(-r, r + 1)
        g = np.exp(-(xs**2) / (2 * soft_sigma_px**2)).astype(np.float32)
        g /= g.sum()

        tmp = np.pad(mask, ((r, r), (0, 0)), mode="edge")
        tmp = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, tmp)

        tmp2 = np.pad(tmp, ((0, 0), (r, r)), mode="edge")
        tmp2 = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, tmp2)

        mask = tmp2
        mask /= (mask.max() + 1e-8)

    return mask


def make_source_mask(
    input_port: int,
    port_centers_um: dict,
    y_span_um: float,
    Lx_um: float,
    Ly_um: float,
    ny: int,
    nx: int,
    stripe_thickness_px: int = 3,
    soft_sigma_px: float = 1.0,
):
    """
    Generic source mask builder for any device that provides port centers.
    input_port=0 -> all zeros (no excitation)
    """
    if input_port == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    if input_port not in port_centers_um:
        raise ValueError(f"Unknown port {input_port}. Valid: {sorted(port_centers_um.keys())} or 0.")

    x_um, y_um = port_centers_um[input_port]
    return make_port_stripe_mask(
        x_port_um=x_um,
        y_port_um=y_um,
        y_span_um=y_span_um,
        Lx_um=Lx_um,
        Ly_um=Ly_um,
        ny=ny,
        nx=nx,
        stripe_thickness_px=stripe_thickness_px,
        soft_sigma_px=soft_sigma_px,
    )
