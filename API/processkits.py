from __future__ import annotations

from pathlib import Path

from FDTD.utils import neff_siwire_from_tables

from .process import ProcessKit2D  # type: ignore[import]


DEFAULT_SOI220_TE_WIDTH_UM = 0.45
DEFAULT_SOI220_TE_LAMBDA_UM = 1.55
DEFAULT_BACKGROUND_INDEX = 1.444
DEFAULT_CORE_LAYER = (1, 0)


def SOI220_TE_1550(
    wg_width_um: float = DEFAULT_SOI220_TE_WIDTH_UM,
    wavelength_um: float = DEFAULT_SOI220_TE_LAMBDA_UM,
    *,
    background_eps: float | None = None,
    tables_dir: str | Path = "neff_tables",
    core_spec: tuple[int, int] = DEFAULT_CORE_LAYER,
) -> ProcessKit2D:
    """
    Process kit that maps the SOI 220 nm core layer to the effective ε derived
    from the precomputed `FDTD/neff_tables` CSVs and leaves the surrounding
    SiO₂ cladding at its native permittivity.

    The effective index is looked up from the `neff_siwire_w###_t0220.csv`
    tables (width range 0.38–0.60 µm, λ range 1.40–1.60 µm) so you can build
    layout-specific kits just by changing `wg_width_um`.
    """

    if background_eps is None:
        background_eps = DEFAULT_BACKGROUND_INDEX**2

    neff = neff_siwire_from_tables(
        wg_width_um,
        wavelength_um,
        tables_dir=tables_dir,
    )
    core_eps = float(neff**2)

    kit_name = f"SOI220_TE_{wavelength_um:.3f}_{wg_width_um:.2f}"

    return ProcessKit2D(
        name=kit_name,
        background_eps=background_eps,
        layer_to_eps={core_spec: core_eps},
        port_label_layer=200,
        port_label_texttype=0,
    )
