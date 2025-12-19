from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GridSpec:
    dx_um: float
    pad_um: float = 2.0
    snap_multiple: int = 16  # pad H,W to multiples (UNet-friendly)


@dataclass(frozen=True)
class SourceSpec:
    port: str
    mode: int = 0
    amplitude: complex = 1 + 0j


@dataclass(frozen=True)
class SimSpec:
    wavelength_um: float = 1.55
    polarization: str = "TE"
    source: SourceSpec = SourceSpec("in1")
    outputs: Tuple[str, ...] = ("fields", "sparams", "eps")
