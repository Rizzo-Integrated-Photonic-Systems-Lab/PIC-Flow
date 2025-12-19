from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Spec = Tuple[int, int]


@dataclass(frozen=True)
class ProcessKit2D:
    """
    Maps GDS layer/datatype -> effective permittivity eps_eff.

    v1 assumption: one fixed vertical stack -> 2D effective-index model.
    """
    name: str
    background_eps: float
    layer_to_eps: Dict[Spec, float]  # (layer,datatype) -> eps_eff

    # Label parsing defaults for ports
    port_label_layer: int = 200
    port_label_texttype: int = 0
