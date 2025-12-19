from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

Spec = Tuple[int, int]  # (layer, datatype)

@dataclass(frozen=True)
class GDSLayout:
    path: str
    cell_name: str  
    unit: float
    polygons_by_spec: Dict[Spec, List[np.ndarray]]  # each poly: (N,2) float64 in GDS user units
    labels: List[Any]  # gdstk.Label objects