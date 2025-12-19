from __future__ import annotations

from typing import Optional, Set, Tuple
import gdstk

from .types import GDSLayout, Spec


def load_gds(path: str, *, cell: str | None = None, layer_filter: Set[Spec] | None = None, flatten: bool = True) -> GDSLayout:
    """
    Read a GDS, select a cell, optionally filter layers, return polygons + labels.

    v1 defualts to flattenining references to simplify rasterization.
    """
    lib = gdstk.read_gds(path)
    top = lib.top_level()
    if not top:
        raise ValueError(f"No top-level cells found in GDS: {path}")
    
    if cell is None:
        cell_obj = top[0]
    else:
        cell_obj = None
        for c in lib.cells:
            if c.name == cell:
                cell_obj = c
                break
        if cell_obj is None:
            raise ValueError(f"Cell '{cell}' not found in GDS: {path}")
    
    # copy so dont mutate the lib
    cell_obj = cell_obj.copy(cell_obj.name + "__rayfield_tmp")

    if flatten:
        cell_obj.flatten()
    
    polygons_by_spec = cell_obj.get_polygons(by_spec=True)
    if layer_filter is not None:
        polygons_by_spec = {k: v for (k, v) in polygons_by_spec.items() if k in layer_filter}

    labels = list(cell_obj.labels)

    return GDSLayout(
        path=path,
        cell_name=cell_obj.name,
        unit=lib.unit,
        precision=lib.precision,
        polygons_by_spec=polygons_by_spec,
        labels=labels,
    )