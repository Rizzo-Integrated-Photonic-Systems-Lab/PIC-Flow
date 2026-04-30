#!/usr/bin/env python3
"""Generate a test GDS file of a directional coupler for predict_gds.py testing."""

import gdstk

# --- Parameters (typical Si photonic DC) ---
wg_width = 0.45       # um
gap = 0.20             # um, edge-to-edge
coupling_length = 10.0 # um
lead_length = 5.0      # um, straight leads on each side

# Derived positions
y_top = gap / 2 + wg_width / 2   # center of top waveguide
y_bot = -(gap / 2 + wg_width / 2)  # center of bottom waveguide

lib = gdstk.Library()
cell = lib.new_cell("DirectionalCoupler")

total_length = coupling_length + 2 * lead_length
x_start = -total_length / 2
x_end = total_length / 2

# Top waveguide (full length, straight)
cell.add(gdstk.rectangle(
    (x_start, y_top - wg_width / 2),
    (x_end,   y_top + wg_width / 2),
    layer=1, datatype=0,
))

# Bottom waveguide (full length, straight)
cell.add(gdstk.rectangle(
    (x_start, y_bot - wg_width / 2),
    (x_end,   y_bot + wg_width / 2),
    layer=1, datatype=0,
))

out_path = "tools/test_directional_coupler.gds"
lib.write_gds(out_path)
print(f"Wrote {out_path}")
print(f"  wg_width={wg_width}, gap={gap}, coupling_length={coupling_length}")
print(f"  total length={total_length} um")
print(f"  top arm y={y_top}, bottom arm y={y_bot}")
