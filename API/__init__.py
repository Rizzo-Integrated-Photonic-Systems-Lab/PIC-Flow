"""
v1 API skeleton (GDS -> eps grid -> model -> fields/S-params).

Minimal and function-based so you can evolve into a full SDK.
"""

from .process import ProcessKit2D
from .processkits import SOI220_TE_1550
from .spec import GridSpec, SourceSpec, SimSpec
from .run import simulate_gds, run_forward
from .result import Result