"""
conftest.py — pytest configuration for the python/tests suite.

vp_utils.parameters_sim is now resolved lazily and accepts an explicit
override via set_parameters_sim().  We inject a fiducial cosmology here so
tests run without needing the real Arepo parameters-usedvalues file (or
the VP_PARAMS_FILE environment variable).
"""

import sys
import tempfile
from pathlib import Path

# Make sure `python/` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

_PARAMS_CONTENT = (
    "Omega0 0.3089\n"
    "OmegaLambda 0.6911\n"
    "OmegaBaryon 0.0486\n"
    "HubbleParam 0.6774\n"
    "BoxSize 500.0\n"
    "UnitLength_in_cm 3.08568e+24\n"
    "UnitMass_in_g 1.989e+43\n"
    "UnitVelocity_in_cm_per_s 100000.0\n"
)

with tempfile.NamedTemporaryFile(
    mode="w", suffix="-parameters-usedvalues", delete=False
) as _f:
    _f.write(_PARAMS_CONTENT)
    _MOCK_PARAMS_PATH = _f.name

import vp_utils

vp_utils.set_parameters_sim(_MOCK_PARAMS_PATH)
