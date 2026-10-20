"""aqfm_prescreen: faithful port of the frozen AQFM runners + pre-screen tools.

House rule: nothing is written to the user's home directory.  Caches that
libraries put under ~ by default are redirected into the package folder (which
lives on scratch) unless the caller already set them.
"""
import os as _os
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _var, _sub in (("SCIKIT_LEARN_DATA", "data/sklearn_cache"),   # fetch_openml
                   ("CUPY_CACHE_DIR", ".cupy_cache"),             # GPU kernels
                   ("PIP_CACHE_DIR", ".pip_cache")):
    _os.environ.setdefault(_var, str(_ROOT / _sub))
