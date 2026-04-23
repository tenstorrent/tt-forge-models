# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mantis-8M time series classification foundation model.
"""
import os
import site

try:
    from .pytorch import ModelLoader, ModelVariant
except ImportError:
    # Relative imports fail when mantis is imported as a top-level package
    # (e.g. by video_score which needs mantis-vl's mantis.models.idefics2).
    # Extend __path__ with the installed mantis-vl so submodule lookups work.
    pass

for _sp in site.getsitepackages():
    _candidate = os.path.join(_sp, "mantis")
    if os.path.isdir(_candidate) and _candidate not in __path__:
        __path__.append(_candidate)
        break
