# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os as _os
import site as _site

# This directory shadows the installed PyPI 'vibevoice' library. Extend __path__
# so submodules like 'vibevoice.modular' can still be found in site-packages.
_this_dir = _os.path.abspath(_os.path.dirname(__file__))
for _sp in _site.getsitepackages():
    _candidate = _os.path.join(_sp, "vibevoice")
    if _os.path.isdir(_candidate) and _os.path.abspath(_candidate) != _this_dir:
        if _candidate not in __path__:
            __path__.append(_candidate)
        break
