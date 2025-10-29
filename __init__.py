# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
tt-forge-models: Shared model implementations used across TT-Forge frontend projects

See README.md for more information.
"""


# Allow importing as either `tt_forge_models` or `third_party.tt_forge_models`
import sys
import types

# Create dummy parent package `third_party` if missing
if "third_party" not in sys.modules:
    third_party_pkg = types.ModuleType("third_party")
    sys.modules["third_party"] = third_party_pkg
else:
    third_party_pkg = sys.modules["third_party"]

# Register `third_party.tt_forge_models` to point to this same module
sys.modules["third_party.tt_forge_models"] = sys.modules[__name__]

# Attach as attribute so normal dotted imports work
setattr(third_party_pkg, "tt_forge_models", sys.modules[__name__])
