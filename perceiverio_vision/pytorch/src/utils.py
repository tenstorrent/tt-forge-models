# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math


class MathShim:
    """Drop-in replacement for numpy ops used in modeling_perceiver.py.
    Avoids numpy references that break torch.compile/dynamo tracing."""

    pi = math.pi

    def prod(self, x):
        return math.prod(x) if hasattr(x, "__iter__") else x

    def round(self, x):
        return round(x)
