# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Vendored Pyramid Flow CausalVideoVAE, from upstream `video_vae/`. Changes are
limited to the context-parallel `utils` imports (replaced by a local stub), an
SPDX header, and the reformatting the repo's black hook applies.

Only the modules the decode path needs are vendored. Upstream's `__init__`
also exports `LPIPSWithDiscriminator` and `CausalVideoVAELossWrapper`, which
pull in `modeling_loss`, `modeling_lpips` and `modeling_discriminator` — all
training-only, so they are deliberately left out.
"""

from .modeling_causal_vae import CausalVideoVAE
