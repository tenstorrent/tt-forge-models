# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedModel

# The Cosmos Embed1 custom BertModel calls self.init_weights() directly in
# __init__ instead of self.post_init(). In transformers 5.x, init_weights()
# requires all_tied_weights_keys which is only set by post_init().
_original_init_weights = PreTrainedModel.init_weights


def _patched_init_weights(self):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
            all_submodels=False
        )
    return _original_init_weights(self)


PreTrainedModel.init_weights = _patched_init_weights
