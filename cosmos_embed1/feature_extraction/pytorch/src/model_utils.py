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


# CosmosEmbed1.__init__ does not call post_init(), so all_tied_weights_keys is
# never set on the top-level model.  _adjust_tied_keys_with_tied_pointers
# (called from _finalize_model_loading) accesses it unconditionally — patch it
# to initialise the attribute if absent before delegating to the original.
_original_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers


def _patched_adjust_tied(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    return _original_adjust_tied(self, *args, **kwargs)


PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied
