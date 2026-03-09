# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GemmaConfig(PretrainedConfig):
    model_type = "gemma3"

    def __init__(
        self,
        vocab_size=262208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        max_position_embeddings=131072,
        initializer_range=0.02,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256.0,
        sliding_window=4096,
        sliding_window_pattern=6,
        final_logit_soft_cap=None,
        attn_logit_soft_cap=None,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=True,
        rope_scaling=None,
        hidden_activation="gelu_pytorch_tanh",
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_local_base_freq = rope_local_base_freq
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.final_logit_soft_cap = final_logit_soft_cap
        self.attn_logit_soft_cap = attn_logit_soft_cap
        self.rope_scaling = rope_scaling
        self.hidden_activation = hidden_activation
        self.use_cache = use_cache

        # Compute per-layer attention type (same logic as bounty)
        self.layer_types = [
            (
                "sliding_attention"
                if bool((i + 1) % self.sliding_window_pattern)
                else "full_attention"
            )
            for i in range(self.num_hidden_layers)
        ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
