# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RWKV6 model implementation compatible with GGUF weight loading via transformers.

The transformers library (v5.x) does not natively support loading RWKV6 from GGUF.
This module provides:
  - RWKV6Config: config class registered with AutoConfig
  - RWKV6ForCausalLM: model class registered with AutoModelForCausalLM
  - Monkey-patching to register the rwkv6 GGUF architecture

Parameter names match the gguf library's HF name mapping for rwkv6 so that
get_gguf_hf_weights_map() can build the correct gguf->hf tensor name mapping.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class RWKV6Config(PretrainedConfig):
    model_type = "rwkv6"

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_heads=64,
        head_size=64,
        time_mix_extra_dim=64,
        time_decay_extra_dim=128,
        layer_norm_epsilon=1e-5,
        rescale_every=6,
        context_length=1048576,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.time_mix_extra_dim = time_mix_extra_dim
        self.time_decay_extra_dim = time_decay_extra_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.context_length = context_length
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class _WeightParam(nn.Module):
    """Wraps a parameter tensor so it appears as `.weight` in state_dict."""

    def __init__(self, *shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


class RWKV6TimeMixing(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        h = config.hidden_size
        num_heads = config.num_heads
        head_size = config.head_size
        extra = config.time_mix_extra_dim
        decay_extra = config.time_decay_extra_dim

        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_size = h

        self.time_maa_x = _WeightParam(h)
        self.time_maa_w = _WeightParam(h)
        self.time_maa_k = _WeightParam(h)
        self.time_maa_v = _WeightParam(h)
        self.time_maa_r = _WeightParam(h)
        self.time_maa_g = _WeightParam(h)

        self.time_maa_w1 = _WeightParam(h, extra * 5)
        self.time_maa_w2 = _WeightParam(extra, h, 5)

        self.time_decay = _WeightParam(h)
        self.time_decay_w1 = _WeightParam(h, decay_extra)
        self.time_decay_w2 = _WeightParam(decay_extra, h)

        self.time_faaaa = _WeightParam(num_heads, head_size)

        self.receptance = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)
        self.value = nn.Linear(h, h, bias=False)
        self.gate = nn.Linear(h, h, bias=False)
        self.output = nn.Linear(h, h, bias=False)

        self.ln_x = nn.GroupNorm(num_heads, h, eps=64e-5)

    def forward(self, x):
        B, T, C = x.size()
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x

        xxx = x + delta * self.time_maa_x.weight
        mw = torch.tanh(xxx @ self.time_maa_w1.weight)
        mw = mw.view(B, T, 5, -1)

        w2 = self.time_maa_w2.weight
        mk = (mw[:, :, 0] @ w2[:, :, 0]) + self.time_maa_k.weight
        mv = (mw[:, :, 1] @ w2[:, :, 1]) + self.time_maa_v.weight
        mr = (mw[:, :, 2] @ w2[:, :, 2]) + self.time_maa_r.weight
        mg = (mw[:, :, 3] @ w2[:, :, 3]) + self.time_maa_g.weight
        mww = (mw[:, :, 4] @ w2[:, :, 4]) + self.time_maa_w.weight

        r = self.receptance(x + delta * mr)
        k = self.key(x + delta * mk)
        v = self.value(x + delta * mv)
        g = self.gate(x + delta * mg)

        w = self.time_decay.weight + torch.tanh(
            (x + delta * mww) @ self.time_decay_w1.weight
        ) @ self.time_decay_w2.weight
        w = -torch.exp(w)

        r = r.view(B, T, self.num_heads, self.head_size)
        k = k.view(B, T, self.num_heads, self.head_size)
        v = v.view(B, T, self.num_heads, self.head_size)
        bonus = self.time_faaaa.weight

        a = torch.einsum("bthn,bthn->bth", k, v).unsqueeze(-1) * r
        out = a.reshape(B, T, C)

        out = self.ln_x(out.transpose(1, 2)).transpose(1, 2)
        out = out * torch.sigmoid(g)
        return self.output(out)


class RWKV6ChannelMixing(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        h = config.hidden_size
        inter = config.intermediate_size

        self.time_maa_k = _WeightParam(h)
        self.time_maa_r = _WeightParam(h)

        self.key = nn.Linear(h, inter, bias=False)
        self.receptance = nn.Linear(h, h, bias=False)
        self.value = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x

        k = self.key(x + delta * self.time_maa_k.weight)
        k = torch.relu(k) ** 2
        v = self.value(k)
        r = torch.sigmoid(self.receptance(x + delta * self.time_maa_r.weight))
        return r * v


class RWKV6Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        h = config.hidden_size
        eps = config.layer_norm_epsilon

        if layer_idx == 0:
            self.pre_ln = nn.LayerNorm(h, eps=eps)

        self.ln1 = nn.LayerNorm(h, eps=eps)
        self.ln2 = nn.LayerNorm(h, eps=eps)
        self.attention = RWKV6TimeMixing(config, layer_idx)
        self.feed_forward = RWKV6ChannelMixing(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, x):
        if hasattr(self, "pre_ln"):
            x = self.pre_ln(x)
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class RWKV6PreTrainedModel(PreTrainedModel):
    config_class = RWKV6Config
    base_model_prefix = "rwkv"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV6Block"]


class RWKV6Model(RWKV6PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [RWKV6Block(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.ln_out(hidden_states)


class RWKV6ForCausalLM(RWKV6PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RWKV6Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        hidden_states = self.rwkv(input_ids=input_ids, inputs_embeds=inputs_embeds)
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def patch_transformers_rwkv6_gguf():
    """Register RWKV6 architecture with transformers' GGUF loading pipeline."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "rwkv6" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("rwkv6")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["rwkv6"] = {
        "context_length": "context_length",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.head_count": "num_heads",
        "rescale_every_n_layers": "rescale_every",
        "wkv.head_size": "head_size",
        "time_mix_extra_dim": "time_mix_extra_dim",
        "time_decay_extra_dim": "time_decay_extra_dim",
        "vocab_size": "vocab_size",
    }

    AutoConfig.register("rwkv6", RWKV6Config)
    AutoModelForCausalLM.register(RWKV6Config, RWKV6ForCausalLM)

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    class GGUFRwkv6Converter:
        """Converter for RWKV World tokenizer from GGUF.

        The RWKV World tokenizer is trie-based without BPE merges or scores.
        We construct a BPE tokenizer with the GGUF vocabulary and no merges;
        byte_fallback handles unknown characters.
        """

        def __init__(self, tokenizer_dict):
            self.tokens = tokenizer_dict.get("tokens", [])
            self.additional_kwargs = {}

        def converted(self):
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders

            vocab = {tok: i for i, tok in enumerate(self.tokens)}
            tokenizer = Tokenizer(
                models.BPE(vocab=vocab, merges=[], byte_fallback=True)
            )
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
            tokenizer.decoder = decoders.ByteLevel()
            return tokenizer

    if "rwkv6" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["rwkv6"] = GGUFRwkv6Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "rwkv6":
            if "num_heads" not in config or config.get("num_heads") == 0:
                head_size = config.get("head_size", 64)
                hidden_size = config.get("hidden_size", 4096)
                config["num_heads"] = hidden_size // head_size
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint
