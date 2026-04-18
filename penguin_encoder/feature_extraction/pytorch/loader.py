# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Penguin-Encoder vision encoder model loader implementation for feature extraction (PyTorch).
"""

import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from typing import Optional


def _flash_attn_varlen_func_stub(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    causal=False,
):
    batch_size = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    for i in range(batch_size):
        qs, qe = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        ks, ke = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[qs:qe].transpose(0, 1)
        ki = k[ks:ke].transpose(0, 1)
        vi = v[ks:ke].transpose(0, 1)
        if num_q_heads != num_kv_heads:
            repeats = num_q_heads // num_kv_heads
            ki = ki.repeat_interleave(repeats, dim=0)
            vi = vi.repeat_interleave(repeats, dim=0)
        out_i = F.scaled_dot_product_attention(
            qi, ki, vi, is_causal=causal, dropout_p=dropout_p
        )
        output[qs:qe] = out_i.transpose(0, 1)
    return output


from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class PenguinEncoderWrapper(torch.nn.Module):
    """Wraps Penguin-Encoder to keep grid_sizes/merge_sizes as CPU-only metadata.

    These integer metadata tensors cannot be materialized on the TT XLA device,
    so they are stored as plain Python attributes (not buffers/parameters) to
    prevent them from being moved to device with .to().
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._cpu_grid_sizes = None
        self._cpu_merge_sizes = None

    def set_metadata(self, grid_sizes, merge_sizes):
        self._cpu_grid_sizes = grid_sizes.cpu().to(torch.int64).clone()
        self._cpu_merge_sizes = merge_sizes.cpu().to(torch.int64).clone()
        cu_seqlens = torch.repeat_interleave(
            self._cpu_grid_sizes[:, 1] * self._cpu_grid_sizes[:, 2],
            self._cpu_grid_sizes[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        self._cpu_cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self._patch_encoder_forward()

    def _patch_encoder_forward(self):
        encoder = self.model.encoder
        original_forward = encoder.forward
        cpu_cu_seqlens = self._cpu_cu_seqlens

        import functools

        @functools.wraps(original_forward)
        def patched_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return result

        from types import MethodType

        def forward_with_precomputed_cu_seqlens(self_enc, **kwargs):
            from transformers.modeling_outputs import BaseModelOutputWithPast
            from transformers.cache_utils import DynamicCache
            from functools import partial
            from transformers.processing_utils import Unpack
            from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

            inputs_embeds = kwargs.get("inputs_embeds")
            grid_sizes = kwargs.get("grid_sizes")
            merge_sizes = kwargs.get("merge_sizes")
            output_attentions = kwargs.get(
                "output_attentions", self_enc.config.output_attentions
            )
            output_hidden_states = kwargs.get(
                "output_hidden_states", self_enc.config.output_hidden_states
            )
            use_cache = kwargs.get("use_cache", self_enc.config.use_cache)
            past_key_values = kwargs.get("past_key_values")
            cache_position = kwargs.get("cache_position")
            position_ids = kwargs.get("position_ids")

            if self_enc.gradient_checkpointing and self_enc.training and use_cache:
                use_cache = False

            if inputs_embeds is None:
                input_ids = kwargs.get("input_ids")
                inputs_embeds = self_enc.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(
                    2, inputs_embeds.shape[0], -1
                )
            elif position_ids.dim() == 2:
                position_ids = position_ids[None, ...].expand(
                    2, position_ids.shape[0], -1
                )

            position_ids = self_enc.get_rope_index(
                grid_sizes, merge_sizes, position_ids
            )

            causal_mask = None
            hidden_states = inputs_embeds
            position_embeddings = self_enc.rotary_emb(hidden_states, position_ids)

            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            # Use precomputed cu_seqlens instead of dynamic repeat_interleave
            cu_seqlens_val = cpu_cu_seqlens.to(device=hidden_states.device)

            for decoder_layer in self_enc.layers[: self_enc.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    cu_seqlens=cu_seqlens_val,
                )

                hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = past_key_values if use_cache else None
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        encoder.forward = MethodType(forward_with_precomputed_cu_seqlens, encoder)

    def forward(self, pixel_values):
        return self.model(pixel_values, self._cpu_grid_sizes, self._cpu_merge_sizes)

    def _apply(self, fn, recurse=True):
        if recurse:
            self.model._apply(fn, recurse)
        return self


class ModelVariant(StrEnum):
    """Available Penguin-Encoder feature extraction model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Penguin-Encoder vision encoder model loader for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/Penguin-Encoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._wrapper = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Penguin-Encoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoImageProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(self.processor, "merge_size"):
            self.processor.merge_size = 1
        return self.processor

    @staticmethod
    def _patch_rope_init_functions():
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:

            def _compute_default_rope_parameters(
                config=None, device=None, seq_len=None, **rope_kwargs
            ):
                if config is not None:
                    rope_params = getattr(config, "rope_parameters", {}) or {}
                    base = rope_params.get(
                        "rope_theta", getattr(config, "rope_theta", 10000)
                    )
                    partial_rotary_factor = (
                        getattr(config, "partial_rotary_factor", 1.0) or 1.0
                    )
                    head_dim = (
                        getattr(config, "head_dim", None)
                        or config.hidden_size // config.num_attention_heads
                    )
                    dim = int(head_dim * partial_rotary_factor)
                else:
                    base = rope_kwargs.get("base", 10000)
                    dim = rope_kwargs["dim"]
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).to(
                            device=device, dtype=torch.float
                        )
                        / dim
                    )
                )
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig, AutoModel

        self._patch_rope_init_functions()

        pretrained_model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if (
            getattr(config, "rope_scaling", None)
            and config.rope_scaling.get("rope_type") == "default"
        ):
            config.rope_scaling = {
                **config.rope_scaling,
                "rope_type": "linear",
                "factor": 1.0,
            }

        model_kwargs = {
            "trust_remote_code": True,
            "config": config,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        model_module = sys.modules.get(type(model).__module__)
        if model_module and not hasattr(model_module, "flash_attn_varlen_func"):
            model_module.flash_attn_varlen_func = _flash_attn_varlen_func_stub

        self._wrapper = PenguinEncoderWrapper(model)
        self._wrapper.eval()
        return self._wrapper

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self.processor is None:
            self._load_processor()

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        grid_sizes = inputs.pop("grid_sizes")
        merge_sizes = inputs.pop("merge_sizes")

        if self._wrapper is not None:
            self._wrapper.set_metadata(grid_sizes, merge_sizes)

        return inputs
