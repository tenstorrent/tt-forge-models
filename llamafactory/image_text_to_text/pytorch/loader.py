# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaMAFactory tiny-random-Llama-4 model loader implementation for image-text-to-text tasks.
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Optional

# Disable transformers runtime checks that use boolean mask indexing
# (inputs_embeds[special_image_mask]) during torch.compile graph capture.
# Under XLA that is a data-dependent op which causes Error code 13.
os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available LLaMAFactory Llama-4 model variants."""

    TINY_RANDOM_LLAMA_4 = "tiny_random_Llama_4"


class ModelLoader(ForgeModel):
    """LLaMAFactory tiny-random-Llama-4 model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_LLAMA_4: LLMModelConfig(
            pretrained_model_name="llamafactory/tiny-random-Llama-4",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_LLAMA_4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LLaMAFactory",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            **kwargs,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        import transformers.models.llama4.modeling_llama4 as _llama4_mod
        from transformers.models.llama4.modeling_llama4 import (
            Llama4VisionRotaryEmbedding,
            Llama4TextRotaryEmbedding,
        )

        # TT does not support complex64 tensors. Llama4 uses torch.polar / view_as_complex
        # throughout both vision and text RoPE. Replace with equivalent real (cos, sin) arithmetic.

        class _RoPETuple(tuple):
            def to(self, *args, **kwargs):
                return _RoPETuple(t.to(*args, **kwargs) for t in self)

        def _text_rope_forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            cos_freqs = torch.cos(freqs) * self.attention_scaling
            sin_freqs = torch.sin(freqs) * self.attention_scaling
            return _RoPETuple([cos_freqs, sin_freqs])

        Llama4TextRotaryEmbedding.forward = _text_rope_forward

        def _apply_rotary_emb(xq, xk, freqs_cis):
            if isinstance(freqs_cis, tuple):
                cos, sin = freqs_cis
            else:
                freqs_cpu = freqs_cis.cpu()
                cos = freqs_cpu.real.to(xq.device)
                sin = freqs_cpu.imag.to(xq.device)
            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]
            xq_f = xq.float()
            xk_f = xk.float()
            xq_r, xq_i = xq_f[..., ::2], xq_f[..., 1::2]
            xk_r, xk_i = xk_f[..., ::2], xk_f[..., 1::2]
            xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], -1).flatten(-2)
            xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], -1).flatten(-2)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        _llama4_mod.apply_rotary_emb = _apply_rotary_emb

        def _vision_apply_rotary_emb(query, key, freqs_ci):
            if isinstance(freqs_ci, tuple):
                cos, sin = freqs_ci
            else:
                cos = freqs_ci.real
                sin = freqs_ci.imag
            device = query.device
            cos = cos.to(device).view(1, query.shape[1], 1, -1)
            sin = sin.to(device).view(1, query.shape[1], 1, -1)
            query_f = query.float()
            key_f = key.float()
            q_r, q_i = query_f[..., ::2], query_f[..., 1::2]
            k_r, k_i = key_f[..., ::2], key_f[..., 1::2]
            q_out = torch.stack([q_r * cos - q_i * sin, q_r * sin + q_i * cos], -1).flatten(-2)
            k_out = torch.stack([k_r * cos - k_i * sin, k_r * sin + k_i * cos], -1).flatten(-2)
            return q_out.type_as(query), k_out.type_as(key)

        _llama4_mod.vision_apply_rotary_emb = _vision_apply_rotary_emb

        # freqs_ci is computed in __init__ as a plain attribute (not register_buffer).
        # Under init_empty_weights (default low_cpu_mem_usage) it becomes a meta tensor.
        # Re-init from vision config, then decompose complex into real (cos, sin) components.
        vision_config = model.vision_model.config
        for module in model.modules():
            if isinstance(module, Llama4VisionRotaryEmbedding):
                Llama4VisionRotaryEmbedding.__init__(module, vision_config)
                module.cos_freqs = module.freqs_ci.real.contiguous()
                module.sin_freqs = module.freqs_ci.imag.contiguous()

        def _vision_rope_forward(self, hidden_states):
            return (self.cos_freqs, self.sin_freqs)

        Llama4VisionRotaryEmbedding.forward = _vision_rope_forward

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image?",
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.processor is None:
            self._load_processor()

        tokenizer = self.processor.tokenizer

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return tokenizer.decode(next_token_id)
