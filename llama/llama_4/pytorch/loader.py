# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 model loader implementation for multimodal visual question answering
"""
import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from PIL import Image
from typing import Optional
from ....tools.utils import get_file, cast_input_to_type
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


class ModelVariant(StrEnum):
    """Available Llama 4 model variants."""

    LLAMA_4_TINY_RANDOM = "4_Tiny_Random"


class ModelLoader(ForgeModel):
    """Llama 4 model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_4_TINY_RANDOM: ModelConfig(
            pretrained_model_name="yujiepan/llama-4-tiny-random",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_4_TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llama",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "_attn_implementation": "sdpa",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Llama4ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        import transformers.models.llama4.modeling_llama4 as _llama4_mod
        from transformers.models.llama4.modeling_llama4 import (
            Llama4VisionRotaryEmbedding,
            Llama4TextRotaryEmbedding,
        )

        # TT does not support complex64 tensors. Llama4 uses torch.polar / view_as_complex
        # throughout both vision and text RoPE. Replace with equivalent real (cos, sin) arithmetic.

        # _RoPETuple: tuple subclass that exposes .to(device) so (cos, sin) pairs pass
        # through "position_embeddings.to(query_states.device)" in Llama4TextAttention.forward.
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
                # Complex fallback: extract on CPU to avoid TT complex tensor ops.
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
            # reshape_for_broadcast: (seq, 1, head_dim//2) → (1, seq, 1, head_dim//2)
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
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "placeholder"},
                    {"type": "text", "text": "What is this image about?"},
                ],
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        device = self.model.device if self.model is not None else "cpu"
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(
            device
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        arguments = {
            **inputs,
            "use_cache": False,
            "max_new_tokens": 20,
            "do_sample": False,
            "pad_token_id": (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else self.tokenizer.pad_token_id
            ),
        }

        return arguments

    def decode_output(self, outputs, input_length=None):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            decoded_output = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output
