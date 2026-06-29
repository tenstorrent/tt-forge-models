# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral-Small-4 model loader implementation for multimodal modeling.

Mistral-Small-4 (``model_type: mistral3``) is an image-text-to-text model that
pairs a Pixtral vision tower with a new ``mistral4`` text decoder. The text
decoder uses Multi-head Latent Attention (MLA, DeepSeek-style q/kv LoRA
projections) and a 128-expert Mixture-of-Experts FFN (top-4 routed + 1 shared),
and is distributed in fp8 on HuggingFace (vision tower / projector / lm_head are
kept in bf16). The model is loaded through the standard HuggingFace
``Mistral3ForConditionalGeneration`` + ``AutoProcessor`` path.
"""

import re
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    Mistral3ForConditionalGeneration,
)

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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available Mistral-Small-4 multimodal model variants."""

    MISTRAL_SMALL_4_119B = "mistralai/Mistral-Small-4-119B-2603"


class ModelLoader(ForgeModel):
    """Mistral-Small-4 model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MISTRAL_SMALL_4_119B),
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mistral_small_4_multimodal",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mistral-Small-4 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Mistral-Small-4 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Complete the fp8 -> bf16 dequantization for the fused MoE experts.
        #
        # The checkpoint is fp8-quantized (quant_method="fp8"). On a GPU the fp8
        # weights are used directly; with no GPU available transformers' FP8
        # quantizer falls back to dequantizing the model to bf16. That fallback
        # correctly converts the regular ``FP8Linear`` layers (attention, shared
        # experts, lm_head) but leaves the *fused* routed-expert weights
        # (``experts.gate_up_proj`` / ``experts.down_proj``, a stacked
        # [num_experts, ...] tensor consumed by ``torch.grouped_mm``) in
        # Float8_e4m3fn. ``torch._grouped_mm`` rejects fp8 on CPU, so the forward
        # would crash. We finish the dequant here using the per-expert
        # ``*_scale_inv`` scales stored in the checkpoint. Guarded so it is a
        # no-op whenever the experts are not fp8 (e.g. the GPU path).
        self._dequantize_fused_experts(model, pretrained_model_name)

        model.eval()
        self.model = model
        self.config = model.config
        return model

    @staticmethod
    def _dequantize_fused_experts(model, pretrained_model_name):
        """Dequantize any fp8 fused MoE expert weights to bf16 in place.

        The per-expert weight scales (``*_scale_inv``, shape [num_experts, 1, 1])
        are read from the checkpoint safetensors, since the fp8 fallback drops
        them from the loaded module. ``dequant = fp8_weight * scale_inv``.
        """
        fused_experts = [
            (name, mod)
            for name, mod in model.named_modules()
            if re.search(r"layers\.(\d+)\.mlp\.experts$", name)
            and any(
                getattr(mod, attr, None) is not None
                and getattr(mod, attr).dtype == torch.float8_e4m3fn
                for attr in ("gate_up_proj", "down_proj")
            )
        ]
        if not fused_experts:
            return

        # Lazily build {layer_idx: {"gate_up"|"down": scale_inv}} from the checkpoint.
        import json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        index = json.load(
            open(hf_hub_download(pretrained_model_name, "model.safetensors.index.json"))
        )
        weight_map = index["weight_map"]
        openers = {}
        scale_map = {}
        for key, shard in weight_map.items():
            m = re.match(
                r"language_model\.model\.layers\.(\d+)\.mlp\.experts\."
                r"(gate_up_proj|down_proj)_scale_inv$",
                key,
            )
            if not m:
                continue
            layer_idx, proj = int(m.group(1)), m.group(2)
            if shard not in openers:
                openers[shard] = safe_open(
                    hf_hub_download(pretrained_model_name, shard), framework="pt"
                )
            which = "gate_up" if proj == "gate_up_proj" else "down"
            scale_map.setdefault(layer_idx, {})[which] = (
                openers[shard].get_tensor(key).to(torch.float32)
            )

        for name, mod in fused_experts:
            layer_idx = int(re.search(r"layers\.(\d+)\.mlp\.experts$", name).group(1))
            for attr, which in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
                param = getattr(mod, attr, None)
                if param is None or param.dtype != torch.float8_e4m3fn:
                    continue
                scale = scale_map[layer_idx][which].view(-1, 1, 1)
                deq = (param.data.to(torch.float32) * scale).to(torch.bfloat16)
                setattr(mod, attr, torch.nn.Parameter(deq, requires_grad=False))

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        batch_size: int = 1,
    ):
        """Load and return sample inputs for the Mistral-Small-4 multimodal model.

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values, ...)
                  that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
