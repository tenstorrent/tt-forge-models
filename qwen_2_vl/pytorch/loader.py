# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2-VL model loader implementation for vision-language tasks (tensor-parallel).

Qwen2-VL-72B is Qwen2VLForConditionalGeneration: a Qwen2 (GQA, with q/k/v bias)
language backbone + a vision transformer. Tensor-parallel shards only the
language backbone; the vision tower stays replicated (small relative to the 72B LM).
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen2-VL model variants for vision-language tasks."""

    QWEN_2_VL_72B_INSTRUCT = "72B_Instruct"


class ModelLoader(ForgeModel):
    """Qwen2-VL model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_VL_72B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2-VL-72B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_VL_72B_INSTRUCT

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen2-VL",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the (wrapped) Qwen2-VL model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.bfloat16
        )
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # Single forward; disable cache (lives on text_config for VL models).
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        else:
            model.config.use_cache = False
        model.eval()
        self.config = model.config

        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen2-VL model."""
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Keys/order match Wrapper.forward(...). Qwen2-VL also needs
        # mm_token_type_ids (M-RoPE), returned by the processor.
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        return result

    def _num_heads(self):
        tc = getattr(self.config, "text_config", None)
        return getattr(tc, "num_attention_heads", None) or self.config.num_attention_heads

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel (driven by LM heads)."""
        n_heads = self._num_heads()
        if n_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif n_heads % (num_devices // 2) == 0 and num_devices % 2 == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {n_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard spec for the Qwen2 backbone (GQA, q/k/v bias).

        Only the language model is sharded; the vision tower stays replicated.
        Qwen2 attention has bias on q/k/v_proj (not o_proj) — bias is sharded
        the same way as its weight's output dim.
        """
        # Unwrap the Wrapper, then locate the Qwen2 decoder layers. Across
        # transformers versions the LM lives at model.model.language_model
        # (5.x) or model.model; probe for the object that has `.layers`.
        hf = getattr(model, "model", model)
        base = getattr(hf, "model", hf)
        lm = getattr(base, "language_model", None)
        if lm is None or not hasattr(lm, "layers"):
            lm = base
        layers = lm.layers if hasattr(lm, "layers") else lm.model.layers

        shard_specs = {}
        for layer in layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.k_proj.weight] = ("model", "batch")
            shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                if getattr(proj, "bias", None) is not None:
                    shard_specs[proj.bias] = ("model",)

            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        lm_head = getattr(hf, "lm_head", None) or getattr(base, "lm_head", None)
        if lm_head is not None:
            shard_specs[lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
