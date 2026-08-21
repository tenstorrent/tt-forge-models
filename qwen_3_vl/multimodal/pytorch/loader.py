# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL model loader implementation for multimodal (image + text) inference.
"""

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Qwen 3 VL model variants for multimodal inference."""

    QWEN_3_VL_32B_INSTRUCT = "32b_instruct"


class ModelLoader(ForgeModel):
    """Qwen 3 VL model loader implementation for image + text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_INSTRUCT

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    min_pixels = 56 * 56
    max_pixels = 14 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the Qwen3-VL processor with the same pixel bounds as Qwen 3.5."""
        kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3 VL model for multimodal inference.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the compiled graph does not emit a
        # DynamicCache (the runner's pytree comparator can't leaf-diff it, and
        # min-PCC over 128 KV tensors hides the logits score). Set on both the
        # outer VLM config and nested text_config; from_pretrained kwargs are
        # overwritten when the model rebuilds config from the checkpoint.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        print(f"model: {model}")
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build a multimodal (image + text) input dict via the Qwen 3 VL processor.

        Args:
            dtype_override: If given, cast pixel_values to this dtype.
            batch_size: Only batch_size=1 is supported; pixel_values shapes are image-specific.
            prompt: Override the default sample text prompt.
            image_url: Override the default sample image URL.

        Returns:
            dict with input_ids, attention_mask, pixel_values, image_grid_thw.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url or self.sample_image_url},
                    {"type": "text", "text": prompt or self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specifications for the full VLM.

        Module layout (Qwen3VLForConditionalGeneration):
            model.visual.blocks / merger / deepstack_merger_list
            model.language_model.layers (dense full-attention)
            lm_head

        Language attention is head-parallel (q/k/v column, o row) so the residual
        stays on ``batch``, matching embed/MLP. The Qwen 3.5 27B flipped qkv
        layout (``("batch", "model")``) is for hybrid gated-delta residuals; on
        this dense 32B GQA (64 Q / 8 KV heads) it reshards batch<->model through
        attention and collapses PCC (~0.57). 8 KV heads tile the 8-wide model
        axis.
        """
        if not hasattr(model, "lm_head") and hasattr(model, "model"):
            model = model.model

        shard_specs = {}

        for block in model.model.visual.blocks:
            # Megatron-style: fused qkv is column-parallel (shard output / heads),
            # proj is row-parallel (shard input / heads -> all-reduce after).
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            if block.attn.qkv.bias is not None:
                shard_specs[block.attn.qkv.bias] = ("model",)
            shard_specs[block.attn.proj.weight] = ("batch", "model")

            shard_specs[block.mlp.linear_fc1.weight] = ("model", None)
            if block.mlp.linear_fc1.bias is not None:
                shard_specs[block.mlp.linear_fc1.bias] = ("model",)
            shard_specs[block.mlp.linear_fc2.weight] = (None, "model")

        def _shard_patch_merger(merger):
            shard_specs[merger.linear_fc1.weight] = ("model", "batch")
            if merger.linear_fc1.bias is not None:
                shard_specs[merger.linear_fc1.bias] = ("model",)
            shard_specs[merger.linear_fc2.weight] = ("batch", "model")

        _shard_patch_merger(model.model.visual.merger)
        for merger in model.model.visual.deepstack_merger_list:
            _shard_patch_merger(merger)

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

            sa = layer.self_attn
            shard_specs[sa.q_proj.weight] = ("model", "batch")
            shard_specs[sa.k_proj.weight] = ("model", "batch")
            shard_specs[sa.v_proj.weight] = ("model", "batch")
            shard_specs[sa.o_proj.weight] = ("batch", "model")

        # VLM text decoder embedding lives at model.model.language_model
        # (model.model only exposes .visual / .language_model). Vocab on
        # "model", hidden on "batch" -> embedding output keeps hidden on
        # "batch", matching the residual axis.
        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
