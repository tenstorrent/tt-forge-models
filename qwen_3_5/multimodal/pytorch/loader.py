# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 multimodal (VLM) loader for causal language modeling on n300 / llmbox / galaxy.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available Qwen 3.5 multimodal model variants."""

    QWEN_3_5_27B = "Qwen/Qwen3.5-27B"
    QWEN_3_5_35B_A3B = "Qwen/Qwen3.5-35B-A3B"
    QWEN_3_5_122B_A10B = "Qwen/Qwen3.5-122B-A10B"
    QWEN_3_5_397B_A17B = "Qwen/Qwen3.5-397B-A17B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 VLM loader (image + text → text) for n300, llmbox, and galaxy."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_27B),
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_35B_A3B),
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_122B_A10B),
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_397B_A17B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_397B_A17B),
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(
        self, variant: Optional[ModelVariant] = None):
        """
        Args:
            variant: Which Qwen 3.5 variant to load.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full Qwen 3.5 VLM (vision encoder + hybrid/MoE text decoder).

        AutoModelForImageTextToText resolves to Qwen3_5ForConditionalGeneration.
        MTP weights (^mtp.*) are silently dropped on load.

        Args:
            dtype_override: torch.dtype to use; defaults to bfloat16.

        Returns:
            torch.nn.Module in eval mode with use_cache=False.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {
            "torch_dtype": dtype_override if dtype_override is not None else torch.bfloat16,
        }

        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build a multimodal (image + text) input dict via the Qwen 3.5 processor.

        Args:
            dtype_override: If given, cast pixel_values to this dtype.
            batch_size: Only batch_size=1 supported; pixel_values shapes are image-specific.
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
        """Device mesh for tensor parallelism.
        
        Returns:
            (mesh_shape, axis_names)
        """
        if num_devices == 2:
            mesh_shape = (1, 2)
        elif num_devices == 8:
            mesh_shape = (2, 4)
        elif num_devices == 32:
            mesh_shape = (8, 4)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specifications for the full VLM.
        """
        shard_specs = {}

        for block in model.model.visual.blocks:
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            if block.attn.qkv.bias is not None:
                shard_specs[block.attn.qkv.bias] = ("model",)
            shard_specs[block.attn.proj.weight] = ("batch", "model")

            shard_specs[block.mlp.linear_fc1.weight] = ("model", "batch")
            if block.mlp.linear_fc1.bias is not None:
                shard_specs[block.mlp.linear_fc1.bias] = ("model",)
            shard_specs[block.mlp.linear_fc2.weight] = ("batch", "model")

        merger = model.model.visual.merger
        shard_specs[merger.linear_fc1.weight] = ("model", "batch")
        if merger.linear_fc1.bias is not None:
            shard_specs[merger.linear_fc1.bias] = ("model",)
        shard_specs[merger.linear_fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            if hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    shard_specs[expert.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if layer.layer_type == "full_attention":
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            elif layer.layer_type == "linear_attention":
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_b.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_a.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")

                shard_specs[layer.linear_attn.conv1d.weight] = ("model", None, None)
                shard_specs[layer.linear_attn.dt_bias] = ("model",)
                shard_specs[layer.linear_attn.A_log] = ("model",)

        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Return the top-level Qwen3_5Config (VLM).

        Sub-configs: config.text_config, config.vision_config
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

