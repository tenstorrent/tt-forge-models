# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 vision-language (image + text -> text) model loader.

Loads the Qwen 3.5 VLM (vision encoder + hybrid Gated DeltaNet / full-attention
text decoder) via AutoModelForImageTextToText for image-conditioned generation.
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
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    min_pixels = 56 * 56
    max_pixels = 14 * 28 * 1280

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """
        Args:
            variant: Which Qwen 3.5 variant to load.
            num_layers: If set, truncate the text decoder to this many layers.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen 3.5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full Qwen 3.5 VLM (vision encoder + hybrid/MoE text decoder).

        AutoModelForImageTextToText resolves to Qwen3_5ForConditionalGeneration
        for the dense 27B and Qwen3_5MoeForConditionalGeneration for the 35B-A3B
        MoE variant.

        Args:
            dtype_override: torch.dtype to use; defaults to bfloat16.

        Returns:
            torch.nn.Module in eval mode with use_cache=False.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {
            "torch_dtype": dtype_override
            if dtype_override is not None
            else torch.bfloat16,
        }

        if self.num_layers is not None:
            # Qwen 3.5 keeps the decoder depth in the nested text_config; setting
            # it on the outer VLM config is ignored (the model still builds all 64
            # layers). Set text_config and keep layer_types consistent so the
            # hybrid linear/full pattern still includes a full_attention layer.
            config = AutoConfig.from_pretrained(pretrained_model_name)
            text_cfg = getattr(config, "text_config", config)
            text_cfg.num_hidden_layers = self.num_layers
            if getattr(text_cfg, "layer_types", None) is not None:
                text_cfg.layer_types = text_cfg.layer_types[: self.num_layers]
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # Qwen3_5DynamicCache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden. Set it on both the outer VLM config
        # and the nested text_config (the language_model reads text_config);
        # passing use_cache via from_pretrained kwargs is overwritten when the
        # model rebuilds its config from the checkpoint.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

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
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specifications for the full VLM."""
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

        merger = model.model.visual.merger
        shard_specs[merger.linear_fc1.weight] = ("model", "batch")
        if merger.linear_fc1.bias is not None:
            shard_specs[merger.linear_fc1.bias] = ("model",)
        shard_specs[merger.linear_fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # MoE layer (35B-A3B): the routed experts' fused weights
                # (mlp.experts.gate_up_proj / down_proj) are sharded on the
                # expert dimension by get_tt_moe_shard_specs. The router
                # (mlp.gate.weight) and shared_expert_gate stay replicated so
                # every device can score all experts before dispatch. The
                # always-on shared expert is a dense MLP: column-parallel
                # gate/up, row-parallel down.
                shared = mlp.shared_expert
                shard_specs[shared.gate_proj.weight] = ("model", "batch")
                shard_specs[shared.up_proj.weight] = ("model", "batch")
                shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                # Dense layer (27B): plain gate/up/down MLP.
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            if layer.layer_type == "full_attention":
                shard_specs[layer.self_attn.q_proj.weight] = ("batch", "model")
                shard_specs[layer.self_attn.k_proj.weight] = ("batch", "model")
                shard_specs[layer.self_attn.v_proj.weight] = ("batch", "model")
                shard_specs[layer.self_attn.o_proj.weight] = ("model", "batch")

            elif layer.layer_type == "linear_attention":
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_b.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_a.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")

                shard_specs[layer.linear_attn.conv1d.weight] = (None, None, None)
                shard_specs[layer.linear_attn.dt_bias] = ("model",)
                shard_specs[layer.linear_attn.A_log] = ("model",)

        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_activation_shard_spec(self, model):
        """Sharding constraints for intermediate ACTIVATIONS.

        The gated-delta block's fused ``in_proj_qkv`` is sharded contiguously on
        the "model" axis; the subsequent ``torch.split`` into [Q, K, V] cuts that
        sharded axis at points that don't align with the per-device boundaries,
        which miscompiles under Shardy and scrambles q/k/v before the recurrence
        (full-model PCC collapses). Replicating the conv output before the split
        makes the split run on correct data.
        """
        constraints = {}
        for layer in model.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                constraints[layer.linear_attn.conv1d] = None
        return constraints

    def load_config(self):
        """Return the top-level Qwen3_5Config (VLM).

        Sub-configs: config.text_config, config.vision_config
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
