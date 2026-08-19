# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KAT-Coder vision-language (image + text -> text) model loader.

KAT-Coder-V2.5-Dev declares ``Qwen3_5MoeForConditionalGeneration`` (hybrid
Gated DeltaNet / full-attention MoE text decoder + vision tower). The open
checkpoint ships language-model weights only under
``model.language_model.*``; vision tower weights are absent. This loader still
uses the ConditionalGeneration path and multimodal (image + text) inputs via
the processor / chat template, matching ``qwen_3_5/multimodal``.
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
    """Available KAT-Coder multimodal model variants."""

    KAT_CODER_V2_5_DEV = "KAT-Coder-V2.5-Dev"


class ModelLoader(ForgeModel):
    """KAT-Coder VLM loader (image + text → text) for n300, llmbox, and galaxy."""

    _VARIANTS = {
        ModelVariant.KAT_CODER_V2_5_DEV: LLMModelConfig(
            pretrained_model_name="Kwaipilot/KAT-Coder-V2.5-Dev",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KAT_CODER_V2_5_DEV

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    min_pixels = 56 * 56
    max_pixels = 14 * 28 * 1280

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """
        Args:
            variant: Which KAT-Coder variant to load.
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
            model="KAT-Coder",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        """Load the KAT-Coder ConditionalGeneration model.

        AutoModelForImageTextToText resolves to
        Qwen3_5MoeForConditionalGeneration. Open weights cover the text decoder
        only; missing vision keys are left at init defaults.

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
            # Decoder depth lives in nested text_config; setting it on the outer
            # VLM config is ignored. Keep layer_types consistent with the hybrid
            # linear/full pattern.
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

        # Force use_cache=False so the forward output does not include a hybrid
        # DynamicCache with LinearAttentionLayer entries (no .keys/.values). The
        # runner's pytree comparator (_cache_to_legacy) only handles standard KV
        # layers. Set on outer + nested text_config after load.
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
        """Build a multimodal (image + text) input dict via the processor.

        Args:
            dtype_override: If given, cast pixel_values to this dtype.
            batch_size: Only batch_size=1 supported; pixel_values shapes are image-specific.
            prompt: Override the default sample text prompt.
            image_url: Override the default sample image URL.

        Returns:
            dict with input_ids, attention_mask, and vision tensors when present
            (e.g. pixel_values, image_grid_thw).
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

        Mirrors ``qwen_3_5/multimodal`` MoE path and ``kat_coder/causal_lm``:
        shared expert is sharded here; routed experts are left to custom MoE
        injection. KAT-Coder has 2 KV heads, so full-attention uses the
        contracted-input sharding scheme (not head-parallel).
        """
        shard_specs = {}

        visual = getattr(model.model, "visual", None)
        if visual is not None:
            for block in visual.blocks:
                # Megatron-style: fused qkv column-parallel, proj row-parallel.
                shard_specs[block.attn.qkv.weight] = ("model", "batch")
                if block.attn.qkv.bias is not None:
                    shard_specs[block.attn.qkv.bias] = ("model",)
                shard_specs[block.attn.proj.weight] = ("batch", "model")

                shard_specs[block.mlp.linear_fc1.weight] = ("model", None)
                if block.mlp.linear_fc1.bias is not None:
                    shard_specs[block.mlp.linear_fc1.bias] = ("model",)
                shard_specs[block.mlp.linear_fc2.weight] = (None, "model")

            merger = visual.merger
            shard_specs[merger.linear_fc1.weight] = ("model", "batch")
            if merger.linear_fc1.bias is not None:
                shard_specs[merger.linear_fc1.bias] = ("model",)
            shard_specs[merger.linear_fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # MoE: do NOT shard routed experts here (custom MoE injection).
                # Shared expert is a dense MLP: column-parallel gate/up,
                # row-parallel down.
                shared = mlp.shared_expert
                shard_specs[shared.gate_proj.weight] = ("model", "batch")
                shard_specs[shared.up_proj.weight] = ("model", "batch")
                shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            if layer.layer_type == "full_attention":
                # 2 KV heads: shard contracted (input) dim — same as causal_lm.
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("batch", "model")
                shard_specs[sa.k_proj.weight] = ("batch", "model")
                shard_specs[sa.v_proj.weight] = ("batch", "model")
                shard_specs[sa.o_proj.weight] = ("model", "batch")

            elif layer.layer_type == "linear_attention":
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[la.in_proj_z.weight] = ("model", "batch")
                shard_specs[la.in_proj_b.weight] = ("model", "batch")
                shard_specs[la.in_proj_a.weight] = ("model", "batch")
                shard_specs[la.out_proj.weight] = ("batch", "model")
                if hasattr(la, "conv1d"):
                    shard_specs[la.conv1d.weight] = (None, None, None)
                if hasattr(la, "dt_bias"):
                    shard_specs[la.dt_bias] = ("model",)
                if hasattr(la, "A_log"):
                    shard_specs[la.A_log] = ("model",)

        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_activation_shard_spec(self, model):
        """Sharding constraints for intermediate ACTIVATIONS.

        Replicate gated-delta conv output before the fused-qkv split so the
        split aligns with per-device boundaries under Shardy.
        """
        constraints = {}
        for layer in model.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                constraints[layer.linear_attn.conv1d] = None
        return constraints

    def load_config(self):
        """Return the top-level VLM config (text_config + vision_config)."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
