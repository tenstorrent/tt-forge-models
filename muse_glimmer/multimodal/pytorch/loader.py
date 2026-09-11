# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Muse-Glimmer 30B multimodal (image + text → text) model loader.

``meta-models/Muse-Glimmer-30B`` is MuseGlimmerForConditionalGeneration.
Native support requires transformers>=5.15 (pinned in requirements.txt).
"""

from typing import Optional

import torch
from PIL import Image

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
from ....tools.utils import cast_input_to_type, get_file

# NOTE: ``transformers`` is intentionally NOT imported at module top level.
# This model pins transformers==5.15.0 (see requirements.txt). The test runner
# installs that pin at test time and purges transformers from sys.modules. A
# top-level import would bind Auto* classes to whatever transformers was loaded
# during pytest collection. Auto* classes are imported lazily in the methods
# that use them.


class ModelVariant(StrEnum):
    """Available Muse-Glimmer multimodal model variants."""

    MUSE_GLIMMER_30B = "Muse-Glimmer-30B"


class ModelLoader(ForgeModel):
    """Muse-Glimmer 30B loader for image-conditioned generation."""

    _VARIANTS = {
        ModelVariant.MUSE_GLIMMER_30B: LLMModelConfig(
            pretrained_model_name="meta-models/Muse-Glimmer-30B",
            # Must fit max_image_tokens + prompt; keep short for DRAM.
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSE_GLIMMER_30B

    sample_text = "Describe this image."
    sample_image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/p-blog/candy.JPG"
    )

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
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Muse-Glimmer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _load_processor(self):
        """Load Muse-Glimmer multimodal processor (image + tokenizer)."""
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return MuseGlimmerForConditionalGeneration.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: MuseGlimmerForConditionalGeneration in eval mode.
        """
        from transformers import AutoModelForMultimodalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.eval()
        self.config = model.config
        self.model = model
        print(f"Model loaded: {model}")
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build image + text inputs via MuseGlimmerProcessor.

        Args:
            dtype_override: Optional dtype for floating-point image tensors.
            batch_size: Only ``batch_size=1`` is supported (image grids are
                image-specific and do not tile cleanly).
            prompt: Optional text prompt; defaults to ``sample_text``.
            image_url: Optional image URL/path; defaults to ``sample_image_url``.

        Returns:
            dict: ``input_ids``, ``attention_mask``, ``pixel_values``,
            ``image_grid_thw`` (and related processor keys).
        """
        if batch_size != 1:
            raise ValueError(
                "Muse-Glimmer multimodal bring-up only supports batch_size=1 "
                f"(got {batch_size})"
            )

        if self.processor is None:
            self._load_processor()

        # Resolve to a local file so offline/CI hosts still work; pass a PIL
        # image in the chat content (HF Muse demo accepts path/URL/PIL).
        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")
        # Extra bound on pixel area before the processor's smart_resize; keeps
        # vision matmuls off the critical DRAM path on 8-chip meshes.
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        text = prompt or self.sample_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
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

    def unpack_forward_output(self, fwd_output):
        """Extract logits from MuseGlimmerCausalLMOutputWithPast."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        mesh_shape = (1, num_devices)
        if attn_heads % mesh_shape[1] != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {mesh_shape[1]}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for the Muse-Glimmer text decoder.

        Same ``qwen_3_5``-style map as the causal_lm loader. Vision tower /
        adapter / projection stay replicated (out of the map) for bring-up.
        """
        shard_specs = {}

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

            sa = layer.self_attn
            # k_proj / v_proj replicated: GQA num_key_value_heads=2 cannot split
            # evenly across an 8-wide model axis.
            shard_specs[sa.q_proj.weight] = ("batch", "model")
            shard_specs[sa.gate_proj.weight] = ("batch", "model")
            shard_specs[sa.o_proj.weight] = ("model", "batch")

        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
