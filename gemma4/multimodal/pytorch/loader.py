# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 model loader implementation for multimodal (image-text-to-text) modeling.

google/gemma-4-31B-it is a ``Gemma4ForConditionalGeneration`` vision-language
model: a SigLIP-style vision tower (27 layers, patch 16) feeding a Gemma4 text
decoder (60 layers, GQA 32/16 heads, head_dim 256, sliding-window 1024). This
loader drives its image+text path — the processor turns a chat prompt with an
``<image>`` placeholder plus a PIL image into ``input_ids`` (with the image
token span), ``attention_mask``, ``token_type_ids`` (0=text/1=image) and
``pixel_values``. Distinct from ``gemma4/pytorch`` which brings up the any-to-any
``Gemma4UnifiedForConditionalGeneration`` (google/gemma-4-12B) text path.
"""

import os
from typing import Optional

from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma4ForConditionalGeneration,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Gemma4 multimodal model variants."""

    GEMMA_4_31B_IT = "google/gemma-4-31B-it"


class ModelLoader(ForgeModel):
    """Gemma4 multimodal (image-text-to-text) model loader."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_31B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_4_31B_IT),
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_31B_IT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def requires_model_rewrites(self) -> bool:
        """Gemma4's interleaved sliding-window/full attention needs the TT
        causal-mask rewrite applied before compiling on device."""
        return True

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="gemma_4_multimodal",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the multimodal processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma4 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                (checkpoint-native) dtype. The checkpoint ships in bfloat16.

        Returns:
            torch.nn.Module: The Gemma4ForConditionalGeneration model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Gemma4ForConditionalGeneration.__init__ does not accept ``use_cache``
        # as a kwarg, so set it on the config (text sub-config too) — a single
        # forward pass needs no KV cache.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False
        if hasattr(config, "text_config"):
            config.text_config.use_cache = False
        model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = Gemma4ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def unpack_forward_output(self, fwd_output):
        """Extract the logits tensor from the Gemma4 conditional-generation output."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        batch_size: int = 1,
    ):
        """Load and return sample image+text inputs for the Gemma4 VLM.

        google/gemma-4-31B-it is an instruct checkpoint with a chat template, so
        the prompt is built with ``apply_chat_template`` over an image + text
        turn. Only ``pixel_values`` is cast to ``dtype_override``; the id/mask
        tensors stay integer.

        Returns:
            dict: {input_ids, attention_mask, token_type_ids, pixel_values}.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        # Diagnostic-only text-only path (isolates the text decoder from the
        # vision tower when triage-ing PCC). Not used by the default runner path.
        if os.environ.get("GEMMA4_TEXT_ONLY") == "1":
            text_prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt or self.sample_text}]}],
                add_generation_prompt=True,
            )
            inputs = self.processor(text=text_prompt, return_tensors="pt")
            return dict(inputs)

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
        inputs = dict(inputs)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP.

        The Gemma4 text decoder is a standard GQA causal-LM stack (32 query /
        16 KV heads), both divisible by a 4-way model axis, so query and KV
        heads are sharded on the ``model`` axis.
        """
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        assert (
            text_cfg.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map for the Gemma4 text decoder.

        Column-parallel (shard out_features on the model axis) for q/k/v_proj
        and the MLP gate/up projections; row-parallel for o_proj and down_proj.
        The vision tower, norms and embeddings are left replicated.
        """
        shard_specs = {}
        base = getattr(model, "model", model)
        language_model = getattr(base, "language_model", None)
        if language_model is None:
            return None

        for layer in language_model.layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            if getattr(attn, "k_proj", None) is not None:
                shard_specs[attn.k_proj.weight] = ("model", "batch")
            if getattr(attn, "v_proj", None) is not None:
                shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        return shard_specs
