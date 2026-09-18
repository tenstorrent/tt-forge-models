# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiffusionGemma loader (text and image+text paths).

google/diffusiongemma-26B-A4B-it: a multimodal block-diffusion LLM on a Gemma 4
MoE backbone that denoises a block of tokens instead of decoding left-to-right.
~25.8B params.

The checkpoint takes text and images only. Its ``vision_config`` is a 27-layer
``gemma4_vision`` tower (up to 280 soft tokens per image); there is no audio tower and
no audio/video token in ``config.json``, and the transformers encoder documents
itself as not supporting audio or video inputs. The shared ``Gemma4Processor``
does carry an audio feature extractor and a video processor, but those are
inherited Gemma 4 plumbing that this checkpoint cannot consume.

The ``26B-A4B-it`` variant drives the text-only path, ``26B-A4B-it-image``
image+text, and ``26B-A4B-it-image-only`` an image with no text part -- all on
the same checkpoint and the same shard spec. The vision tower and
``embed_vision`` are left replicated (out of the shard map) for this bring-up,
matching the gemma4 loader.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available DiffusionGemma model variants."""

    DIFFUSIONGEMMA_26B_A4B_IT = "26B-A4B-it"
    DIFFUSIONGEMMA_26B_A4B_IT_IMAGE = "26B-A4B-it-image"
    DIFFUSIONGEMMA_26B_A4B_IT_IMAGE_ONLY = "26B-A4B-it-image-only"
    ENCODER = "encoder"


class ModelLoader(ForgeModel):
    """DiffusionGemma loader for the text and image+text block-diffusion paths."""

    _VARIANTS = {
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/diffusiongemma-26B-A4B-it",
        ),
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE: LLMModelConfig(
            pretrained_model_name="google/diffusiongemma-26B-A4B-it",
        ),
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE_ONLY: LLMModelConfig(
            pretrained_model_name="google/diffusiongemma-26B-A4B-it",
        ),
        ModelVariant.ENCODER: LLMModelConfig(
            pretrained_model_name="google/diffusiongemma-26B-A4B-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT

    # The runner calls load_inputs with only dtype_override/batch_size, so the
    # variant is the only channel through which it can ask for image inputs.
    _MODALITY_BY_VARIANT = {
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE: "image",
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE_ONLY: "image_only",
    }
    _TASK_BY_VARIANT = {
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE: ModelTask.MM_IMAGE_TTT,
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT_IMAGE_ONLY: ModelTask.MM_IMAGE_TTT,
    }

    sample_text = "Why is the sky blue?"
    sample_image_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DiffusionGemma",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=cls._TASK_BY_VARIANT.get(variant, ModelTask.NLP_CAUSAL_LM),
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the DiffusionGemmaForBlockDiffusion model (its encoder carries the vision tower)."""
        from transformers import DiffusionGemmaForBlockDiffusion

        if self.processor is None:
            self._load_processor()
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        # The encoder is returned standalone so the staged residency can free it
        # independently. https://github.com/tenstorrent/tt-xla/issues/5538
        if self._variant == ModelVariant.ENCODER:
            self.model = model.model.encoder
            return self.model
        self.model = model
        return model

    def _apply_chat_template(self, content):
        """Run one user turn through the checkpoint's chat template.

        ``content`` is either a plain string (text-only) or the list-of-parts
        form; transformers loads any ``{"type": "image", ...}`` part and hands it
        to the processor, so this one call yields every tensor the forward needs.
        """
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _finalize_inputs(self, inputs, dtype_override, batch_size):
        """Drop the processor's non-model keys, batch, and dtype-cast (dict -> keyword-bound).

        ``cast_input_to_type`` only casts within a numeric category, so the float
        tensors (``pixel_values``) follow ``dtype_override`` while the id/mask
        tensors (``input_ids``, ``attention_mask``, ``mm_token_type_ids``,
        ``image_position_ids``) stay integer.
        """
        inputs = dict(inputs)
        # e.g. num_soft_tokens_per_image: sizes the placeholder span, not a forward arg.
        for key in getattr(self.processor, "unused_input_names", []):
            inputs.pop(key, None)
        for key in list(inputs):
            value = inputs[key].repeat_interleave(batch_size, dim=0)
            inputs[key] = cast_input_to_type(value, dtype_override)
        return inputs

    def load_text_inputs(
        self, dtype_override=None, batch_size=1, prompt: Optional[str] = None
    ):
        """Build text-only inputs: {input_ids, attention_mask, mm_token_type_ids}."""
        if self.processor is None:
            self._load_processor()
        return self._finalize_inputs(
            self._apply_chat_template(prompt or self.sample_text),
            dtype_override,
            batch_size,
        )

    def load_image_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build image+text inputs for the vision path.

        The image occupies a span of soft tokens between ``<boi>``/``<eoi>`` that
        the encoder replaces with vision-tower features. The span is aspect-ratio
        dependent, up to ``vision_soft_tokens_per_image`` (280) -- the sample image
        yields 266 -- so a different image can change the sequence length and thus
        force a recompile. ``mm_token_type_ids`` marks that span, and the encoder
        needs it: ``text_config.use_bidirectional_attention == "vision"``, so the
        span is also what makes attention bidirectional over the image.

        ``pixel_values`` is raw 16x16 RGB patches (768 = 16*16*3), zero-padded to
        the 280-soft-token maximum (280 * pooling_kernel_size^2 = 2520 patches);
        ``image_position_ids`` gives each patch its 2D coordinate, with (-1, -1)
        marking the padding. The 3x3 pooling down to soft tokens happens inside
        the tower, not here.

        Pass ``prompt=""`` for the image-only path -- the message then carries no
        text part and the template emits just the image span. ``prompt=None``
        uses ``sample_image_text``.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids,
                   pixel_values (B, 2520, 768), image_position_ids (B, 2520, 2)}
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        content = [{"type": "image", "image": image}]
        text = self.sample_image_text if prompt is None else prompt
        if text:
            content.append({"type": "text", "text": text})

        return self._finalize_inputs(
            self._apply_chat_template(content), dtype_override, batch_size
        )

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build inputs for the variant's modality.

        The runner passes only dtype_override/batch_size, so the variant is what
        selects image+text (or image-only) over text-only.
        """
        modality = self._MODALITY_BY_VARIANT.get(self._variant)
        if modality in ("image", "image_only"):
            # image_only: prompt="" drops the text part of the user turn.
            if modality == "image_only" and prompt is None:
                prompt = ""
            return self.load_image_inputs(dtype_override, batch_size, prompt, image_url)
        return self.load_text_inputs(dtype_override, batch_size, prompt)

    def _text_layers(self, model):
        """Encoder then decoder text transformer layers."""
        base = model.model
        return list(base.encoder.language_model.layers) + list(base.decoder.layers)

    def get_mesh_config(self, num_devices: int):
        """((1, num_devices), ("batch", "model")); attention is replicated, so
        only the expert axis rides the model axis and must divide it."""
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        assert text_cfg.num_experts % mesh_shape[1] == 0
        return mesh_shape, ("batch", "model")

    def _layers_for_variant(self, model):
        """Layers to shard: the encoder variants' `model` is the encoder submodule, so
        shard its own language_model layers; the vision-tower component shards
        nothing; else shard both encoder+decoder text layers."""
        if self._variant == ModelVariant.ENCODER:
            return list(model.language_model.layers)
        return self._text_layers(model)

    def load_shard_spec(self, model):
        """Shard the dense MLP (col->row), expert-parallel MoE, and the LM head.
        Attention is replicated: the global layers' 2 KV heads can't shard the model
        axis, and head-sharding Q crashes the repeat_kv reshard."""
        shard_specs = {}
        for layer in self._layers_for_variant(model):
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            experts = getattr(layer, "experts", None)
            if experts is not None:
                shard_specs[experts.gate_up_proj] = ("model", None, None)
                shard_specs[experts.down_proj] = ("model", None, None)

        # Vocab-shard the head: replicated it is 1.375 GiB per device, and tilizing
        # it on top of the weights lands 1.09 GiB over the 11.97 GiB budget.
        # lm_head and embed_tokens are one tied parameter in the checkpoint, but
        # model.to(device) breaks the tie, so both must be marked by identity, and
        # here rather than after placement -- by then the replicated copy exists.
        for holder, attr in (
            (model, "lm_head"),
            (getattr(getattr(model, "model", None), "decoder", None), "embed_tokens"),
            (
                getattr(getattr(model, "language_model", None), "embed_tokens", None),
                None,
            ),
        ):
            if holder is None:
                continue
            mod = holder if attr is None else getattr(holder, attr, None)
            w = getattr(mod, "weight", None)
            if w is not None:
                shard_specs[w] = ("model", None)
        return shard_specs
