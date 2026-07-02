# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 MoE model loader (google/gemma-4-26B-A4B-it).

google/gemma-4-26B-A4B-it is a ``Gemma4ForConditionalGeneration`` image-text-to-text
VLM with two device-brought-up components:

* **text decoder** (``gemma4_text``) — a 30-layer decoder-only stack with a
  sparse **Mixture-of-Experts** FFN (128 experts, top-8) interleaved with a
  dense MLP, GQA attention with a sliding/full-attention layer pattern, and a
  262k vocab. This is the *key* component; the ``26B-A4B`` (default) variant
  drives the text-only causal-LM path (``input_ids`` + ``attention_mask``).
* **vision tower** (``gemma4_vision``) — a NaFlex-style 27-layer ViT whose patch
  embedder is an ``nn.Linear`` over flattened ``3*patch^2`` pixels (no conv).
  The ``26B-A4B-vision`` variant returns just ``model.model.vision_tower`` and
  drives it with ``pixel_values`` / ``pixel_position_ids`` from the processor.

Both variants share the one 51 GB checkpoint; the vision variant loads the full
model and extracts the vision sub-tower so the checkpoint's real (non-random)
weights are used.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
)

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
    """Available gemma-4-26B-A4B-it components.

    ``26B-A4B`` is the text-only causal-LM (MoE decoder) path — the key
    component. ``26B-A4B-vision`` drives the NaFlex vision tower on its own.
    """

    GEMMA_4_26B_A4B = "26B-A4B"
    GEMMA_4_26B_A4B_TEXT = "26B-A4B-text-decoder"
    GEMMA_4_26B_A4B_VISION = "26B-A4B-vision"


class ModelLoader(ForgeModel):
    """Loader for the google/gemma-4-26B-A4B-it MoE VLM (text + vision)."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=64,
        ),
        ModelVariant.GEMMA_4_26B_A4B_TEXT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=64,
        ),
        ModelVariant.GEMMA_4_26B_A4B_VISION: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=64,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B

    # Variants that bring up the vision tower rather than the text decoder.
    _VISION_VARIANTS = {ModelVariant.GEMMA_4_26B_A4B_VISION}

    sample_text = "What is your favorite city?"
    sample_image_text = "Describe the image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.processor = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        # The vision tower is a CV image feature-extractor; the text decoder is
        # the multimodal image-text-to-text path (driven text-only here).
        if variant in cls._VISION_VARIANTS:
            task = ModelTask.CV_IMAGE_FE
        else:
            task = ModelTask.MM_IMAGE_TTT
        return ModelInfo(
            model="Gemma 4 26B-A4B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_vision(self) -> bool:
        return self._variant in self._VISION_VARIANTS

    def _is_text_decoder(self) -> bool:
        """Isolated text-decoder (``Gemma4TextModel``) variant.

        Drives ``model.model.language_model`` directly, bypassing the
        ``Gemma4Model`` multimodal input-merge glue (whose boolean-mask
        ``index_put`` blocks the full-VLM text path). This isolates the MoE
        decoder core — the key component.
        """
        return self._variant == ModelVariant.GEMMA_4_26B_A4B_TEXT

    def _load_tokenizer(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the gemma-4-26B-A4B-it model.

        For the text variant this returns the full
        ``Gemma4ForConditionalGeneration`` (driven text-only). For the vision
        variant it returns just ``model.model.vision_tower`` (a
        ``Gemma4VisionModel``), keeping the checkpoint's real weights.

        Args:
            dtype_override: torch dtype to load weights in. The checkpoint ships
                in bfloat16; when None, transformers uses the native dtype.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Gemma4ForConditionalGeneration.__init__ does not accept ``use_cache``
        # as a kwarg, so set it on the config (and the text sub-config).
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False
        if hasattr(config, "text_config"):
            config.text_config.use_cache = False
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = Gemma4ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if self._is_vision():
            model = model.model.vision_tower
            model.eval()
        elif self._is_text_decoder():
            model = model.model.language_model
            model.eval()

        self.model = model
        self.config = model.config
        return model

    def unpack_forward_output(self, fwd_output):
        """Extract the primary tensor from the model output.

        The text decoder returns a ``Gemma4CausalLMOutputWithPast`` (``.logits``);
        the vision tower returns a ``BaseModelOutputWithPast``
        (``.last_hidden_state``).
        """
        if hasattr(fwd_output, "logits") and fwd_output.logits is not None:
            return fwd_output.logits
        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state
        return super().unpack_forward_output(fwd_output)

    # ------------------------------------------------------------------
    # Tensor-parallel hooks (text decoder). The MoE experts are stored as
    # stacked 3D parameters and are left replicated; attention and the dense
    # MLP follow the standard Megatron column->row plan. NOTE: the single-chip
    # MoE expert loop is the bring-up blocker, so TP is only meaningful once
    # that is resolved (see report).
    # ------------------------------------------------------------------
    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        n_heads = text_cfg.num_attention_heads
        assert (
            n_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        if self._is_vision():
            return shard_specs
        for layer in model.model.language_model.layers:
            # Dense MLP: column-parallel gate/up, row-parallel down.
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", None)
            shard_specs[attn.o_proj.weight] = (None, "model")
            # k_proj/v_proj and the MoE experts are replicated (skipped):
            # global layers carry a single global KV head (v_proj is None) and
            # the experts are stacked 3D params grouped per token.
        return shard_specs

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def load_image_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Vision-tower inputs: ``pixel_values`` + ``pixel_position_ids``.

        The ``Gemma4Processor`` turns an image into ``pixel_values``
        ``(1, num_patches, 3*patch^2)`` NaFlex patches and ``image_position_ids``
        ``(1, num_patches, 2)`` (x, y) patch coordinates. The vision tower's
        forward signature is ``forward(pixel_values, pixel_position_ids)``, so
        ``image_position_ids`` is passed under the ``pixel_position_ids`` key.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        image_token = getattr(self.processor, "image_token", "<|image|>")
        input_text = f"{image_token}{prompt or self.sample_image_text}"

        proc_out = self.processor(text=input_text, images=image, return_tensors="pt")
        pixel_values = proc_out["pixel_values"]
        pixel_position_ids = proc_out["image_position_ids"]
        if dtype_override is not None:
            pixel_values = cast_input_to_type(pixel_values, dtype_override)
        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Return sample inputs for the current variant.

        Vision variant → ``{pixel_values, pixel_position_ids}``. Text variant →
        ``{input_ids, attention_mask}`` (text-only causal-LM path). Returns a
        dict so the harness passes tensors by keyword.
        """
        if self._is_vision():
            return self.load_image_inputs(
                dtype_override=dtype_override, prompt=prompt, image_url=image_url
            )

        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()
        if getattr(self.tokenizer, "chat_template", None):
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt or self.sample_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            input_text = prompt or self.sample_text
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return {"input_ids": input_ids, "attention_mask": attn_mask}
