# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 90B Vision Instruct multimodal (image + text → text) loader.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

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


class ModelVariant(StrEnum):
    """Available Llama 3.2 multimodal model variants."""

    LLAMA_3_2_90B_VISION_INSTRUCT = "Llama-3.2-90B-Vision-Instruct"


class ModelLoader(ForgeModel):
    """
    Llama 3.2 Vision Instruct loader for image-conditioned generation.
    """

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_90B_VISION_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_90B_VISION_INSTRUCT

    sample_text = "What is this image about?"
    sample_image_url = (
        "https://cdn.britannica.com/61/93061-050-99147DCE/"
        "Statue-of-Liberty-Island-New-York-Bay.jpg"
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
            model="Llama 3.2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _load_processor(self):
        """Load Mllama processor (image + tokenizer)."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return MllamaForConditionalGeneration.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: MllamaForConditionalGeneration in eval mode.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        self._patch_cross_attention_mask()

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # AutoModelForCausalLM maps this checkpoint to MllamaForCausalLM and
        # drops the vision tower. ImageTextToText keeps ConditionalGeneration.
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.eval()
        self.config = model.config
        self.model = model
        return model

    @staticmethod
    def _patch_cross_attention_mask():
        """Replace boolean ``masked_fill`` in Mllama mask prep.

        Hugging Face inverts the 0/1 tile mask then does
        ``masked_fill(mask.to(bool), -inf)``. torch_xla SPMD routinely
        miscompiles that boolean fill (same class of bug as Qwen3-VL
        DeepStack ``index_put_``), which corrupts every cross-attn layer
        and collapses logits PCC. Equivalent math: ``(1 - attend) * -inf``.
        """
        import transformers.models.mllama.modeling_mllama as mllama_mod

        def _prepare_cross_attention_mask(
            cross_attention_mask, num_vision_tokens, dtype
        ):
            batch_size, text_total_length, *_ = cross_attention_mask.shape
            cross_attention_mask = cross_attention_mask.repeat_interleave(
                num_vision_tokens, dim=3
            )
            attend = (
                cross_attention_mask.view(batch_size, text_total_length, -1)
                .unsqueeze(1)
                .to(dtype)
            )
            # amax, not any(): bool OR-reduce legalizes to stablehlo.reduce
            # that TTIR rejects (reduce.14010). For a 0/1 mask, max == any.
            full_text_row_masked_out_mask = attend.amax(dim=-1, keepdim=True)
            cross_attention_mask = (1.0 - attend) * torch.finfo(attend.dtype).min
            cross_attention_mask = cross_attention_mask * full_text_row_masked_out_mask
            return cross_attention_mask, full_text_row_masked_out_mask

        mllama_mod._prepare_cross_attention_mask = _prepare_cross_attention_mask

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build image + text inputs via MllamaProcessor.

        Args:
            dtype_override: Optional dtype for floating-point image tensors.
            batch_size: Only ``batch_size=1`` is supported (tile layouts are
                image-specific and do not tile cleanly).
            prompt: Optional text prompt; defaults to ``sample_text``.
            image_url: Optional image URL/path; defaults to ``sample_image_url``.

        Returns:
            dict: ``input_ids``, ``attention_mask``, ``pixel_values``,
            ``aspect_ratio_ids``, ``aspect_ratio_mask``, ``cross_attention_mask``.
        """
        if batch_size != 1:
            raise ValueError(
                "Llama 3.2 Vision multimodal bring-up only supports batch_size=1 "
                f"(got {batch_size})"
            )

        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")
        text = prompt or self.sample_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        kv_heads = getattr(text_cfg, "num_key_value_heads", attn_heads)
        model_axis = mesh_shape[1]
        if attn_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {model_axis}"
            )
        if kv_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {kv_heads} KV heads "
                f"across model axis size {model_axis}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for Mllama text + vision MLP.

        Text decoder is the same self-attn / cross-attn map as causal_lm.

        Vision ``transformer`` (32 layers) and ``global_transformer`` (8 gated
        layers) use CLIP-style ``mlp.fc1`` / ``fc2`` (1280 → 5120 → 1280).
        Those widths tile the model axis, so they are column/row sharded like
        text MLP. Vision attention stays replicated: 16 heads × head_dim 80,
        and 80 is not a multiple of the 32-wide TT tile, so column-sharding
        q/k/v (the first bring-up) pad/split heads and dropped logits PCC
        into the 0.3s. ``multi_modal_projector`` stays replicated so
        cross-attn k/v see a full 8192-d vision stream.
        """
        shard_specs = {}

        vision = model.model.vision_model
        for encoder in (vision.transformer, vision.global_transformer):
            for layer in encoder.layers:
                shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
                shard_specs[layer.mlp.fc1.bias] = ("model",)
                shard_specs[layer.mlp.fc2.weight] = ("batch", "model")
                shard_specs[layer.mlp.fc2.bias] = (None,)

        language_model = model.model.language_model
        for layer in language_model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.cross_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.k_proj.weight] = ("model", "batch")
            shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")

        shard_specs[language_model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
