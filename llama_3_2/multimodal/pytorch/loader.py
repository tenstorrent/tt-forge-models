# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 90B Vision Instruct multimodal (image + text → text) loader.

``meta-llama/Llama-3.2-90B-Vision-Instruct`` is ``MllamaForConditionalGeneration``
(vision encoder + ``MllamaTextModel`` with interleaved self-attn / cross-attn).
Hub weights are gated.
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

    LLAMA_3_2_11B_VISION_INSTRUCT = "Llama-3.2-11B-Vision-Instruct"
    LLAMA_3_2_90B_VISION_INSTRUCT = "Llama-3.2-90B-Vision-Instruct"


class ModelLoader(ForgeModel):
    """Llama 3.2 Vision Instruct loader for image-conditioned generation.

    The 11B variant shares the vision tower and cross-attention design with
    the 90B (identical ``vision_config``; text hidden 4096 / 32 q heads / 8 KV
    heads vs 8192 / 64 / 8), so it is the cheap reproduction for anything on
    the image-conditioned path. Note that ``llama/llama_3_2_vision`` is *not*
    such a reproduction: it loads via ``AutoModelForCausalLM``, which maps
    mllama to ``MllamaForCausalLM`` and drops the vision tower entirely.
    """

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_11B_VISION_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            max_length=256,
        ),
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

    def unpack_forward_output(self, fwd_output):
        """Extract logits from CausalLMOutputWithPast."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

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
        vision_cfg = getattr(self.config, "vision_config", None)
        if vision_cfg is not None:
            vision_heads = vision_cfg.attention_heads
            if vision_heads % model_axis != 0:
                raise ValueError(
                    f"Cannot evenly distribute {vision_heads} vision attention heads "
                    f"across model axis size {model_axis}"
                )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for Mllama vision + text.

        Vision (from the checkpoint keys dropped on the CausalLM path):
        ``vision_model.transformer`` (32 layers) and
        ``vision_model.global_transformer`` (8 gated layers). Each layer is
        MHA ``self_attn`` q/k/v/o plus CLIP-style ``mlp.fc1`` / ``fc2``.
        Patch embed, class/tile embeddings, and LayerNorms stay replicated, as
        does ``multi_modal_projector`` (see below).

        Text decoder is the same self-attn / cross-attn map as causal_lm,
        keyed on ``model.model.language_model``.
        """
        shard_specs = {}

        vision = model.model.vision_model
        for encoder in (vision.transformer, vision.global_transformer):
            for layer in encoder.layers:
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("model", "batch")
                shard_specs[sa.k_proj.weight] = ("model", "batch")
                shard_specs[sa.v_proj.weight] = ("model", "batch")
                shard_specs[sa.o_proj.weight] = ("batch", "model")
                shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
                shard_specs[layer.mlp.fc1.bias] = ("model",)
                shard_specs[layer.mlp.fc2.weight] = ("batch", "model")
                shard_specs[layer.mlp.fc2.bias] = (None,)

        # The projector is the vision -> text bridge: its output is
        # `cross_attention_states`, the k/v input of all 20 cross-attn layers,
        # whose k_proj/v_proj are column-parallel and so contract over a
        # *replicated* hidden dim. Qwen-VL style mergers are two linears
        # (column then row), so the all-reduce of the row-parallel second one
        # hands the text decoder a replicated tensor. Mllama's projector is a
        # single linear, so column-parallel here would leave the handoff
        # sharded along the 8192 text hidden dim with no reduction to restore
        # it. Keep it replicated: the input (7680) is already replicated and
        # this is one small matmul over the image tokens.
        projector = model.model.multi_modal_projector
        shard_specs[projector.weight] = (None, None)
        shard_specs[projector.bias] = (None,)

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
