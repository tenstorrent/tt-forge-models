# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth GLM-4.6V GGUF model loader implementation for image-text-to-text tasks.

The GGUF checkpoint only contains the text backbone (glm4moe architecture).
We load the vision config from the base (non-GGUF) model and combine
them into a full Glm4vMoeConfig, following the same pattern as GLM-OCR GGUF.
"""
import importlib.metadata

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig,
    Glm4vMoeConfig,
)
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils

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
from ....tools.utils import get_file
from PIL import Image


def _patched_is_gguf_available(min_version="0.10.0"):
    """Fix is_gguf_available for dynamically-installed gguf (lacks __version__)."""
    try:
        from packaging import version

        gguf_version = importlib.metadata.version("gguf")
        return version.parse(gguf_version) >= version.parse(min_version)
    except Exception:
        return False


_gguf_utils.is_gguf_available = _patched_is_gguf_available


def _patch_transformers_glm4moe_gguf():
    """Patch transformers to support loading glm4moe GGUF checkpoints as glm4v_moe.

    The unsloth GLM-4.6V GGUF uses general.architecture=glm4moe, which is the
    text backbone of the GLM-4.6V vision-language model. This patch registers
    glm4moe in GGUF_SUPPORTED_ARCHITECTURES with field mappings for
    Glm4vMoeTextConfig, and fixes the model_type and rope_parameters after loading.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm4moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
            "attention.value_length": None,
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_feed_forward_length": "moe_intermediate_size",
            "nextn_predict_layers": None,
            "expert_gating_func": None,
            "rope.dimension_sections": None,
        }

    if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFQwen2Converter
    if "glm4v_moe_text" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4v_moe_text"] = GGUFQwen2Converter

    orig_load = _gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4v_moe_text"
            # Convert rope_theta scalar to rope_parameters dict expected by Glm4vMoeTextConfig.
            # partial_rotary_factor = rope_dim / head_dim = qk_rope_head_dim / (hidden_size / num_heads)
            rope_theta = config.pop("rope_theta", 500000)
            qk_rope_head_dim = config.pop("qk_rope_head_dim", 64)
            num_heads = config.get("num_attention_heads", 96)
            hidden_size = config.get("hidden_size", 4096)
            head_dim = hidden_size // num_heads
            partial_rotary_factor = qk_rope_head_dim / head_dim if head_dim > 0 else 0.5
            config["rope_parameters"] = {
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
                "rope_type": "default",
            }
        return result

    _gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Map glm4v_moe_text model_type back to glm4moe for gguf-py tensor name mapping.
    # Also extract num_layers from text_config for composite Glm4vMoeConfig which
    # lacks a top-level num_hidden_layers attribute.
    orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("glm4v_moe_text", "glm4v_moe"):
            model_type = "glm4moe"
            if num_layers is None:
                text_cfg = getattr(hf_model.config, "text_config", None)
                if text_cfg is not None:
                    num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_glm4moe_gguf()


class ModelVariant(StrEnum):
    """Available Unsloth GLM-4.6V GGUF model variants for image-text-to-text tasks."""

    GLM_4_6V_GGUF_Q2_K = "glm_4_6v_gguf_q2_k"


class ModelLoader(ForgeModel):
    """Unsloth GLM-4.6V GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_6V_GGUF_Q2_K: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-4.6V-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_6V_GGUF_Q2_K

    GGUF_FILE = "GLM-4.6V-Q2_K.gguf"

    # Processor is loaded from the original GLM-4.6V repo since the GGUF repo
    # only hosts quantized model weights without processor/tokenizer configs.
    PROCESSOR_MODEL = "zai-org/GLM-4.6V"

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="unsloth_glm_4_6v_gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_full_config(self):
        """Build a full Glm4vMoeConfig combining GGUF text config with vision config from base model."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        base_config = AutoConfig.from_pretrained(self.PROCESSOR_MODEL)
        return Glm4vMoeConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_MODEL, **kwargs)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        config = self._build_full_config()
        model_kwargs["config"] = config

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = self._build_full_config()
        return self.config
