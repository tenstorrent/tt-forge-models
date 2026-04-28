# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-OCR GGUF model loader implementation for image-to-text tasks.

The GGUF checkpoint only contains the text backbone (glm4 architecture).
We load the vision config from the base (non-GGUF) model and combine
them into a full GlmOcrConfig, following the same pattern as Gemma3 GGUF.
"""
import importlib.metadata

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig,
    GlmOcrConfig,
)
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


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


def _patch_glm_ocr_gguf_support():
    """Patch transformers to support loading GlmOcr from GGUF.

    Two gaps are bridged:

    1. The GLM-OCR GGUF file uses the 'glm4' architecture identifier which
       transformers 5.x does not yet register. We add the architecture and
       its config mapping so load_gguf_checkpoint can parse the file.
       (glm_4_32b_0414_gguf does the same; we duplicate here so this loader
       is self-contained when run in isolation.)

    2. GlmOcrConfig is a composite config (text_config + vision_config) that
       does not expose num_hidden_layers at the top level. The transformers
       get_gguf_hf_weights_map unconditionally accesses config.num_hidden_layers
       and uses model_type to look up the gguf architecture. We wrap
       get_gguf_hf_weights_map to translate model_type='glm_ocr' → 'glm4' and
       to read num_layers from text_config.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if hasattr(gguf_utils, "_glm_ocr_gguf_patched"):
        return
    gguf_utils._glm_ocr_gguf_patched = True

    # Register glm4 architecture if not already done by another loader.
    if "glm4" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4")
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": "head_dim",
            "attention.value_length": None,
            "vocab_size": "vocab_size",
        }
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
        if "glm4" not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["glm4"] = GGUFQwen2Converter

    # Wrap get_gguf_hf_weights_map to handle GlmOcrConfig's composite layout.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "glm_ocr":
            model_type = "glm4"
            if num_layers is None:
                try:
                    num_layers = hf_model.config.text_config.num_hidden_layers
                except AttributeError:
                    pass
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_glm_ocr_gguf_support()


class ModelVariant(StrEnum):
    """Available GLM-OCR GGUF model variants for image-to-text tasks."""

    GLM_OCR_Q8_0 = "glm_ocr_q8_0"


class ModelLoader(ForgeModel):
    """GLM-OCR GGUF model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.GLM_OCR_Q8_0: LLMModelConfig(
            pretrained_model_name="ggml-org/GLM-OCR-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_OCR_Q8_0

    GGUF_FILE = "GLM-OCR-Q8_0.gguf"

    _BASE_PROCESSOR_MODEL = "zai-org/GLM-OCR"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLM-OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_full_config(self):
        """Build a full GlmOcrConfig by wrapping the GGUF text config with a vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        base_config = AutoConfig.from_pretrained(self._BASE_PROCESSOR_MODEL)
        # The GGUF config lacks mrope_section (GLM-specific, not in GGUF spec).
        # Use the base model's rope_parameters which has the correct mrope_section.
        text_config_dict = text_config.to_dict()
        text_config_dict["rope_parameters"] = base_config.text_config.rope_parameters
        return GlmOcrConfig(
            text_config=text_config_dict,
            vision_config=base_config.vision_config.to_dict(),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL, use_fast=False)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        config = self._build_full_config()
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL, use_fast=False)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Text Recognition:"},
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
