# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher GLM-4.6V GGUF model loader implementation for image-text-to-text tasks.
"""
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.modeling_utils as _modeling_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
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
from ....tools.utils import get_file
from PIL import Image


def _patch_transformers_glm4v_moe_gguf():
    """Register glm4v_moe GGUF architecture support.

    GLM-4.6V GGUF files declare architecture as 'glm4v_moe' but gguf-py 0.18
    only knows glm4moe (text-only). The language backbone tensor layout is
    identical to glm4moe; we add the missing alias so get_gguf_hf_weights_map
    can resolve the architecture and build the tensor-name mapping.
    """
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm4v_moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4v_moe")

    config_section = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]
    if "glm4v_moe" not in config_section and "glm4moe" in config_section:
        config_section["glm4v_moe"] = config_section["glm4moe"]

    for key in ("glm4v_moe", "glm4v_moe_text"):
        GGUF_TO_FAST_CONVERTERS.setdefault(key, GGUFQwen2Converter)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_transformers_glm4v_moe_gguf()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Redirect glm4v_moe → glm4moe and use text_config layer count."""
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type in ("glm4v_moe", "glm4v_moe_text"):
        model_type = "glm4moe"
        if num_layers is None:
            cfg = hf_model.config
            text_cfg = getattr(cfg, "text_config", cfg)
            num_layers = text_cfg.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_transformers_glm4v_moe_gguf()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
for _mod in (_config_utils, _auto_tokenizer, _modeling_utils):
    if hasattr(_mod, "load_gguf_checkpoint"):
        _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available mradermacher GLM-4.6V GGUF model variants for image-text-to-text tasks."""

    GLM_4_6V_GGUF_Q2_K = "glm_4_6v_gguf_q2_k"


class ModelLoader(ForgeModel):
    """mradermacher GLM-4.6V GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_6V_GGUF_Q2_K: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.6V-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_6V_GGUF_Q2_K

    GGUF_FILE = "GLM-4.6V.Q2_K.gguf"

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
            model="mradermacher_glm_4_6v_gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
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

        # The GGUF metadata reports model type as glm4_moe (text-only), but
        # AutoModelForImageTextToText requires Glm4vMoeConfig. Load config from
        # the original multimodal repo to get the correct architecture.
        model_kwargs["config"] = AutoConfig.from_pretrained(self.PROCESSOR_MODEL)

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
        self.config = AutoConfig.from_pretrained(self.PROCESSOR_MODEL)
        return self.config
