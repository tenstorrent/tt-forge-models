# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher GLM-4.6V GGUF model loader implementation for image-text-to-text tasks.
"""
import threading

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
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

_glm4v_ctx = threading.local()


def _patch_gguf_for_glm4v_moe():
    """
    Patch GGUF loading to handle Glm4vMoeConfig (GLM-4.6V VLM).

    Problems:
    1. load_gguf_checkpoint is called with model_to_load as a keyword arg that
       gets stripped by other loader patches (e.g. glm_4_5_air_derestricted_gguf),
       so get_gguf_hf_weights_map receives hf_model=None.
    2. Glm4vMoeConfig has no top-level num_hidden_layers (it lives in text_config).
    3. model_type "glm4v_moe" is not in gguf-py MODEL_ARCH_NAMES; must be "glm4moe".

    Fix: wrap load_gguf_checkpoint to capture model_to_load before it gets stripped,
    then wrap get_gguf_hf_weights_map to inject num_layers and remap model_type.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if getattr(gguf_utils.load_gguf_checkpoint, "_glm4v_moe_patched", False):
        return

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        # Capture model_to_load before inner patches (e.g. glm_4_5_air) strip it.
        _glm4v_ctx.model_to_load = kwargs.get("model_to_load", None)
        try:
            return orig_load(*args, **kwargs)
        finally:
            _glm4v_ctx.model_to_load = None

    patched_load_gguf_checkpoint._glm4v_moe_patched = True
    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        model = (
            hf_model
            if hf_model is not None
            else getattr(_glm4v_ctx, "model_to_load", None)
        )
        if model is not None:
            cfg = model.config
            # Glm4vMoeConfig stores num_hidden_layers in text_config.
            if (
                num_layers is None
                and not hasattr(cfg, "num_hidden_layers")
                and hasattr(cfg, "text_config")
            ):
                num_layers = cfg.text_config.num_hidden_layers
            # Map glm4v_moe to the gguf-py arch name "glm4moe".
            if model_type is None:
                model_type = cfg.model_type
            if model_type in ("glm4v_moe", "glm4v_moe_text"):
                model_type = "glm4moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    patched_get_gguf_hf_weights_map._glm4v_moe_patched = True
    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_gguf_for_glm4v_moe()


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

        # The GGUF metadata maps to Glm4MoeConfig (glm4_moe arch) which is not
        # recognized by AutoModelForImageTextToText. Load the config from the
        # original vision model repo so the correct Glm4vMoeConfig is used.
        config = AutoConfig.from_pretrained(self.PROCESSOR_MODEL)
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        return self.config
