# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LightOnOCR GGUF model loader implementation for image-to-text OCR tasks.

The GGUF checkpoint (mradermacher/LightOnOCR-2-1B-GGUF) only contains the text
backbone declared as qwen3 architecture.  We load it as Qwen3ForCausalLM, then
transplant those weights into the language_model sub-module of a freshly
constructed LightOnOcrForConditionalGeneration.  The vision encoder is randomly
initialised because the mmproj weights are not shipped in this GGUF repo.
"""
import importlib.metadata

import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    LightOnOcrConfig,
    LightOnOcrForConditionalGeneration,
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


class ModelVariant(StrEnum):
    """Available LightOnOCR GGUF model variants for image-to-text tasks."""

    LIGHTON_OCR_2_1B_Q8_0 = "lighton_ocr_2_1b_q8_0"


class ModelLoader(ForgeModel):
    """LightOnOCR GGUF model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LIGHTON_OCR_2_1B_Q8_0: LLMModelConfig(
            pretrained_model_name="mradermacher/LightOnOCR-2-1B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIGHTON_OCR_2_1B_Q8_0

    GGUF_FILE = "LightOnOCR-2-1B.Q8_0.gguf"

    _BASE_PROCESSOR_MODEL = "lightonai/LightOnOCR-2-1B"

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
            model="LightOnOCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_full_config(self):
        """Build a LightOnOcrConfig from the GGUF text config + base model vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        base_config = AutoConfig.from_pretrained(self._BASE_PROCESSOR_MODEL)
        config = LightOnOcrConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL)

        # Load text backbone as Qwen3ForCausalLM — the GGUF declares general.architecture=qwen3,
        # so this path is natively supported.  AutoModelForImageTextToText cannot be used here
        # because the GGUF repo has no config.json, making the repo config resolve to Qwen3Config
        # which AutoModelForImageTextToText does not recognise.
        qwen3_kwargs = {}
        if dtype_override is not None:
            qwen3_kwargs["torch_dtype"] = dtype_override
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE, **qwen3_kwargs
        )

        # Build a full LightOnOcrConfig (Qwen3 text + Pixtral vision from base model).
        config = self._build_full_config()
        config.use_cache = False  # avoid KV tensors polluting the PCC comparison

        # Construct the full multimodal model with randomly initialised weights, then
        # transplant the GGUF text weights into the language_model sub-module.
        model = LightOnOcrForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        # qwen3_model.model == Qwen3Model; LightOnOCR exposes it as model.language_model
        model.model.language_model.load_state_dict(
            qwen3_model.model.state_dict(), strict=True
        )
        model.tie_weights()
        model.eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
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
