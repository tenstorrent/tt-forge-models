# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAI-UI-8B i1 GGUF model loader implementation for image to text.
"""
import importlib.metadata

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
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


def _patch_transformers_qwen3vl_gguf():
    """Register qwen3vl as a supported GGUF architecture in transformers.

    Transformers 5.x supports qwen3 but not qwen3vl. We add the config
    mapping (identical to qwen3) and remap model_type to qwen3_vl so
    AutoConfig resolves to Qwen3VLConfig.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen3"].copy()

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            text_hidden = config.get("hidden_size")
            if text_hidden is not None:
                config.setdefault("vision_config", {})
                config["vision_config"]["out_hidden_size"] = text_hidden
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available MAI-UI-8B i1 GGUF model variants for image to text."""

    MAI_UI_8B_I1_GGUF = "mai_ui_8b_i1_gguf"


class ModelLoader(ForgeModel):
    """MAI-UI-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MAI_UI_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/MAI-UI-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAI_UI_8B_I1_GGUF

    GGUF_FILE = "MAI-UI-8B.i1-Q4_K_M.gguf"

    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MAI-UI-8B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_gguf_version_detection():
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Tongyi-MAI/MAI-UI-8B",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()

        return full_model.model.language_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("Tongyi-MAI/MAI-UI-8B")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
