# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Qwen2.5-VL 3B Instruct GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

from transformers import (
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_qwen2vl_support():
    """Register qwen2vl GGUF architecture as a supported architecture for Qwen2.5-VL.

    Transformers 5.x does not natively support loading qwen2vl GGUF checkpoints.
    The GGUF file uses 'qwen2vl' as the architecture name, but transformers only
    knows 'qwen2'. We alias the config key mapping and fix the tensor name lookup
    so from_pretrained can load the text-model weights from the GGUF file.

    We also add num_hidden_layers as a property on Qwen2_5_VLConfig since the
    get_gguf_hf_weights_map function accesses it unconditionally, but the VL config
    only exposes it via text_config.
    """
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen2vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"],
            )
    if not hasattr(Qwen2_5_VLConfig, "num_hidden_layers"):
        Qwen2_5_VLConfig.num_hidden_layers = property(
            lambda self: self.text_config.num_hidden_layers
        )


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle qwen2_5_vl model type.

    Qwen2_5_VLConfig stores num_hidden_layers under text_config (not top-level),
    and the model_type 'qwen2_5_vl' has no matching entry in gguf.MODEL_ARCH_NAMES.
    We remap it to 'qwen2vl' and supply num_layers from text_config.
    """
    if model_type is None and hasattr(hf_model, "config"):
        cfg = hf_model.config
        if getattr(cfg, "model_type", None) == "qwen2_5_vl":
            model_type = "qwen2vl"
            if num_layers is None:
                text_cfg = getattr(cfg, "text_config", cfg)
                num_layers = getattr(text_cfg, "num_hidden_layers", None)
    elif model_type == "qwen2_5_vl":
        model_type = "qwen2vl"
        if num_layers is None and hasattr(hf_model, "config"):
            text_cfg = getattr(hf_model.config, "text_config", hf_model.config)
            num_layers = getattr(text_cfg, "num_hidden_layers", None)

    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen2vl architecture support."""
    _patch_qwen2vl_support()
    return _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )


_patch_qwen2vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Mradermacher Qwen2.5-VL 3B Instruct GGUF variants for image to text."""

    QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF = "3B_INSTRUCT_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher Qwen2.5-VL 3B Instruct GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen2.5-VL-3B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF: "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher Qwen2.5-VL 3B Instruct GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )

        # The GGUF repo has no config.json, so from_pretrained would fall back to
        # Qwen2_5_VLConfig defaults which have wrong vision dimensions (out_hidden_size
        # 3584 vs the correct 2048 for 3B). Load the correct config from the base model.
        config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        model_kwargs.setdefault("config", config)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
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
        return inputs
