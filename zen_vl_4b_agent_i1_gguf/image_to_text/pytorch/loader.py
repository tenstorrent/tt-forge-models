# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zen VL 4B Agent i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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


def _patch_qwen3vl_support():
    """Register qwen3vl architecture as an alias for qwen3.

    Qwen3-VL uses the same text backbone as Qwen3 but the GGUF file
    declares architecture as 'qwen3vl', which transformers 5.x does not yet
    recognise as a supported GGUF architecture.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_vl -> qwen3vl for gguf-py lookup.

    Also resolves num_hidden_layers from text_config for composite VL configs
    and corrects the output_norm.weight mapping to point at the language model
    norm rather than the vision merger norm (which is registered first in
    Qwen3VLModel.__init__ and incorrectly wins the key).
    """
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
        if num_layers is None and hasattr(hf_model.config, "text_config"):
            num_layers = hf_model.config.text_config.num_hidden_layers
    result = _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )
    # Fix: output_norm.weight belongs to the language model norm, not the
    # vision merger norm (visual is registered before language_model, so it
    # incorrectly wins the mapping when both have a 'norm' sub-layer).
    for gguf_key in ("output_norm.weight", "output_norm.bias"):
        if gguf_key in result and "visual" in result[gguf_key]:
            lm_key = result[gguf_key].replace(
                "model.visual.merger.norm", "model.language_model.norm"
            )
            result[gguf_key] = lm_key
    return result


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Zen VL 4B Agent i1 GGUF model variants for image to text."""

    ZEN_VL_4B_AGENT_I1_GGUF = "4b_agent_i1_gguf"


class ModelLoader(ForgeModel):
    """Zen VL 4B Agent i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ZEN_VL_4B_AGENT_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/zen-vl-4b-agent-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZEN_VL_4B_AGENT_I1_GGUF

    GGUF_FILE = "zen-vl-4b-agent.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Zen VL 4B Agent i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # Base model provides processor and config; GGUF repo only has quantized weights.
    BASE_MODEL = "zenlm/zen-vl-4b-agent"

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Qwen3VLConfig

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

        # Load config from base model so tensor shapes match the GGUF weights.
        # The GGUF repo has no config.json and qwen3vl is not yet in
        # GGUF_SUPPORTED_ARCHITECTURES, so we must provide the config explicitly.
        config = Qwen3VLConfig.from_pretrained(self.BASE_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
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
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
