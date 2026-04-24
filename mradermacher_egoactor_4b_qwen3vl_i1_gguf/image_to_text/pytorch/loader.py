# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher EgoActor 4B Qwen3VL i1 GGUF model loader implementation for image to text.
"""
import importlib.metadata

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
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

_QWEN3VL_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.dimension_sections": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "attention.key_length": None,
    "attention.value_length": None,
    "n_deepstack_layers": None,
    "vocab_size": "vocab_size",
}


def _patch_qwen3vl_support():
    """Register qwen3vl architecture so transformers can load the GGUF checkpoint.

    The GGUF file declares architecture as 'qwen3vl' but transformers uses 'qwen3_vl'
    as the model_type. Without this patch load_gguf_checkpoint raises ValueError.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "qwen3vl", _QWEN3VL_CONFIG_MAPPING
    )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle Qwen3VL composite config.

    Qwen3VLConfig stores num_hidden_layers in text_config, not at the top level.
    Also maps model_type 'qwen3_vl' to 'qwen3vl' for the gguf-py library lookup.
    """
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
    if num_layers is None:
        if hasattr(hf_model.config, "num_hidden_layers"):
            num_layers = hf_model.config.num_hidden_layers
        elif hasattr(hf_model.config, "text_config"):
            num_layers = hf_model.config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available mradermacher EgoActor 4B Qwen3VL i1 GGUF model variants for image to text."""

    EGOACTOR_4B_QWEN3VL_I1_GGUF = "4b_qwen3vl_i1_gguf"


class ModelLoader(ForgeModel):
    """mradermacher EgoActor 4B Qwen3VL i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.EGOACTOR_4B_QWEN3VL_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/EgoActor-4b-Qwen3VL-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EGOACTOR_4B_QWEN3VL_I1_GGUF

    GGUF_FILE = "EgoActor-4b-Qwen3VL.i1-Q4_K_M.gguf"

    # Base unquantized model for config and processor (GGUF repo lacks these)
    _BASE_MODEL = "BAAI-Agents/EgoActor-4b-Qwen3VL"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher EgoActor 4B Qwen3VL i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()

        pretrained_model_name = self._variant_config.pretrained_model_name

        # GGUF repo lacks processor and uses a wrong default vision config for 4B;
        # load both from the base unquantized model.
        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL)
        base_config = AutoConfig.from_pretrained(self._BASE_MODEL)
        # Expose num_hidden_layers at the top level so get_gguf_hf_weights_map can
        # find it without descending into text_config.
        base_config.num_hidden_layers = base_config.text_config.num_hidden_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs["config"] = base_config
        model_kwargs |= kwargs

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
