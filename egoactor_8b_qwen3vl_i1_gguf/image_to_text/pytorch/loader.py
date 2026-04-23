# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EgoActor 8B Qwen3VL i1 GGUF model loader implementation for image to text.
"""
import importlib.metadata

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import (
    AutoConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
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


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle qwen3_vl -> qwen3vl mapping.

    transformers uses 'qwen3_vl' as model_type but gguf-py uses 'qwen3vl'.
    Also handles composite VL configs that lack num_hidden_layers at the top level.
    """
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
    if num_layers is None:
        cfg = hf_model.config
        if hasattr(cfg, "num_hidden_layers"):
            num_layers = cfg.num_hidden_layers
        elif hasattr(cfg, "text_config") and hasattr(
            cfg.text_config, "num_hidden_layers"
        ):
            num_layers = cfg.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl as a supported GGUF architecture and patch the weights map lookup."""
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vl_gguf_support()


class ModelVariant(StrEnum):
    """Available EgoActor 8B Qwen3VL i1 GGUF model variants for image to text."""

    EGOACTOR_8B_QWEN3VL_I1_GGUF = "8b_qwen3vl_i1_gguf"


class ModelLoader(ForgeModel):
    """EgoActor 8B Qwen3VL i1 GGUF model loader implementation for image to text tasks."""

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

    _VARIANTS = {
        ModelVariant.EGOACTOR_8B_QWEN3VL_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/EgoActor-8b-Qwen3VL-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EGOACTOR_8B_QWEN3VL_I1_GGUF

    GGUF_FILE = "EgoActor-8b-Qwen3VL.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EgoActor 8B Qwen3VL i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        _patch_qwen3vl_gguf_support()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "BAAI-Agents/EgoActor-8b-Qwen3VL"
        )

        # Load config from base HF repo; GGUF config mapping doesn't support
        # qwen3vl yet, so we need the actual architecture dimensions up front.
        config = AutoConfig.from_pretrained("BAAI-Agents/EgoActor-8b-Qwen3VL")
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
