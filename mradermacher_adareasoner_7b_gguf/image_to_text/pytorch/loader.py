# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher AdaReasoner 7B GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add GGUF support for the qwen2vl architecture.

    Transformers 5.x has Qwen2_5_VLForConditionalGeneration but lacks GGUF loading
    support for the 'qwen2vl' architecture identifier used in GGUF metadata.
    We bridge the gap by:

    1. Registering config and tensor processor mappings for 'qwen2vl'.
    2. Wrapping load_gguf_checkpoint to nest flat config fields under 'text_config'
       so that Qwen2_5_VLConfig propagates them to Qwen2_5_VLTextConfig instead of
       silently using defaults (e.g. num_hidden_layers defaulting to 24 instead of 28).
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils

    from transformers.modeling_gguf_pytorch_utils import (
        load_gguf_checkpoint as _orig_load_gguf_checkpoint,
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_CONFIG_MAPPING,
    )
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # ------------------------------------------------------------------ #
    # 1. Config field mapping                                              #
    # ------------------------------------------------------------------ #
    GGUF_CONFIG_MAPPING["qwen2vl"] = dict(GGUF_CONFIG_MAPPING["qwen2"])
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    GGUF_CONFIG_DEFAULTS_MAPPING["qwen2vl"] = dict(
        GGUF_CONFIG_DEFAULTS_MAPPING.get("qwen2", {})
    )

    # ------------------------------------------------------------------ #
    # 2. load_gguf_checkpoint wrapper                                     #
    # ------------------------------------------------------------------ #
    # Qwen2_5_VLConfig is composite: its __init__ only forwards an explicit
    # `text_config` dict to Qwen2_5_VLTextConfig; it does NOT propagate
    # top-level kwargs like num_hidden_layers.  Without this wrapper the
    # text model is built with wrong defaults.  Fields are kept at the
    # top level too so that get_gguf_hf_weights_map can access
    # num_hidden_layers via hf_model.config.num_hidden_layers.
    _OUTER_ONLY = frozenset(
        {
            "model_type",
            "architectures",
            "tie_word_embeddings",
            "_model_name_or_path",
            "bos_token_id",
            "eos_token_id",
            "tokenizer_class",
        }
    )

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "qwen2vl":
            text_cfg = {k: v for k, v in cfg.items() if k not in _OUTER_ONLY}
            cfg["text_config"] = text_cfg
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_qwen2vl_gguf()

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


class ModelVariant(StrEnum):
    """Available Mradermacher AdaReasoner 7B GGUF variants for image to text."""

    ADAREASONER_7B_Q4_K_M_GGUF = "7B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher AdaReasoner 7B GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ADAREASONER_7B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/AdaReasoner-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ADAREASONER_7B_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.ADAREASONER_7B_Q4_K_M_GGUF: "AdaReasoner-7B.Q4_K_M.gguf",
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
            model="Mradermacher AdaReasoner 7B GGUF",
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
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )

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
