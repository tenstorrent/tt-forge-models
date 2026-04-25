# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL GGUF model loader implementation for image to text.
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
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

_TEXT_CONFIG_FIELDS = {
    "max_position_embeddings",
    "num_hidden_layers",
    "intermediate_size",
    "hidden_size",
    "rope_theta",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "vocab_size",
}


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    The Qwen2.5-VL GGUF file uses the 'qwen2vl' architecture identifier, which
    transformers does not support yet. We bridge the gap by registering the
    config/tokenizer mappings and remapping model_type to 'qwen2_5_vl'.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            text_config = {
                k: config.pop(k)
                for k in list(config.keys())
                if k in _TEXT_CONFIG_FIELDS
            }
            config["text_config"] = text_config
            # Vision merger must output text_hidden_size for image token replacement.
            config["vision_config"] = {
                "out_hidden_size": text_config.get("hidden_size", 4096)
            }
            config["model_type"] = "qwen2_5_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_type = (
            model_type
            if model_type is not None
            else getattr(getattr(hf_model, "config", None), "model_type", None)
        )
        if effective_type == "qwen2_5_vl":
            model_type = "qwen2vl"
            if num_layers is None:
                text_cfg = getattr(
                    getattr(hf_model, "config", None), "text_config", None
                )
                if text_cfg is not None:
                    num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # Patch modules that imported load_gguf_checkpoint via from-import
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_qwen2vl_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 2.5 VL GGUF model variants for image to text."""

    QWEN_2_5_VL_72B_INSTRUCT_GGUF = "72b_instruct_gguf"
    BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF = "bartowski_72b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen_Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen_Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
    }

    _BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 2.5 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 VL GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 2.5 VL GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
            )
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # GGUF repos do not ship a processor; use the base Qwen2.5-VL model.
        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL_NAME)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 2.5 VL GGUF model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
