# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2 VL GGUF model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    The gguf library supports qwen2vl tensors, but transformers lacks the
    config mapping and architecture registration needed to load qwen2vl GGUF
    checkpoints. Also patches get_gguf_hf_weights_map to handle the composite
    Qwen2VLConfig which stores num_hidden_layers in text_config.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.modeling_utils as modeling_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "rope.dimension_sections": None,
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_vl"
            # mrope_section is required by Qwen2VL attention but absent in GGUF metadata
            config.setdefault(
                "rope_scaling",
                {
                    "type": "mrope",
                    "mrope_section": [16, 24, 24],
                    "rope_theta": config.get("rope_theta", 1000000.0),
                    "rope_type": "default",
                },
            )
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen2_vl" and num_layers is None:
            num_layers = hf_model.config.text_config.num_hidden_layers
            model_type = "qwen2vl"
        return orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    modeling_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen2vl_gguf()


from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen 2 VL GGUF model variants for vision-language tasks."""

    QWEN_2_VL_2B_INSTRUCT_GGUF = "2B_Instruct_GGUF"
    QWEN_2_VL_7B_INSTRUCT_GGUF = "7B_Instruct_GGUF"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.QWEN_2_VL_2B_INSTRUCT_GGUF: "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
    ModelVariant.QWEN_2_VL_7B_INSTRUCT_GGUF: "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
}

# Map variants to the canonical (non-GGUF) repo used for processor/tokenizer
# since GGUF repos do not include those files.
_PROCESSOR_NAMES = {
    ModelVariant.QWEN_2_VL_2B_INSTRUCT_GGUF: "Qwen/Qwen2-VL-2B-Instruct",
    ModelVariant.QWEN_2_VL_7B_INSTRUCT_GGUF: "Qwen/Qwen2-VL-7B-Instruct",
}


class ModelLoader(ForgeModel):
    """Qwen 2 VL GGUF model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_VL_2B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen2-VL-2B-Instruct-GGUF",
        ),
        ModelVariant.QWEN_2_VL_7B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2-VL-7B-Instruct-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_VL_2B_INSTRUCT_GGUF

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Qwen 2-VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        self.processor = AutoProcessor.from_pretrained(
            _PROCESSOR_NAMES[self._variant], **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2 VL GGUF model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Qwen 2 VL GGUF model instance for vision-language tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "gguf_file": _GGUF_FILES[self._variant],
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen 2 VL GGUF model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
