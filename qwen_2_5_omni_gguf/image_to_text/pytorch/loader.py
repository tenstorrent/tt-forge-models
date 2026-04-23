# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 Omni GGUF model loader implementation for image to text.
"""
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    Qwen2.5-Omni GGUF files use the qwen2vl architecture identifier in their
    metadata, which is not in transformers' supported GGUF architectures list.
    The GGUF contains only the LLM backbone tensors (no vision/audio weights),
    so we map qwen2vl to qwen2_vl which has a compatible text backbone.

    Called both at module import and at load_model() time so that later-imported
    loaders that replace gguf_utils.load_gguf_checkpoint with an incompatible
    signature (no model_to_load arg) don't break our model loading.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
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

    # Skip if our patch is already the active one (avoid redundant wrapping).
    if getattr(gguf_utils.load_gguf_checkpoint, "_qwen2vl_omni_patch", False):
        return

    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        # Some loaders in the chain may not accept model_to_load; fall back gracefully.
        try:
            result = _orig_load_gguf_checkpoint(
                gguf_checkpoint_path,
                return_tensors=return_tensors,
                model_to_load=model_to_load,
            )
        except TypeError:
            result = _orig_load_gguf_checkpoint(
                gguf_checkpoint_path,
                return_tensors=return_tensors,
            )
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_vl"
        return result

    _patched_load_gguf_checkpoint._qwen2vl_omni_patch = True

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly at module level.
    import transformers.configuration_utils as _config_utils

    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    try:
        import transformers.models.auto.tokenization_auto as _tok_auto

        _tok_auto.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    except (ImportError, AttributeError):
        pass

    try:
        import transformers.tokenization_utils_tokenizers as _tok_utils

        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    except (ImportError, AttributeError):
        pass


_patch_transformers_qwen2vl_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 2.5 Omni GGUF model variants for image to text."""

    QWEN_2_5_OMNI_7B_Q4_K_M = "7b_q4_k_m"


class ModelLoader(ForgeModel):
    """Qwen 2.5 Omni GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="ggml-org/Qwen2.5-Omni-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M: "Qwen2.5-Omni-7B-Q4_K_M.gguf",
    }

    # Processor is loaded from the original Qwen2.5-VL repo since the GGUF
    # repo only contains quantized model weights without tokenizer/processor configs.
    _PROCESSOR_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 2.5 Omni GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Re-apply to ensure our patch is active if another loader replaced it.
        _patch_transformers_qwen2vl_gguf()

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
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_SOURCE)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
