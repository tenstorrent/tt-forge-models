# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    The qwen2vl GGUF architecture (used by Qwen2-VL and Qwen2.5-VL models) is
    not natively supported by transformers' GGUF loader. The text decoder uses
    the same architecture as qwen2, so we map qwen2vl -> qwen2_5_vl for loading.

    Multiple transformers modules hold their own reference to load_gguf_checkpoint
    via module-level imports, so we must patch all of them.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

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

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        # For weight loading, gguf-py needs model_type="qwen2vl" to find the arch.
        # Temporarily remap qwen2_5_vl -> qwen2vl on the model config while calling
        # orig_load, then restore it afterward.
        model_to_load = kwargs.get("model_to_load")
        original_model_type = None
        if model_to_load is not None and hasattr(model_to_load, "config"):
            if getattr(model_to_load.config, "model_type", None) == "qwen2_5_vl":
                original_model_type = "qwen2_5_vl"
                model_to_load.config.model_type = "qwen2vl"
        try:
            result = orig_load(*args, **kwargs)
        finally:
            if original_model_type is not None:
                model_to_load.config.model_type = original_model_type
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_5_vl"
        # GGUF metadata omits rope_scaling/mrope_section; inject defaults for qwen2vl.
        # mrope_section=[16,24,24] is the standard value for Qwen2.5-VL-7B architecture
        # (from GGUF field qwen2vl.rope.dimension_sections = [16, 24, 24, 0]).
        rope_scaling = config.setdefault("rope_scaling", {})
        rope_scaling.setdefault("type", "mrope")
        rope_scaling.setdefault("mrope_section", [16, 24, 24])
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_tokenizers as tok_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint

    from transformers import Qwen2_5_VLConfig

    if not hasattr(Qwen2_5_VLConfig, "num_hidden_layers"):
        Qwen2_5_VLConfig.num_hidden_layers = property(
            lambda self: self.text_config.num_hidden_layers
        )


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
    """Available Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF variants for image to text."""

    JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF = "7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/JSL-VL-7B-MedAgentBench-v2-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF: "JSL-VL-7B-MedAgentBench-v2.i1-Q4_K_M.gguf",
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
            model="Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF",
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
        _patch_transformers_qwen2vl_gguf()
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
