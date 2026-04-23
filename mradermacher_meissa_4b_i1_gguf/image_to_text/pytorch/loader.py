# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/Meissa-4B-i1-GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
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
    """Register qwen3vl GGUF architecture support in transformers.

    The Qwen3-VL GGUF files use 'qwen3vl' as the architecture identifier, but
    transformers 5.x only registers 'qwen3' for GGUF loading. We bridge the gap
    by reusing the qwen3 config/tokenizer mappings and remapping model_type to
    qwen3_vl after parsing so Qwen3VLForConditionalGeneration can be initialised.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vl as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # 2. Add config mapping reusing qwen3 field names
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
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

    # 3. Register qwen3vl tokenizer converter
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    # 4. Patch get_gguf_hf_weights_map to handle composite Qwen3VLConfig and
    #    map qwen3_vl model_type to the qwen3vl gguf arch name.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor=None, model_type=None, num_layers=None, qual_name=None
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
        if num_layers is None:
            cfg = hf_model.config
            if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
                num_layers = cfg.text_config.num_hidden_layers
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # 5. Patch load_gguf_checkpoint to remap model_type qwen3 -> qwen3_vl
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") in ("qwen3", "qwen3vl"):
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available mradermacher Meissa-4B-i1-GGUF model variants for image to text."""

    MEISSA_4B_I1_GGUF = "4b_i1_gguf"


class ModelLoader(ForgeModel):
    """mradermacher Meissa-4B-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEISSA_4B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Meissa-4B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEISSA_4B_I1_GGUF

    GGUF_FILE = "Meissa-4B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher Meissa-4B-i1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import importlib.metadata

        import transformers.utils.import_utils as _transformers_import_utils

        # gguf is installed at runtime; refresh the static mapping so
        # transformers' is_gguf_available() can look up the version.
        if "gguf" not in _transformers_import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _transformers_import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = [
                    "gguf"
                ]
            except importlib.metadata.PackageNotFoundError:
                pass

        _patch_transformers_qwen3vl_gguf()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        # Pass explicit config so random_weights AutoConfig skips GGUF parsing
        # (qwen3vl architecture is not yet registered in transformers GGUF support).
        config = Qwen3VLConfig()
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
