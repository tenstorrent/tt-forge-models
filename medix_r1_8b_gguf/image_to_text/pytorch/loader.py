# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 8B GGUF model loader implementation for image to text.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF loading
    support for the qwen3vl architecture. We bridge the gap by:
    1. Registering qwen3vl config/tensor mappings.
    2. Adding a TensorProcessor that strips the model.language_model. prefix so
       the standard qwen3vl gguf-py name_map can resolve text-backbone params.
    3. Patching load_gguf_checkpoint to nest flat text params into text_config
       and rename model_type from 'qwen3vl' to 'qwen3_vl'.
    4. Patching get_gguf_hf_weights_map to handle the qwen3_vl model_type and
       fetch num_hidden_layers from text_config.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vl as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # 2. Add config mapping for qwen3vl (text backbone fields identical to qwen3)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # 3. Custom processor: strip model.language_model. so gguf-py name_map resolves
    #    text-backbone HF names (e.g. model.language_model.layers.0.self_attn.q_proj
    #    → model.layers.0.self_attn.q_proj → blk.0.attn_q).
    class Qwen3VLTensorProcessor(TensorProcessor):
        def preprocess_name(self, hf_name: str) -> str:
            return hf_name.replace("model.language_model.", "model.")

    TENSOR_PROCESSORS["qwen3vl"] = Qwen3VLTensorProcessor

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
        GGUF_TO_FAST_CONVERTERS["qwen3_vl"] = GGUF_TO_FAST_CONVERTERS["qwen3"]

    # 5. Patch load_gguf_checkpoint: move flat text params into text_config and
    #    translate model_type from 'qwen3vl' to 'qwen3_vl' so AutoConfig resolves
    #    Qwen3VLConfig correctly.
    orig_load = gguf_utils.load_gguf_checkpoint

    _TEXT_CONFIG_KEYS = [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "vocab_size",
        "tie_word_embeddings",
    ]

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3vl":
            config = result["config"]
            text_config = {}
            for k in _TEXT_CONFIG_KEYS:
                if k in config:
                    text_config[k] = config.pop(k)
            config["text_config"] = text_config
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    modeling_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map: handle qwen3_vl model_type (gguf-py uses
    #    'qwen3vl') and fetch num_hidden_layers from text_config sub-config.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            if getattr(hf_model.config, "model_type", None) == "qwen3_vl":
                model_type = "qwen3vl"
                if num_layers is None:
                    num_layers = hf_model.config.text_config.num_hidden_layers
        return orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available MediX R1 8B GGUF model variants for image to text."""

    MEDIX_R1_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-8B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: "MediX-R1-8B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_8B_Q4_K_M

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
        )

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
