# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 2B Instruct 1M GGUF model loader implementation for image to text.
"""

from transformers import (
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

# GGUF text-config fields produced by the qwen3 key mapping
_QWEN3VL_TEXT_CONFIG_FIELDS = {
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


def _register_qwen3vl_gguf():
    """Register qwen3vl in the GGUF architecture tables.

    Transformers 5.2.0 knows about qwen3 (text-only) but not qwen3vl (the
    vision-language variant).  This function registers the architecture so
    that load_gguf_checkpoint does not reject the checkpoint outright.
    Called once at module import time.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    if "qwen3vl" not in GGUF_TO_TRANSFORMERS_MAPPING["config"]:
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

    # Qwen3VL reuses the Qwen2/Qwen3 BPE tokenizer
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter


_register_qwen3vl_gguf()


def _make_qwen3vl_load_patch(inner_load):
    """Return a patched load_gguf_checkpoint that handles the qwen3vl case.

    The patch is built dynamically at load_model() time so it is always the
    outermost wrapper – regardless of which other loaders' patches were
    installed during test collection.

    inner_load: the current value of gguf_utils.load_gguf_checkpoint (the
                chain of previously installed patches).
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    from gguf import GGUFReader, dequantize
    from transformers.modeling_gguf_pytorch_utils import (
        TensorProcessor,
        get_gguf_hf_weights_map,
    )

    def patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        # Always get config+tokenizer via the inner chain (no model_to_load,
        # because the other patches in the chain do not support that kwarg).
        result = inner_load(gguf_checkpoint_path, return_tensors=False)

        config = result.get("config", {})
        if config.get("model_type") != "qwen3vl":
            # Non-qwen3vl: replicate the pre-existing chain semantics.
            if return_tensors:
                result = inner_load(gguf_checkpoint_path, return_tensors=True)
            return result

        # ── Fix config for Qwen3VLConfig compatibility ───────────────────────
        config["model_type"] = "qwen3_vl"
        text_config = {}
        for field in list(config.keys()):
            if field in _QWEN3VL_TEXT_CONFIG_FIELDS:
                text_config[field] = config.pop(field)
        if text_config:
            config["text_config"] = text_config

        if not return_tensors or model_to_load is None:
            return result

        # ── Tensor loading for qwen3vl ────────────────────────────────────────
        reader = GGUFReader(gguf_checkpoint_path)
        processor = TensorProcessor(config=config)
        num_layers = model_to_load.config.text_config.num_hidden_layers
        tensor_key_mapping = get_gguf_hf_weights_map(
            model_to_load,
            processor,
            model_type="qwen3vl",
            num_layers=num_layers,
        )

        result["tensors"] = {}
        for tensor in tqdm(reader.tensors, desc="Converting qwen3vl GGUF tensors..."):
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)
            tensor_result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=result,
            )
            weights = tensor_result.weights
            name = tensor_result.name
            if name not in tensor_key_mapping:
                continue
            name = tensor_key_mapping[name]
            result["tensors"][name] = torch.from_numpy(np.copy(weights))

        return result

    return patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 2B Instruct 1M GGUF model variants for image to text."""

    QWEN_3_VL_2B_INSTRUCT_1M_GGUF = "2b_instruct_1m_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 2B Instruct 1M GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_2B_INSTRUCT_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-2B-Instruct-1M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_2B_INSTRUCT_1M_GGUF

    GGUF_FILE = "Qwen3-VL-2B-Instruct-1M-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 2B Instruct 1M GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        # Install our patch as the outermost wrapper right before from_pretrained
        # so that model_to_load is handled correctly regardless of import order.
        saved_load = gguf_utils.load_gguf_checkpoint
        gguf_utils.load_gguf_checkpoint = _make_qwen3vl_load_patch(saved_load)
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            gguf_utils.load_gguf_checkpoint = saved_load

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
