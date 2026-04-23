# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
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

# Text-level config fields extracted from qwen3vl GGUF metadata
_QWEN3VL_TEXT_FIELDS = {
    "num_hidden_layers",
    "max_position_embeddings",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_theta",
    "rms_norm_eps",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
}


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF config loading support.

    Registers the qwen3vl architecture so that AutoConfig.from_pretrained
    succeeds when reading a qwen3vl GGUF file (return_tensors=False only).

    We avoid patching the tensor-loading path because other loaders imported
    after this one overwrite gguf_utils.load_gguf_checkpoint with functions
    that strip model_to_load.  Instead, load_model() calls _load_qwen3vl()
    which handles tensor loading directly via _load_qwen3vl_tensors().
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

    # 2. Add config field mapping for qwen3vl text backbone (GGUF only contains
    #    text model metadata; vision encoder fields are not present in the file)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_sections": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "n_deepstack_layers": None,
        "vocab_size": "vocab_size",
    }

    # 3. Register tokenizer converter — Qwen3-VL uses the same BPE tokenizer
    #    as Qwen2/Qwen3 text models
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    # 4. Patch load_gguf_checkpoint (config path only) to restructure the
    #    flat GGUF config dict into the nested text_config/vision_config layout
    #    that Qwen3VLConfig expects.
    #
    #    We do NOT handle return_tensors=True here: the patch is overwritten by
    #    loaders imported after us, so we bypass that path in load_model().
    orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") != "qwen3vl":
            return result

        config["model_type"] = "qwen3_vl"
        text_config = {"model_type": "qwen3_vl_text"}
        for field in list(_QWEN3VL_TEXT_FIELDS):
            if field in config:
                text_config[field] = config.pop(field)
        config["text_config"] = text_config
        config.setdefault("vision_config", {})
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Also update the reference held by configuration_utils so that
    # AutoConfig.from_pretrained() picks up our restructuring logic
    import transformers.configuration_utils as _config_utils

    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


def _load_qwen3vl(pretrained_model_name, gguf_file_name, gguf_path, dtype_override):
    """Load a Qwen3-VL model from a local GGUF file.

    The qwen3vl GGUF format only contains the quantized text-model weights;
    vision-encoder weights are absent and retain default (random) values.

    We bypass from_pretrained's tensor-loading path because other loaders in
    the test suite overwrite gguf_utils.load_gguf_checkpoint with functions
    that strip model_to_load, breaking the chain for every model.  Instead we:
      1. Load the config via AutoConfig (config-only GGUF read; works fine).
      2. Create the model from that config.
      3. Load text-model tensors directly via gguf-py.
    """
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    # Step 1: Config
    config = AutoConfig.from_pretrained(pretrained_model_name, gguf_file=gguf_file_name)
    # GGUF only embeds text-model metadata; vision_config is defaulted from the 7B model.
    # The vision merger output (out_hidden_size) must match the text hidden_size so that
    # deepstack residual additions are shape-compatible.
    config.vision_config.out_hidden_size = config.text_config.hidden_size

    # Step 2: Model (vision encoder weights are random; text model loaded below)
    torch_dtype = dtype_override if dtype_override is not None else torch.bfloat16
    model = Qwen3VLForConditionalGeneration(config).to(dtype=torch_dtype)

    # Step 3: Build GGUF-name -> HF-state-dict-name mapping.
    #   Pass model_type="qwen3vl" and explicit num_layers to avoid
    #   attribute errors on the top-level Qwen3VLConfig.
    num_layers = config.text_config.num_hidden_layers
    processor_cls = gguf_utils.TENSOR_PROCESSORS.get(
        "qwen3vl", gguf_utils.TensorProcessor
    )
    processor = processor_cls(config={})
    tensor_key_mapping = gguf_utils.get_gguf_hf_weights_map(
        model, processor, model_type="qwen3vl", num_layers=num_layers
    )

    # Step 4: Read, dequantize, and collect tensors from GGUF
    reader = GGUFReader(gguf_path, "r")
    loaded = {}
    state_dict = model.state_dict()
    for tensor in reader.tensors:
        gguf_name = tensor.name
        if gguf_name not in tensor_key_mapping:
            continue
        hf_name = tensor_key_mapping[gguf_name]
        if hf_name not in state_dict:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.copy(weights)).to(dtype=torch_dtype)
        expected_shape = state_dict[hf_name].shape
        # GGUF stores weight matrices transposed relative to PyTorch convention
        if t.ndim >= 2 and t.shape == tuple(reversed(expected_shape)):
            t = t.T
        if t.shape == expected_shape:
            loaded[hf_name] = t

    model.load_state_dict(loaded, strict=False)
    return model


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 32B Thinking GGUF model variants for image to text."""

    QWEN_3_VL_32B_THINKING_1M_GGUF = "32b_thinking_1m_gguf"
    QWEN_3_VL_32B_THINKING_GGUF = "32b_thinking_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-32B-Thinking-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Thinking-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: "Qwen3-VL-32B-Thinking-1M-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: "Qwen3VL-32B-Thinking-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 32B Thinking GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Resolve the local GGUF path so _load_qwen3vl can open it directly
        from huggingface_hub import hf_hub_download

        gguf_path = hf_hub_download(repo_id=pretrained_model_name, filename=gguf_file)

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Thinking")

        model = _load_qwen3vl(
            pretrained_model_name=pretrained_model_name,
            gguf_file_name=gguf_file,
            gguf_path=gguf_path,
            dtype_override=dtype_override,
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
