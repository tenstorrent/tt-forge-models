# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GUI-Libra 4B i1 GGUF model loader implementation for image to text.
"""

import numpy as np
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
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


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture and patch get_gguf_hf_weights_map.

    Qwen3-VL GGUF files contain only text model weights.  The vision encoder
    keeps its randomly-initialised defaults.  Text model parameters share the
    same GGUF naming convention as qwen3.
    """
    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # Register config field mapping (same as qwen3 text backbone)
    GGUF_CONFIG_MAPPING["qwen3vl"] = dict(GGUF_CONFIG_MAPPING["qwen3"])
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "qwen3vl"
            ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]

    # Patch get_gguf_hf_weights_map to translate qwen3_vl (HF) → qwen3vl (gguf-py).
    # This must be a module-level assignment so that internal recursive calls inside
    # the original function also use the patched version.
    _orig_get_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
        return _orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_weights_map


_patch_qwen3vl_gguf_support()


def _load_qwen3vl_gguf_weights(model, gguf_path, dtype=None):
    """Load GGUF text-model weights into a Qwen3VL model using gguf-py directly.

    Bypasses transformers' load_gguf_checkpoint to avoid compatibility issues
    with other loaders that monkey-patch that function without forwarding the
    model_to_load argument required by transformers >= 5.x.
    """
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from transformers.modeling_gguf_pytorch_utils import TensorProcessor

    reader = GGUFReader(gguf_path)
    processor = TensorProcessor()

    # Our patched get_gguf_hf_weights_map handles the qwen3_vl → qwen3vl translation
    tensor_key_mapping = _gguf_utils.get_gguf_hf_weights_map(model, processor)

    target_dtype = dtype if dtype is not None else torch.bfloat16
    tensors_to_load = {}

    for tensor in reader.tensors:
        if tensor.name not in tensor_key_mapping:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        hf_name = tensor_key_mapping[tensor.name]
        tensors_to_load[hf_name] = torch.from_numpy(np.copy(weights)).to(target_dtype)

    model.load_state_dict(tensors_to_load, strict=False)


class ModelVariant(StrEnum):
    """Available GUI-Libra 4B i1 GGUF model variants for image to text."""

    GUI_LIBRA_4B_I1_GGUF = "4B_i1_GGUF"


class ModelLoader(ForgeModel):
    """GUI-Libra 4B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.GUI_LIBRA_4B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GUI-Libra-4B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GUI_LIBRA_4B_I1_GGUF

    GGUF_FILE = "GUI-Libra-4B.i1-Q4_K_M.gguf"

    # Base model provides the full config (text + vision) and processor
    BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GUI-Libra 4B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Download only the GGUF quantised weights file to the local HF cache
        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )

        # Load full config (text + vision) from the base model.  The GGUF file
        # only contains text model weights so vision encoder params come from here.
        config = AutoConfig.from_pretrained(self.BASE_MODEL)

        # Initialise model with the config; weights start randomly initialised
        model = Qwen3VLForConditionalGeneration(config).to(dtype)

        # Replace text-model weights with the GGUF quantised + dequantised values
        _load_qwen3vl_gguf_weights(model, gguf_path, dtype=dtype)

        model.eval()

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

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
