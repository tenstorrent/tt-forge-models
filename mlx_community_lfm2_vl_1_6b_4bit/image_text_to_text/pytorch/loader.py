# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/LFM2-VL-1.6B-4bit model loader implementation for image-text-to-text tasks.

The checkpoint uses MLX affine 4-bit quantization: each Linear weight is stored as
uint32-packed nibbles alongside per-group bfloat16 scales and biases.  We dequantize
to bfloat16 before loading into the HF model.

Two additional loader bugs are fixed:
  1. config.quantization_config has no quant_method → ValueError from transformers 5.x.
     Fix: delete quantization_config before calling from_config.
  2. _tied_weights_keys is a list (old format) → AttributeError in transformers 5.x
     get_expanded_tied_weights_keys.  Fix: patch to None on the remote model class.
"""
import torch
from transformers import AutoProcessor, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.image_utils import load_image
from safetensors import safe_open
from huggingface_hub import hf_hub_download
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


class ModelVariant(StrEnum):
    """Available mlx-community LFM2-VL-1.6B-4bit model variants for image-text-to-text tasks."""

    LFM2_VL_1_6B_4BIT = "1_6B_4bit"


def _dequantize_mlx_affine_4bit(w_packed, scales, biases, group_size=64):
    """Dequantize MLX affine 4-bit weight from uint32 to bfloat16.

    w_packed: [out_f, packed_cols]  uint32  (each uint32 holds 8 nibbles)
    scales:   [out_f, num_groups]   bfloat16
    biases:   [out_f, num_groups]   bfloat16
    Returns:  [out_f, in_f]         bfloat16
    """
    out_f = w_packed.shape[0]
    # uint8 view gives 4 bytes per uint32; each byte holds 2 nibbles
    w_u8 = w_packed.view(torch.uint8)          # [out_f, packed_cols*4]
    lo = (w_u8 & 0x0F).to(torch.float32)
    hi = ((w_u8 >> 4) & 0x0F).to(torch.float32)
    # interleave lo/hi: nibble order matches MLX's row-major packing
    in_f = w_u8.shape[-1] * 2
    w_int4 = torch.stack([lo, hi], dim=-1).reshape(out_f, in_f)  # [out_f, in_f]
    # per-group scale+bias
    scales_exp = scales.float().repeat_interleave(group_size, dim=-1)   # [out_f, in_f]
    biases_exp = biases.float().repeat_interleave(group_size, dim=-1)   # [out_f, in_f]
    return (w_int4 * scales_exp + biases_exp).bfloat16()


def _load_and_dequantize_mlx_checkpoint(pretrained_model_name, group_size=64, bits=4):
    """Download the MLX safetensors shard, dequantize quantized weights, and remap keys.

    Key remapping (MLX checkpoint → HF transformers state_dict):
      language_model.model.X  →  model.language_model.X
      multi_modal_projector.X →  model.multi_modal_projector.X
      vision_tower.X          →  model.vision_tower.vision_model.X

    Additional fixes:
      - Conv1d weights: MLX stores [out, kernel, in]; PyTorch expects [out, in, kernel].
        Detected via ndim==3 and 'conv' in key name.
    """
    shard_path = hf_hub_download(pretrained_model_name, "model.safetensors")

    # Collect quantized bases (keys that have a .scales counterpart)
    quantized_bases = set()
    with safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".scales"):
                quantized_bases.add(key[: -len(".scales")])

    result = {}
    with safe_open(shard_path, framework="pt") as f:
        all_keys = list(f.keys())
        for key in all_keys:
            # Skip .scales and .biases — consumed during dequantization
            if key.endswith(".scales") or key.endswith(".biases"):
                continue

            base_name = key[: -len(".weight")] if key.endswith(".weight") else None
            if base_name is not None and base_name in quantized_bases:
                # Dequantize uint32-packed nibbles
                w_packed = f.get_tensor(key)
                scales = f.get_tensor(base_name + ".scales")
                biases = f.get_tensor(base_name + ".biases")
                value = _dequantize_mlx_affine_4bit(w_packed, scales, biases, group_size)
            else:
                value = f.get_tensor(key)

            # Prefix remapping
            if key.startswith("language_model.model."):
                new_key = "model.language_model." + key[len("language_model.model."):]
            elif key.startswith("vision_tower."):
                new_key = "model.vision_tower.vision_model." + key[len("vision_tower."):]
            elif key.startswith("multi_modal_projector."):
                new_key = "model." + key
            else:
                new_key = key

            # Conv1d axis fix: MLX [out, kernel, in] → PyTorch [out, in, kernel]
            if value.ndim == 3 and "conv" in new_key:
                value = value.permute(0, 2, 1).contiguous()

            result[new_key] = value

    return result


class ModelLoader(ForgeModel):
    """mlx-community LFM2-VL-1.6B-4bit model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_VL_1_6B_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/LFM2-VL-1.6B-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_VL_1_6B_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mlx-community LFM2-VL-1.6B-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        # use_fast=False: the Siglip2 fast image processor triggers a torchvision
        # normalize path that breaks under tt-xla's TorchFunctionMode during preprocessing.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        # Bug 1: quantization_config has no quant_method → ValueError in transformers 5.x.
        # Load config and strip the quantization metadata before constructing the model.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        if hasattr(config, "quantization"):
            del config.quantization

        # Bug 2: _tied_weights_keys is a list (transformers 4.x format); transformers 5.x
        # expects a dict and calls .keys() on it → AttributeError.  Patch to None so
        # get_expanded_tied_weights_keys returns {} and tie_weights() handles it later.
        model_cls = get_class_from_dynamic_module(
            "modeling_lfm2_vl.Lfm2VlForConditionalGeneration",
            pretrained_model_name,
        )
        model_cls._tied_weights_keys = None

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = model_cls(config).to(dtype).eval()

        # Load and dequantize MLX 4-bit weights
        group_size = getattr(getattr(config, "quantization", None), "group_size", None) or 64
        state_dict = _load_and_dequantize_mlx_checkpoint(pretrained_model_name, group_size)
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()

        self.config = config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
