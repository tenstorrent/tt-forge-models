# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/gemma-3n-E2B-it-4bit model loader implementation for multimodal modeling.
"""

from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3nForConditionalGeneration,
)
from PIL import Image

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type, get_file


def _mlx_dequantize(weight, scales, biases, bits=4, group_size=64):
    """Dequantize MLX-format packed-integer weights to float32 then cast to bfloat16.

    MLX 4-bit quantization packs (32//bits) unsigned int values per uint32 element,
    LSB first.  Each group of `group_size` original values shares one scale and bias:
        original = packed_int4 * scale + bias
    """
    n_per_elem = 32 // bits  # 8 for 4-bit
    out_dim, in_packed = weight.shape
    in_dim = in_packed * n_per_elem

    weight_i32 = weight.view(torch.int32)
    mask = (1 << bits) - 1

    unpacked = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for shift in range(n_per_elem):
        unpacked[:, shift::n_per_elem] = ((weight_i32 >> (shift * bits)) & mask).float()

    scales_f = scales.float().repeat_interleave(group_size, dim=1)
    biases_f = biases.float().repeat_interleave(group_size, dim=1)
    return (unpacked * scales_f + biases_f).to(torch.bfloat16)


def _remap_mlx_key(key):
    """Map an MLX safetensors key to the transformers state-dict key.

    All weights live under Gemma3nForConditionalGeneration.model (Gemma3nModel),
    so a 'model.' prefix is always prepended.  The language-model branch adds an
    extra 'model.' layer in the MLX naming ('language_model.model.*') that does
    NOT appear in the transformers state dict ('language_model.*'); strip it.
    """
    # language_model.model.* → language_model.*
    mapped = key.replace("language_model.model.", "language_model.", 1)
    return "model." + mapped


def _load_mlx_state_dict(pretrained_model_name, dtype):
    """Load the safetensors checkpoint and fix MLX format incompatibilities.

    mlx-community safetensors files have three issues relative to PyTorch/transformers:
    1. Keys omit the leading 'model.' prefix (all weights live under Gemma3nModel),
       and the language model submodule has an extra 'model.' infix that must be
       stripped ('language_model.model.*' → 'language_model.*').
    2. Linear/embedding weights flagged in quantization_config are stored as uint32
       packed 4-bit integers with companion '.scales' and '.biases' tensors.
    3. Conv weight tensors use NHWC layout ([out,H,W,in] or [ch,K,1]) instead of
       PyTorch's NCHW ([out,in,H,W] or [ch,1,K]).
    """
    from huggingface_hub import snapshot_download
    import os
    from safetensors import safe_open

    local_dir = snapshot_download(pretrained_model_name)
    sf_path = os.path.join(local_dir, "model.safetensors")

    with safe_open(sf_path, framework="pt") as f:
        all_keys = list(f.keys())

    quantized_bases = {
        k[: -len(".weight")]
        for k in all_keys
        if k.endswith(".weight") and (k[: -len(".weight")] + ".scales") in all_keys
    }

    state_dict = {}
    with safe_open(sf_path, framework="pt") as f:
        for key in all_keys:
            # Skip scale/bias helpers — consumed alongside their weight
            if key.endswith(".scales") or key.endswith(".biases"):
                continue

            tensor = f.get_tensor(key)
            base = key[: -len(".weight")] if key.endswith(".weight") else None

            if base is not None and base in quantized_bases:
                scales = f.get_tensor(base + ".scales")
                biases = f.get_tensor(base + ".biases")
                tensor = _mlx_dequantize(tensor, scales, biases)
            elif tensor.dim() == 4:
                # NHWC → NCHW: [out, H, W, in] → [out, in, H, W]
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            elif tensor.dim() == 3 and tensor.shape[2] == 1:
                # Conv1D NHWC: [ch, K, 1] → [ch, 1, K]
                tensor = tensor.permute(0, 2, 1).contiguous()

            if tensor.is_floating_point():
                tensor = tensor.to(dtype)

            state_dict[_remap_mlx_key(key)] = tensor

    return state_dict


class ModelVariant(StrEnum):
    """Available mlx-community Gemma 3n E2B IT 4-bit multimodal model variants."""

    GEMMA_3N_E2B_IT_MLX_4BIT = "E2B_IT_MLX_4bit"


class ModelLoader(ForgeModel):
    """mlx-community Gemma 3n E2B IT 4-bit model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3N_E2B_IT_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/gemma-3n-E2B-it-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3N_E2B_IT_MLX_4BIT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="mlx-community Gemma 3n E2B IT 4-bit Multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the mlx-community Gemma 3n E2B IT 4-bit multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gemma 3n model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Strip the MLX-native quantization_config (no quant_method key) so
        # transformers does not try to dispatch to an unsupported quantizer.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        # Initialise architecture on CPU with no pretrained weights, then load the
        # checkpoint manually to fix MLX format issues (packed-int4, NHWC layout).
        with torch.device("cpu"):
            model = Gemma3nForConditionalGeneration(config)
        model = model.to(dtype)

        state_dict = _load_mlx_state_dict(pretrained_model_name, dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Only tied-embedding entries (lm_head.weight) and any newly-added
        # architecture keys should be missing; anything else is a real gap.
        non_tied_missing = [k for k in missing if "lm_head" not in k]
        if non_tied_missing:
            import warnings
            warnings.warn(f"MLX checkpoint missing keys (non-lm_head): {non_tied_missing[:5]}")

        model.tie_weights()
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the mlx-community Gemma 3n E2B IT 4-bit multimodal model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
