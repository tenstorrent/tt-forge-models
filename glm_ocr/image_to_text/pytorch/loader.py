# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-OCR model loader implementation for image-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
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


def _remap_mlx_keys(key):
    """Remap mlx-community/GLM-OCR-8bit key names to transformers GlmOcrForConditionalGeneration names.

    The mlx-community checkpoint uses an older naming convention:
      language_model.lm_head.X  ->  lm_head.X
      language_model.model.X    ->  model.language_model.X
      vision_tower.X            ->  model.visual.X
    """
    if key.startswith("language_model.lm_head."):
        return key[len("language_model."):]
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model."):]
    if key.startswith("vision_tower."):
        return "model.visual." + key[len("vision_tower."):]
    return key


def _permute_mlx_conv_weight(key, tensor):
    """MLX stores conv weights channel-last; permute to PyTorch channel-first."""
    if not key.endswith(".weight"):
        return tensor
    if tensor.ndim == 5:
        # Conv3d: [out, D, H, W, in] -> [out, in, D, H, W]
        return tensor.permute(0, 4, 1, 2, 3).contiguous()
    if tensor.ndim == 4:
        # Conv2d: [out, H, W, in] -> [out, in, H, W]
        return tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def _dequantize_mlx_affine_8bit(raw_sd, group_size=64):
    """Dequantize and remap MLX affine-8bit state dict to standard float tensors.

    mlx-community models store weights as uint32-packed int8 with per-group
    bf16 scales and biases.  Transformers expects float weights with standard
    key names, so we unpack, dequant, and remap here.

    Dequant formula: x_float = x_uint8 * scale + bias
    """
    skip = {k for k in raw_sd if k.endswith(".scales") or k.endswith(".biases")}
    result = {}
    for key, tensor in raw_sd.items():
        if key in skip:
            continue
        scales_key = key[: -len(".weight")] + ".scales"
        biases_key = key[: -len(".weight")] + ".biases"
        if (
            key.endswith(".weight")
            and tensor.dtype == torch.uint32
            and scales_key in raw_sd
            and biases_key in raw_sd
        ):
            scales = raw_sd[scales_key]
            biases = raw_sd[biases_key]
            out_f = tensor.shape[0]
            w_u8 = tensor.view(torch.uint8).reshape(out_f, -1)
            in_f = w_u8.shape[1]
            n_grp = in_f // group_size
            sc = scales.float().reshape(out_f, n_grp, 1).expand(-1, -1, group_size).reshape(out_f, in_f)
            bi = biases.float().reshape(out_f, n_grp, 1).expand(-1, -1, group_size).reshape(out_f, in_f)
            result[_remap_mlx_keys(key)] = (w_u8.float() * sc + bi).to(torch.bfloat16)
        else:
            remapped = _remap_mlx_keys(key)
            result[remapped] = _permute_mlx_conv_weight(remapped, tensor)
    return result


class ModelVariant(StrEnum):
    """Available GLM-OCR model variants for image-to-text tasks."""

    GLM_OCR = "glm_ocr"
    GLM_OCR_MLX_8BIT = "mlx_8bit"


class ModelLoader(ForgeModel):
    """GLM-OCR model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GLM_OCR: LLMModelConfig(
            pretrained_model_name="zai-org/GLM-OCR",
        ),
        ModelVariant.GLM_OCR_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/GLM-OCR-8bit",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GLM_OCR

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
            model="glm_ocr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load Processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GLM-OCR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GLM-OCR model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model_kwargs = {"dtype": dtype}
        model_kwargs |= kwargs

        if self._variant == ModelVariant.GLM_OCR_MLX_8BIT:
            # mlx-community/GLM-OCR-8bit has quantization_config with MLX
            # affine format (group_size/bits/mode) but no quant_method.
            # Transformers >=5.x raises ValueError on this.  Also, the weights
            # are uint32-packed int8 that need manual dequantization before
            # loading into the standard GlmOcrForConditionalGeneration arch.
            # Finally, transformers 5.x forbids passing state_dict with a
            # model name; we must create from config and load separately.
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download

            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "quantization_config"):
                del config.quantization_config

            st_path = hf_hub_download(pretrained_model_name, "model.safetensors")
            raw_sd = load_file(st_path)
            state_dict = _dequantize_mlx_affine_8bit(raw_sd, group_size=64)

            model = AutoModelForImageTextToText.from_config(config, **model_kwargs)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys in state dict: {unexpected[:5]}")
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GLM-OCR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Text Recognition:"},
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
        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
