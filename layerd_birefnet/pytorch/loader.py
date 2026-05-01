# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cyberagent/layerd-birefnet model loader implementation for image segmentation / matting
"""

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available cyberagent/layerd-birefnet model variants."""

    LAYERD_BIREFNET = "layerd-birefnet"


class ModelLoader(ForgeModel):
    """cyberagent/layerd-birefnet model loader implementation for image segmentation / matting tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LAYERD_BIREFNET: ModelConfig(
            pretrained_model_name="cyberagent/layerd-birefnet",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LAYERD_BIREFNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="layerd_birefnet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return self.transform_image

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        model_kwargs["dtype"] = (
            dtype_override if dtype_override is not None else torch.float32
        )
        model_kwargs |= kwargs

        # transformers 5.x instantiates models under torch.device("meta").
        # SwinTransformer calls torch.linspace(...).item() to build drop-path
        # rates during __init__; meta tensors have no data so .item() raises.
        # Return 0.0 for meta scalars — DropPath is a no-op in eval mode anyway.
        _orig_item = torch.Tensor.item

        def _meta_safe_item(self):
            if self.device.type == "meta":
                return 0.0
            return _orig_item(self)

        # transformers 5.x _finalize_model_loading needs all_tied_weights_keys,
        # which post_init() sets. BiRefNet.__init__ never calls post_init().
        from transformers.modeling_utils import PreTrainedModel

        _orig_finalize = PreTrainedModel.__dict__["_finalize_model_loading"].__func__

        @staticmethod
        def _patched_finalize(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                model.post_init()
            return _orig_finalize(model, load_config, loading_info)

        torch.Tensor.item = _meta_safe_item
        PreTrainedModel._finalize_model_loading = _patched_finalize
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )
        finally:
            torch.Tensor.item = _orig_item
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize)

        torch.set_float32_matmul_precision(["high", "highest"][0])

        if self.transform_image is None:
            self._setup_transforms()

        self._patch_deformable_conv(model)

        return model

    @staticmethod
    def _patch_deformable_conv(model):
        """Patch DeformableConv2d.forward to float32-cast for CPU-only bfloat16 compatibility.

        torchvision.ops.deform_conv2d is not implemented for bfloat16 on CPU.
        """
        from torchvision.ops import deform_conv2d
        import types

        def _bf16_safe_forward(self_conv, x):
            orig_dtype = x.dtype
            offset = self_conv.offset_conv(x)
            modulator = 2.0 * torch.sigmoid(self_conv.modulator_conv(x))
            if orig_dtype == torch.bfloat16:
                x = x.float()
                offset = offset.float()
                modulator = modulator.float()
                weight = self_conv.regular_conv.weight.float()
                bias = (
                    self_conv.regular_conv.bias.float()
                    if self_conv.regular_conv.bias is not None
                    else None
                )
                out = deform_conv2d(
                    input=x,
                    offset=offset,
                    weight=weight,
                    bias=bias,
                    padding=self_conv.padding,
                    mask=modulator,
                    stride=self_conv.stride,
                )
                return out.to(orig_dtype)
            return deform_conv2d(
                input=x,
                offset=offset,
                weight=self_conv.regular_conv.weight,
                bias=self_conv.regular_conv.bias,
                padding=self_conv.padding,
                mask=modulator,
                stride=self_conv.stride,
            )

        for module in model.modules():
            if type(module).__name__ == "DeformableConv2d":
                module.forward = types.MethodType(_bf16_safe_forward, module)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        dataset = load_dataset("huggingface/cats-image")["test"]
        self.image = dataset[0]["image"]

        inputs = self.transform_image(self.image).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
