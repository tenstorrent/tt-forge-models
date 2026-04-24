# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cyberagent/layerd-birefnet model loader implementation for image segmentation / matting
"""

import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import deform_conv2d
from transformers import AutoModelForImageSegmentation
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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

    def _patch_birefnet_module(self, pretrained_model_name):
        # transformers 5.x wraps model __init__ in torch.device("meta"), but
        # SwinTransformer.__init__ calls torch.linspace(...).item() which fails
        # on meta tensors. Wrap __init__ in torch.device("cpu") to override.
        SwinTransformer = get_class_from_dynamic_module(
            "birefnet.SwinTransformer",
            pretrained_model_name,
            trust_remote_code=True,
        )
        _orig_swin_init = SwinTransformer.__init__

        def _patched_swin_init(self, *args, **kwargs):
            with torch.device("cpu"):
                _orig_swin_init(self, *args, **kwargs)

        SwinTransformer.__init__ = _patched_swin_init

        # BiRefNet doesn't call self.post_init(), which transformers 5.x requires
        # to set up all_tied_weights_keys before weight loading.
        BiRefNet = get_class_from_dynamic_module(
            "birefnet.BiRefNet",
            pretrained_model_name,
            trust_remote_code=True,
        )
        _orig_birefnet_init = BiRefNet.__init__

        def _patched_birefnet_init(self, *args, **kwargs):
            _orig_birefnet_init(self, *args, **kwargs)
            self.post_init()

        BiRefNet.__init__ = _patched_birefnet_init

        # torchvision deform_conv2d doesn't support BFloat16; cast to float32.
        DeformableConv2d = get_class_from_dynamic_module(
            "birefnet.DeformableConv2d",
            pretrained_model_name,
            trust_remote_code=True,
        )

        def _patched_deformable_forward(self, x):
            dtype = x.dtype
            offset = self.offset_conv(x)
            modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
            bias = self.regular_conv.bias
            return deform_conv2d(
                input=x.float(),
                offset=offset.float(),
                weight=self.regular_conv.weight.float(),
                bias=bias.float() if bias is not None else None,
                padding=self.padding,
                mask=modulator.float(),
                stride=self.stride,
            ).to(dtype)

        DeformableConv2d.forward = _patched_deformable_forward

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._patch_birefnet_module(pretrained_model_name)

        model_kwargs = {}
        model_kwargs["dtype"] = (
            dtype_override if dtype_override is not None else torch.float32
        )
        model_kwargs |= kwargs

        model = AutoModelForImageSegmentation.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        torch.set_float32_matmul_precision(["high", "highest"][0])

        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        self.image = Image.new("RGB", (1024, 1024))

        inputs = self.transform_image(self.image).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
