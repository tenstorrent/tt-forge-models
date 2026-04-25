# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
StarVector model loader implementation for image-to-SVG generation.
"""

from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

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
    """Available StarVector model variants."""

    STARVECTOR_1B = "1B"
    STARVECTOR_8B = "8B"


class ModelLoader(ForgeModel):
    """StarVector model loader implementation for image-to-SVG generation tasks."""

    _VARIANTS = {
        ModelVariant.STARVECTOR_1B: ModelConfig(
            pretrained_model_name="starvector/starvector-1b-im2svg",
        ),
        ModelVariant.STARVECTOR_8B: ModelConfig(
            pretrained_model_name="starvector/starvector-8b-im2svg",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STARVECTOR_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="StarVector",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import sys
        import types
        import torch

        # cairosvg needs system libcairo2-dev which is unavailable; stub it out before
        # starvector imports it at module load time.
        if "cairosvg" not in sys.modules:
            stub = types.ModuleType("cairosvg")
            sys.modules["cairosvg"] = stub

        import starvector.model.llm.starcoder2 as sc2_module
        import starvector.model.models.starvector_v2 as sv2_module

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Patch the inner StarCoderModel to use from_config instead of from_pretrained.
        # The original code uses flash_attention_2 and calls from_pretrained inside the
        # outer from_pretrained meta-device context, both of which fail in this environment.
        original_sc2_init = sc2_module.StarCoderModel.__init__
        original_sv2_init = sv2_module.StarVectorStarCoder2.__init__

        def _patched_starcoder_init(self_inner, config, **kw):
            import torch.nn as nn

            nn.Module.__init__(self_inner)
            self_inner.init_tokenizer(config.starcoder_model_name)
            self_inner.max_length = config.max_length
            model_config = AutoConfig.from_pretrained(
                config.starcoder_model_name, trust_remote_code=True
            )
            model_config.use_cache = config.use_cache
            model_config.use_bfloat16 = True
            inner_model = AutoModelForCausalLM.from_config(
                model_config, torch_dtype=dtype
            )
            inner_model.resize_token_embeddings(len(self_inner.tokenizer))
            self_inner.transformer = inner_model
            self_inner.prompt = "<svg"
            from starvector.train.util import get_module_class_from_name

            transformer_layer_cls = kw.get(
                "transformer_layer_cls", "Starcoder2DecoderLayer"
            )
            self_inner.transformer_layer_cls = get_module_class_from_name(
                self_inner, transformer_layer_cls
            )

        def _patched_sv2_init(self_inner, config, **kw):
            from starvector.model.models.starvector_base import StarVectorBase
            from transformers import SiglipImageProcessor

            StarVectorBase.__init__(self_inner, config, **kw)
            # AutoImageProcessor fails with trust_remote_code on this model;
            # the preprocessor_config.json uses SiglipImageProcessor explicitly.
            self_inner.processor = SiglipImageProcessor.from_pretrained(
                config._name_or_path
            )

        sc2_module.StarCoderModel.__init__ = _patched_starcoder_init
        sv2_module.StarVectorStarCoder2.__init__ = _patched_sv2_init
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=True
            )
        finally:
            sc2_module.StarCoderModel.__init__ = original_sc2_init
            sv2_module.StarVectorStarCoder2.__init__ = original_sv2_init

        model.to(dtype=dtype).eval()
        # Use the image encoder's processor (correct size for the vision backbone)
        # rather than model.model.processor which may have the wrong image size.
        self.processor = model.model.image_encoder.processor

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image

        if self.processor is None:
            pretrained_model_name = self._variant_config.pretrained_model_name
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )

        img_proc = getattr(self.processor, "image_processor", self.processor)
        image_size = img_proc.size.get("height", 384)
        image = Image.new("RGB", (image_size, image_size))

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        # forward(batch) expects {"image": tensor, "svg": [str]} for the training path
        return [{"image": pixel_values, "svg": ["<svg></svg>"] * pixel_values.shape[0]}]
