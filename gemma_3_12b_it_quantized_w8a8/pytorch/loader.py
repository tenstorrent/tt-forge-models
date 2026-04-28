# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RedHatAI Gemma 3 12B IT quantized W8A8 model loader implementation for multimodal vision-language modeling.
"""

from typing import Optional

from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

from PIL import Image

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Gemma 3 12B IT quantized W8A8 model variants."""

    GEMMA_3_12B_IT_QUANTIZED_W8A8 = "12B_IT_Quantized_W8A8"


class ModelLoader(ForgeModel):
    """RedHatAI Gemma 3 12B IT quantized W8A8 model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_12B_IT_QUANTIZED_W8A8: LLMModelConfig(
            pretrained_model_name="RedHatAI/gemma-3-12b-it-quantized.w8a8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_12B_IT_QUANTIZED_W8A8

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
            model="Gemma 3 12B IT Quantized W8A8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=False, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        # compressed-tensors 0.15.x keeps weights in int8 by default (run_compressed=True).
        # TT does not support int8 matmuls via this path, so set run_compressed=False to
        # dequantize all quantized Linear weights to bfloat16 during load.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            qc["run_compressed"] = False
        elif qc is not None:
            qc.run_compressed = False
        model_kwargs["config"] = config

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # compressed-tensors leaves a quantized_forward instance method bound on every
        # quantized Linear after decompression. It accesses weight.data unconditionally,
        # which triggers TT-XLA's __torch_function__ before compilation is ready and
        # raises AttributeError: 'fused_0' object has no attribute 'xla_args'.
        # Restore the class-level forward by removing the instance-level override.
        for m in model.modules():
            if "forward" in m.__dict__:
                del m.__dict__["forward"]

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        for layer in model.model.vision_tower.vision_model.encoder.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

        return shard_specs
