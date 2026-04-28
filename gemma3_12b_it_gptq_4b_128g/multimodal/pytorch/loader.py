# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 12B IT GPTQ 4-bit 128g model loader implementation for multimodal modeling.
"""

from typing import Optional

from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

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
from PIL import Image


class ModelVariant(StrEnum):
    """Available Gemma3 12B IT GPTQ 4-bit 128g multimodal model variants."""

    GEMMA_3_12B_IT_GPTQ_4B_128G = "ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g"


class ModelLoader(ForgeModel):
    """Gemma3 12B IT GPTQ 4-bit 128g model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_12B_IT_GPTQ_4B_128G: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_12B_IT_GPTQ_4B_128G),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_12B_IT_GPTQ_4B_128G

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
            model="gemma_3_12b_it_gptq_4b_128g_multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        pretrained_model_name = self._variant_config.pretrained_model_name
        # use_fast=False: transformers 5.x defaults to the fast Gemma3ImageProcessor,
        # which is a breaking change vs. the slow processor the checkpoint was saved with.
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, use_fast=False, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma3 12B IT GPTQ 4-bit 128g multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gemma3 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # compressed_tensors 0.15.x uses re.match (anchored at start) for ignore
        # patterns. The checkpoint's ignore list uses "re:vision_tower.*" which
        # doesn't match "model.vision_tower.*" module paths, so the vision encoder
        # (intermediate_size=4304, not divisible by group_size=128) gets included
        # in quantization and crashes compress_model. Fix by rewriting re: patterns
        # with a ".*" prefix.
        #
        # Gemma3ForConditionalGeneration also has a top-level .lm_head (tied to
        # embed_tokens.weight) that is separate from "language_model.lm_head".
        # The checkpoint only ignores the inner one, so the outer lm_head gets
        # quantized and mark_tied_weights_as_initialized then crashes. Add "lm_head"
        # to cover it.
        #
        # Set run_compressed=False to dequantize weights to float before TT
        # compilation (TT silicon does not accept GPTQ-packed formats).
        config = AutoConfig.from_pretrained(pretrained_model_name)
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            ignore = qc.get("ignore", [])
            ignore = [
                f"re:.*{p[3:]}" if p.startswith("re:") else p
                for p in ignore
            ]
            if "lm_head" not in ignore:
                ignore.append("lm_head")
            qc["ignore"] = ignore
            qc["run_compressed"] = False
        model_kwargs["config"] = config

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # compressed_tensors leaves a quantized_forward instance-method bound on
        # each Linear after decompression. It accesses weight.data unconditionally,
        # which conflicts with TT-XLA's __torch_function__ during torch.compile.
        # Restore the standard forward by removing the instance-level override.
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
        """Load and return sample inputs for the Gemma3 12B IT GPTQ 4-bit 128g multimodal model.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
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

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
        shard_specs = {}

        # Gemma3ForConditionalGeneration nests vision_tower and language_model
        # under self.model (a Gemma3Model), so access via model.model.*
        for layer in model.model.vision_tower.vision_model.encoder.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs
