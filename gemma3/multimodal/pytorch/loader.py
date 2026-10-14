# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 model loader implementation for multimodal modeling.
"""

from typing import Optional, Any

import torch
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

EXACT_SOFTMAX_ATTN = "tt_exact_softmax_gemma3"


def _exact_softmax_attention(
    module, query, key, value, attention_mask, scaling, **kwargs
):
    """fp32 attention with softmax written as exp * reciprocal(sum).

    The stock exp / sum fuses into ttnn.softmax, which tt-mlir leaves at
    math_approx_mode=True (rows sum to up to 1.039). This form is not fused
    and is exact on TT. Remove once tt-mlir disables the approximation.

    ``num_key_value_groups`` defaults to 1 so this also serves the SigLIP
    vision tower (plain MHA, no grouped-query attention), which gets this
    implementation when ``set_attn_implementation`` propagates to the tower.
    """
    from transformers.models.gemma3.modeling_gemma3 import repeat_kv

    n_rep = getattr(module, "num_key_value_groups", 1)
    key = repeat_kv(key, n_rep).float()
    value = repeat_kv(value, n_rep).float()
    scores = torch.matmul(query.float(), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    exp = torch.exp(scores - torch.amax(scores, dim=-1, keepdim=True))
    probs = exp * torch.reciprocal(torch.sum(exp, dim=-1, keepdim=True))
    out = torch.matmul(probs, value).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def _use_exact_softmax_attention(model):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(EXACT_SOFTMAX_ATTN, _exact_softmax_attention)
    AttentionMaskInterface.register(EXACT_SOFTMAX_ATTN, eager_mask)
    model.set_attn_implementation(EXACT_SOFTMAX_ATTN)


def _use_fp32_vision_path(model):
    """Run the SigLIP vision tower and projector in fp32.

    bf16 rounding in the vision tower carries into the image tokens fed to the
    language model. Keep the tower, the multi-modal projector and pixel_values
    in fp32; the model's merge casts the image features back to the text dtype
    (modeling_gemma3: ``image_features.to(inputs_embeds.device, inputs_embeds.dtype)``),
    so the language model still runs in its native dtype.
    """
    model.model.vision_tower.float()
    model.model.multi_modal_projector.float()


class ModelVariant(StrEnum):
    """Available Gemma3 multimodal model variants."""

    GEMMA_3_4B_IT = "google/gemma-3-4b-it"
    GEMMA_3_12B_IT = "google/gemma-3-12b-it"
    GEMMA_3_27B_IT = "google/gemma-3-27b-it"
    MEDGEMMA_27B_IT = "google/medgemma-27b-it"


class ModelLoader(ForgeModel):
    """Gemma3 model loader implementation for multimodal modeling tasks."""

    _SLIDING_WINDOW_VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT,
        ModelVariant.GEMMA_3_12B_IT,
        ModelVariant.GEMMA_3_27B_IT,
        ModelVariant.MEDGEMMA_27B_IT,
    }

    # Variants large enough to need tensor-parallel weight sharding. Smaller
    # variants fit on one device and return no shard spec.
    _TENSOR_PARALLEL_VARIANTS = {
        ModelVariant.GEMMA_3_27B_IT,
        ModelVariant.MEDGEMMA_27B_IT,
    }

    # Multimodal variants show low PCC against the CPU golden from two sources:
    # the approximate ttnn.softmax (rows sum to ~1.039) and bf16 rounding in the
    # vision tower. For these variants attention runs exact (fp32 exp*reciprocal)
    # and the vision tower/projector run in fp32. Mirrors tt-forge-models PR #944
    # (qwen_3_6 / qwen_3_8) and the gemma4 loader. Applied to all variants.
    _EXACT_SOFTMAX_VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT,
        ModelVariant.GEMMA_3_12B_IT,
        ModelVariant.GEMMA_3_27B_IT,
        ModelVariant.MEDGEMMA_27B_IT,
    }

    _VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_4B_IT),
            max_length=512,
        ),
        ModelVariant.GEMMA_3_12B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_12B_IT),
            max_length=512,
        ),
        ModelVariant.GEMMA_3_27B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_3_27B_IT),
            max_length=512,
        ),
        ModelVariant.MEDGEMMA_27B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MEDGEMMA_27B_IT),
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_4B_IT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def requires_model_rewrites(self) -> bool:
        """Sliding-window variants need the TT causal-mask rewrite applied."""
        return self._variant in self._SLIDING_WINDOW_VARIANTS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if any(x in variant.value for x in ["12b", "27b"]):
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model=(
                "medgemma_multimodal"
                if variant == ModelVariant.MEDGEMMA_27B_IT
                else "gemma_3_multimodal"
            ),
            variant=variant,
            group=group,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma3 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gemma3 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()

        # 27B variants: exact-softmax attention + fp32 vision tower to recover
        # multimodal PCC (see _EXACT_SOFTMAX_VARIANTS).
        if self._variant in self._EXACT_SOFTMAX_VARIANTS:
            _use_exact_softmax_attention(model)
            _use_fp32_vision_path(model)

        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        batch_size: int = 1,
    ):
        """Load and return sample inputs for the Gemma3 multimodal model with default settings.

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

        if self._variant in self._EXACT_SOFTMAX_VARIANTS:
            # fp32 vision tower: keep pixel_values in fp32 regardless of the
            # text dtype_override (see _use_fp32_vision_path). The model's merge
            # casts the resulting image features back to the text dtype.
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], torch.float32
            )
        elif dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        # Non-sliding variants: return standard tokenizer output
        if self._variant not in self._SLIDING_WINDOW_VARIANTS:
            return inputs

        from tools.utils import prepare_inputs_for_sliding_window_attention

        # The shared helper only forwards the text-side keys (input_ids,
        # attention_mask, cache, ...). Preserve the multimodal inputs so the
        # vision tower still receives the image and the image/text token
        # boundaries are kept.
        multimodal_inputs = {
            key: inputs[key]
            for key in ("pixel_values", "token_type_ids")
            if key in inputs
        }

        inputs = prepare_inputs_for_sliding_window_attention(
            inputs,
            batch_size=batch_size,
            max_cache_len=self._variant_config.max_length,
            dtype_override=dtype_override,
            config=self.config,
        )
        inputs.update(multimodal_inputs)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names) where mesh_shape is (batch_dim, model_dim)
                   and mesh_axis_names are ("batch", "model").
        """
        mesh_shape = (1, num_devices)
        if self._variant not in [
            ModelVariant.GEMMA_3_4B_IT,
            ModelVariant.GEMMA_3_12B_IT,
        ]:
            assert (
                self.config.text_config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution.

        Args:
            model: The Gemma3ForConditionalGeneration model instance.

        Returns:
            dict: Dictionary mapping model parameters to their sharding specification,
                  or None if tensor parallelism is not needed for this variant.
        """
        if self._variant not in self._TENSOR_PARALLEL_VARIANTS:
            return None

        shard_specs = {}

        base = getattr(model, "model", model)

        for layer in base.vision_tower.vision_model.encoder.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

        for layer in base.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        shard_specs[model.model.language_model.embed_tokens.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs
