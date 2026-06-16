# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiffusionGemma loader (text-only path).

google/diffusiongemma-26B-A4B-it: a multimodal block-diffusion LLM on a Gemma 4
MoE backbone that denoises a block of tokens instead of decoding left-to-right.
~25.8B params.
"""

from typing import Optional

from transformers import AutoProcessor

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
from ...tools.utils import cast_input_to_type
from .utils import _install_dynamo_safe_param_props


class ModelVariant(StrEnum):
    """Available DiffusionGemma model variants."""

    DIFFUSIONGEMMA_26B_A4B_IT = "26B-A4B-it"


class ModelLoader(ForgeModel):
    """DiffusionGemma loader (text-only block-diffusion path)."""

    _VARIANTS = {
        ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/diffusiongemma-26B-A4B-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT

    sample_text = "Why is the sky blue?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DiffusionGemma",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the DiffusionGemmaForBlockDiffusion model (text-only path)."""
        from transformers import DiffusionGemmaForBlockDiffusion

        if self.processor is None:
            self._load_processor()
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = _install_dynamo_safe_param_props(model)
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self, dtype_override=None, batch_size=1, prompt: Optional[str] = None
    ):
        """Build text inputs via the chat template (dict -> keyword-bound)."""
        if self.processor is None:
            self._load_processor()
        inputs = dict(
            self.processor.apply_chat_template(
                [{"role": "user", "content": prompt or self.sample_text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
        for key in list(inputs):
            value = inputs[key].repeat_interleave(batch_size, dim=0)
            inputs[key] = cast_input_to_type(value, dtype_override)
        return inputs

    def _text_layers(self, model):
        """Encoder then decoder text transformer layers."""
        base = model.model
        return list(base.encoder.language_model.layers) + list(base.decoder.layers)

    def get_mesh_config(self, num_devices: int):
        """((1, num_devices), ("batch", "model")); model axis carries query heads
        and the expert axis, so both must divide it (KV is replicated)."""
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        assert text_cfg.num_attention_heads % mesh_shape[1] == 0
        assert text_cfg.num_experts % mesh_shape[1] == 0
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron TP (q col / o row, dense MLP col->row) + expert-parallel MoE
        (3D expert tensors sharded on the expert axis). K/V replicated: the
        full-attention layers carry only 2 global KV heads (unsharddable)."""
        shard_specs = {}
        for layer in self._text_layers(model):
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", None)
            shard_specs[attn.o_proj.weight] = (None, "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            experts = getattr(layer, "experts", None)
            if experts is not None:
                shard_specs[experts.gate_up_proj] = ("model", None, None)
                shard_specs[experts.down_proj] = ("model", None, None)
        return shard_specs
