# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-122B-A10B multimodal (vision–language) model loader.

Uses Hugging Face `Qwen3_5MoeForConditionalGeneration` via
`AutoModelForImageTextToText` and `AutoProcessor` (Transformers ≥ 4.57).
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen 3.5 model variants."""

    QWEN_3_5_122B_A10B = "122B_A10B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 MoE VLM loader (image/text/video in, text out)."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-122B-A10B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_122B_A10B

    sample_text = 'Type "I love Qwen3.5" backwards'

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load AutoProcessor for the current variant.

        Args:
            dtype_override: Optional torch.dtype passed to ``from_pretrained`` when supported.

        Returns:
            The loaded processor instance.
        """
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load ``AutoModelForImageTextToText`` for this variant (Qwen3.5 MoE VLM).

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the checkpoint default is used (often bfloat16).

        Returns:
            torch.nn.Module: The loaded model in eval mode.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        # Default text MoE backend: "eager" loops per expert instead of torch._grouped_mm.
        # grouped_mm hits "expected data_ptr to be aligned to 16 bytes" under tt_torch on CPU/XLA
        # (see tt-xla tests/benchmark/benchmarks/llm_benchmark.py).
        model_kwargs.setdefault(
            "experts_implementation",
            {"text_config": "eager"},
        )

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        print("model", model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build sample inputs via the processor chat template (text-only).

        Args:
            dtype_override: If set, float32 tensors are cast to this dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Tensor inputs for ``forward`` / ``generate``.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Qwen3.5 processor expects content as a list of typed parts, not a bare string.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.sample_text}],
            }
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

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        core = model.model
        vm = core.visual

        for block in vm.blocks:
            shard_specs[block.norm1.weight] = ("model",)
            shard_specs[block.norm1.bias] = ("model",)
            shard_specs[block.norm2.weight] = ("model",)
            shard_specs[block.norm2.bias] = ("model",)
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            shard_specs[block.attn.qkv.bias] = ("model",)
            shard_specs[block.attn.proj.weight] = ("batch", "model")
            shard_specs[block.attn.proj.bias] = ("model",)
            shard_specs[block.mlp.linear_fc1.weight] = ("model", "batch")
            shard_specs[block.mlp.linear_fc1.bias] = ("model",)
            shard_specs[block.mlp.linear_fc2.weight] = ("batch", "model")
            shard_specs[block.mlp.linear_fc2.bias] = ("model",)

        merger = vm.merger
        shard_specs[merger.norm.weight] = ("model",)
        shard_specs[merger.norm.bias] = ("model",)
        shard_specs[merger.linear_fc1.weight] = ("model", "batch")
        shard_specs[merger.linear_fc1.bias] = ("model",)
        shard_specs[merger.linear_fc2.weight] = ("batch", "model")
        shard_specs[merger.linear_fc2.bias] = ("model",)

        lm = core.language_model
        shard_specs[lm.embed_tokens.weight] = ("model", "batch")
        shard_specs[lm.norm.weight] = ("model",)

        for layer in lm.layers:
            shard_specs[layer.input_layernorm.weight] = ("model",)
            shard_specs[layer.post_attention_layernorm.weight] = ("model",)

            mlp = layer.mlp
            shard_specs[mlp.gate.weight] = ("model", "batch")
            se = mlp.shared_expert
            shard_specs[se.gate_proj.weight] = ("model", "batch")
            shard_specs[se.up_proj.weight] = ("model", "batch")
            shard_specs[se.down_proj.weight] = ("batch", "model")
            shard_specs[mlp.shared_expert_gate.weight] = ("model", "batch")

            if layer.layer_type == "full_attention":
                attn = layer.self_attn
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                shard_specs[attn.k_proj.weight] = ("model", "batch")
                shard_specs[attn.v_proj.weight] = ("model", "batch")
                shard_specs[attn.o_proj.weight] = ("batch", "model")
            else:
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[la.in_proj_z.weight] = ("model", "batch")
                shard_specs[la.in_proj_b.weight] = ("model", "batch")
                shard_specs[la.in_proj_a.weight] = ("model", "batch")
                shard_specs[la.out_proj.weight] = ("batch", "model")

        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
