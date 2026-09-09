# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoConfig
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


class ModelVariant(StrEnum):
    """Available Qwen 3 VL model variants for causal language modeling."""

    QWEN_3_VL_32B_INSTRUCT = "32b_instruct"


class ModelLoader(ForgeModel):
    """Qwen 3 VL model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_INSTRUCT

    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
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
            model="Qwen 3 VL",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        # Qwen3-VL ships a multimodal processor; pad_token lives on the inner tokenizer.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.tokenizer = self.processor.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3 VL model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3 VL model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_tokenizer()

        # Qwen3VLProcessor.__call__ takes images as the first positional arg, so
        # text-only inputs must go through apply_chat_template with tokenize=True
        # (which passes text=...) rather than processor(prompt).
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
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": self._variant_config.max_length,
            },
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specifications for the full VLM.

        Module layout (Qwen3VLForConditionalGeneration):
            model.visual.blocks / merger / deepstack_merger_list
            model.language_model.layers (dense full-attention)
            lm_head
        """
        shard_specs = {}

        for block in model.model.visual.blocks:
            # Megatron-style: fused qkv is column-parallel (shard output / heads),
            # proj is row-parallel (shard input / heads -> all-reduce after).
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            if block.attn.qkv.bias is not None:
                shard_specs[block.attn.qkv.bias] = ("model",)
            shard_specs[block.attn.proj.weight] = ("batch", "model")

            shard_specs[block.mlp.linear_fc1.weight] = ("model", None)
            if block.mlp.linear_fc1.bias is not None:
                shard_specs[block.mlp.linear_fc1.bias] = ("model",)
            shard_specs[block.mlp.linear_fc2.weight] = (None, "model")

        def _shard_patch_merger(merger):
            shard_specs[merger.linear_fc1.weight] = ("model", "batch")
            if merger.linear_fc1.bias is not None:
                shard_specs[merger.linear_fc1.bias] = ("model",)
            shard_specs[merger.linear_fc2.weight] = ("batch", "model")

        _shard_patch_merger(model.model.visual.merger)
        for merger in model.model.visual.deepstack_merger_list:
            _shard_patch_merger(merger)

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

            sa = layer.self_attn
            shard_specs[sa.q_proj.weight] = ("batch", "model")
            shard_specs[sa.k_proj.weight] = ("batch", "model")
            shard_specs[sa.v_proj.weight] = ("batch", "model")
            shard_specs[sa.o_proj.weight] = ("model", "batch")

        # VLM text decoder embedding lives at model.model.language_model
        # (model.model only exposes .visual / .language_model). Vocab on
        # "model", hidden on "batch" -> embedding output keeps hidden on
        # "batch", matching the residual axis.
        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
