# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AI21-Jamba-Large-1.6 model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available AI21 Jamba model variants for causal language modeling."""

    AI21_JAMBA_LARGE_1_6 = "AI21-Jamba-Large-1.6"


class ModelLoader(ForgeModel):
    """AI21-Jamba-Large-1.6 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AI21_JAMBA_LARGE_1_6: LLMModelConfig(
            pretrained_model_name="ai21labs/AI21-Jamba-Large-1.6",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AI21_JAMBA_LARGE_1_6

    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
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
            model="AI21-Jamba-Large-1.6",
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AI21-Jamba-Large-1.6 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The AI21-Jamba-Large-1.6 model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        print("model", model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AI21-Jamba-Large-1.6 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"

        # TT MoE EP runs along the model axis. Jamba has 16 experts; that
        # divides Galaxy's model-axis size 8 (16 % 32 would fail if EP used
        # all devices — same class of bug as WizardLM-2-8x22B).
        num_experts = getattr(self.config, "num_experts", None) or getattr(
            self.config, "num_local_experts", None
        )
        if num_experts is not None and num_experts % mesh_shape[1] != 0:
            raise ValueError(
                f"AI21-Jamba-Large-1.6 has {num_experts} experts, which is not "
                f"divisible by model-axis size {mesh_shape[1]}."
            )

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        # AI21-Jamba-Large-1.6 is JambaForCausalLM: hybrid attention / Mamba
        # blocks with either dense JambaMLP or JambaSparseMoeBlock as
        # ``feed_forward`` (not ``mlp``). No shared experts; no MLA.
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                # Attention decoder layer: standard GQA.
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("model", "batch")
                shard_specs[sa.k_proj.weight] = ("model", "batch")
                shard_specs[sa.v_proj.weight] = ("model", "batch")
                shard_specs[sa.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "mamba"):
                # Mamba decoder layer: channel-parallel along the expanded
                # intermediate dim (mamba_expand * hidden). Keep conv/SSM
                # params aligned with the sharded in_proj channels.
                mamba = layer.mamba
                shard_specs[mamba.in_proj.weight] = ("model", "batch")
                shard_specs[mamba.conv1d.weight] = ("model", None, None)
                if mamba.conv1d.bias is not None:
                    shard_specs[mamba.conv1d.bias] = ("model",)
                # x_proj is row-parallel on the sharded intermediate input.
                shard_specs[mamba.x_proj.weight] = ("batch", "model")
                shard_specs[mamba.dt_proj.weight] = ("model", "batch")
                if mamba.dt_proj.bias is not None:
                    shard_specs[mamba.dt_proj.bias] = ("model",)
                shard_specs[mamba.A_log] = ("model", None)
                shard_specs[mamba.D] = ("model",)
                shard_specs[mamba.out_proj.weight] = ("batch", "model")

            ff = layer.feed_forward
            if hasattr(ff, "experts"):
                # Sparse MoE: fused 3D experts, expert-parallel. Router stays
                # replicated so every device sees all expert logits.
                shard_specs[ff.experts.gate_up_proj] = ("model", None, None)
                shard_specs[ff.experts.down_proj] = ("model", None, None)
            else:
                # Dense MLP on non-expert layers.
                shard_specs[ff.gate_proj.weight] = ("model", "batch")
                shard_specs[ff.up_proj.weight] = ("model", "batch")
                shard_specs[ff.down_proj.weight] = ("batch", "model")
        # vocab_size=65536 divides cleanly by common model-axis sizes.
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
