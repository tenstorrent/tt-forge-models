# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi 4 model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .overrides import override_phi3_modules


class ModelVariant(StrEnum):
    """Available Phi 4 model variants."""

    PHI_4 = "Phi_4"
    PHI_4_REASONING_PLUS = "Phi_4_reasoning_plus"


class ModelLoader(ForgeModel):
    """Phi 4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHI_4: ModelConfig(
            pretrained_model_name="microsoft/phi-4",
        ),
        ModelVariant.PHI_4_REASONING_PLUS: ModelConfig(
            pretrained_model_name="microsoft/Phi-4-reasoning-plus",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHI_4

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Phi-4",
            variant=variant,
            group=ModelGroup.RED,
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
        """Load and return the Phi 4 model instance for this instance's variant.

        Args:
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Phi 4 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model_kwargs["use_cache"] = False
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # De-fuse the fused qkv_proj / gate_up_proj into separate projections so
        # q (and the MLP gate/up) can be sharded column-parallel by head while
        # k/v stay replicated. Numerically identical to the fused model.
        override_phi3_modules(model)

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Phi 4 model with this instance's variant settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            List: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        input_prompt = "Africa is an emerging economy because"

        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        n_heads = self.config.num_attention_heads
        n_kv = self.config.num_key_value_heads
        if n_heads % num_devices == 0 and n_kv % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            num_devices % 2 == 0
            and n_heads % (num_devices // 2) == 0
            and n_kv % (num_devices // 2) == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {n_heads} q-heads / {n_kv} kv-heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style 1D TP for the (de-fused) Phi4 decoder.

        Column-parallel (shard out_features on the ``model`` axis) for q/k/v_proj
        and the MLP gate/up projections; row-parallel (shard in_features) for
        o_proj and down_proj. q/k/v are sharded consistently on the model
        axis (which ``get_mesh_config`` guarantees divides both the query and KV
        head counts), so each chip holds complete, matching q/k/v heads.

        """
        shard_specs = {}
        for layer in model.model.layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", None)
            shard_specs[attn.k_proj.weight] = ("model", None)
            shard_specs[attn.v_proj.weight] = ("model", None)
            shard_specs[attn.o_proj.weight] = (None, "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
