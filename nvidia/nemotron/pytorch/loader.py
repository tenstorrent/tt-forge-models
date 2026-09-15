# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron-3.5-Lightning model loader implementation.
"""

import torch
import torch_xla.runtime as xr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available NVIDIA Nemotron model variants for causal language modeling."""

    Nvidia_Nemotron_3_5_Lightning_30B_A3B_BF16 = (
        "Nvidia_Nemotron_3_5_Lightning_30B_A3B_BF16"
    )


class ModelLoader(ForgeModel):
    """NVIDIA Nemotron-3.5-Lightning model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Nvidia_Nemotron_3_5_Lightning_30B_A3B_BF16: LLMModelConfig(
            pretrained_model_name="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Nvidia_Nemotron_3_5_Lightning_30B_A3B_BF16

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
            model="NVIDIA-Nemotron-3.5-Lightning",
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

        # Some Nemotron tokenizers ship without a pad token; reuse EOS so padding works.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nemotron model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Nemotron model for causal language modeling.
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
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Nemotron model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        if self.tokenizer.chat_template is not None:
            conversation = [{"role": "user", "content": self.sample_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.sample_text
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
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Shard specs for hybrid NemotronH blocks (Mamba2 / Attention / MoE).

        Follows the qwen_3_5 MoE pattern:
        - Routed expert weights (``mixer.experts.up_proj`` / ``down_proj``) are
          *not* set here — ``get_tt_moe_shard_specs`` (via ``inject_custom_moe``)
          shards them on the expert dimension with the compound mesh axis.
        - Router stays replicated so every device can score all experts.
        - Shared expert is a dense up/down MLP (column / row TP).
        - Attention: Q/K/V column TP, O row TP (same residual layout as MoE),
          but only when whole heads land on each device; otherwise attention
          is left replicated, matching the fallback GLM's language model uses
          (see ``glm_image/pytorch/src/model_utils.py::_heads_divide_model_axis``).
        - Mamba2: same layout as qwen_3_5 gated-delta ``linear_attn``.
        """
        num_devices = xr.global_runtime_device_count()
        mesh_shape, _ = self.get_mesh_config(num_devices)
        model_axis_size = mesh_shape[1]
        heads_divide_model_axis = (
            self.config.num_attention_heads % model_axis_size == 0
            and self.config.num_key_value_heads % model_axis_size == 0
        )

        shard_specs = {}
        for layer in model.model.layers:
            mixer = layer.mixer

            if hasattr(mixer, "experts"):
                # MoE: only TP the always-on shared expert. Routed experts are
                # handled by get_tt_moe_shard_specs.
                shared = getattr(mixer, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.up_proj.weight] = ("model", "batch")
                    shard_specs[shared.down_proj.weight] = ("batch", "model")

            elif hasattr(mixer, "q_proj"):
                # NemotronHAttention. num_key_value_heads (2) does not divide
                # evenly across every supported device count (e.g. 8): head-
                # parallel TP would split a KV head's features mid-head,
                # forcing Shardy to emit a sub-axis collective that tt-mlir's
                # sdy -> stablehlo CCL lowering can't lower, which surfaces
                # downstream as "'sdy.all_slice' op operates on axis ... which
                # is already bound by a parent sdy.manual_computation op".
                if heads_divide_model_axis:
                    shard_specs[mixer.q_proj.weight] = ("model", "batch")
                    shard_specs[mixer.k_proj.weight] = ("model", "batch")
                    shard_specs[mixer.v_proj.weight] = ("model", "batch")
                    shard_specs[mixer.o_proj.weight] = ("batch", "model")

            elif hasattr(mixer, "in_proj") and hasattr(mixer, "out_proj"):
                # NemotronHMamba2Mixer — mirror qwen_3_5 linear_attn.
                shard_specs[mixer.in_proj.weight] = ("model", "batch")
                if hasattr(mixer, "conv1d"):
                    shard_specs[mixer.conv1d.weight] = (None, None, None)
                shard_specs[mixer.out_proj.weight] = ("batch", "model")
                if hasattr(mixer, "dt_bias"):
                    shard_specs[mixer.dt_bias] = ("model",)
                if hasattr(mixer, "A_log"):
                    shard_specs[mixer.A_log] = ("model",)
                if hasattr(mixer, "D"):
                    shard_specs[mixer.D] = ("model",)

        if hasattr(model.model, "embed_tokens"):
            shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
