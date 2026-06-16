# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.1 model loader implementation.

Uses a locally modified DeepSeek V3-based model (modified_modeling_deepseek.py)
instead of loading directly from HuggingFace. The modifications adapt the model
for Tenstorrent hardware by replacing cache utilities with a static MLA cache.
"""
from pathlib import Path
from typing import Optional

import torch
import torch_xla.runtime as xr
from transformers import AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .meta_loading import load_model_from_checkpoint

from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts, enable_sparse_mlp

from .configuration_deepseek import DeepseekV3Config
from .modified_modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3MoE


class ModelVariant(StrEnum):
    """Available DeepSeek V3.1 model variants."""

    DEEPSEEK_V3_1_MODIFIED = "deepseek_v3_1_modified"


class ModelLoader(ForgeModel):
    """DeepSeek V3.1 model loader using the locally modified DeepSeek V3-based Transformer."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V3_1_MODIFIED: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V3.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V3_1_MODIFIED

    # BF16-dequantized weight mirror used when loading via the meta-loader path. TODO
    # The primary repo (pretrained_model_name) ships FP8 weights with scale-inv
    # auxiliaries; the BF16 mirror is what the custom Transformer expects.
    _BF16_WEIGHTS_REPO = "DevQuasar-2/deepseek-ai.DeepSeek-V3.1-BF16"

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. If None, uses the default variant.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the full model depth from config.json.
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting."""
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.1",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _load_config(self, num_layers: Optional[int] = None) -> DeepseekV3Config:
        """Load model configuration from the local config.json."""
        config_path = Path(__file__).parent / "config.json"
        config = DeepseekV3Config.from_json_file(str(config_path))
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        self.config = config
        return config

    def get_weight_dtype_config_path(self):
        """Return path to weight dtype config file, or None if not available."""
        return None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek V3.1 model.

        The model is instantiated from the local config.json and the locally
        modified modified_modeling_deepseek.py.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The DeepSeek V3.1 model in eval mode.
        """
        config = self._load_config(num_layers=self.num_layers)

        self._load_tokenizer()

        if self.num_layers is None:
            self.num_layers = 61

        # Load the model using the meta-loader to assign weights from the checkpoint
        model = load_model_from_checkpoint(
            lambda: DeepseekV3ForCausalLM(config),
            self._BF16_WEIGHTS_REPO,
            n_layers=self.num_layers,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model = model.eval()

        has_dense_moe = any(
            isinstance(layer.mlp, DeepseekV3MoE) for layer in model.model.layers
        )
        if has_dense_moe:
            num_devices = xr.global_runtime_device_count()
            mesh_shape, _ = self.get_mesh_config(num_devices)
            enable_sparse_mlp(
                model, mesh=mesh_shape, cluster_axis=0, config=model.config
            )

        self.model = model

        return model

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer()
        sample_prompt = (
            "Here is an exhaustive list of the best practices for writing clean code:"
        )
        inputs = self.tokenizer(
            [sample_prompt] * batch_size,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        return inputs["input_ids"]

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallelism.

        DeepSeek V3.1 only supports Galaxy (32 devices). The mesh is
        ``(4, 8)`` with axis 0 named ``"model"`` (attention TP, MoE/dense
        component) and axis 1 named ``"batch"`` (input/cache batch shard,
        MoE expert outer component).
        """
        if num_devices == 32:
            return (4, 8), ("batch", "model")
        raise ValueError(
            f"DeepSeek V3.1 is only supported on Galaxy (32 devices), got {num_devices}"
        )

    def load_shard_spec(self, model):
        """Build SPMD shard specifications for all model tensors.

        Translated from ``_apply_shard_specs`` in
        ``tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_1_new.py``:
          ``_axis_0`` (size 4) → ``"model"``
          ``_axis_1`` (size 8) → ``"batch"``
        """
        shard_specs = {}

        shard_specs[model.model.embed_tokens.weight] = (None, "model")
        shard_specs[model.model.norm.weight] = ("model",)
        shard_specs[model.lm_head.weight] = (None, "model")

        for layer in model.model.layers:
            sa = layer.self_attn
            shard_specs[sa.q_a_proj.weight] = (None, "model")
            shard_specs[sa.q_b_proj.weight] = ("model", None)
            shard_specs[sa.kv_a_proj_with_mqa.weight] = (None, "model")
            shard_specs[sa.kv_b_proj.weight] = ("model", None)
            shard_specs[sa.o_proj.weight] = (None, "model")

            shard_specs[layer.input_layernorm.weight] = ("model",)
            shard_specs[layer.post_attention_layernorm.weight] = ("model",)

            mlp = layer.mlp
            if isinstance(mlp, A2aSparseMLPWithSharedExperts):
                inner = mlp.mlp if hasattr(mlp, "mlp") else mlp
                shard_specs[inner.router.gate.weight] = (None, "model")
                shard_specs[inner.experts.gate_proj] = (
                    ("model", "batch"),
                    None,
                    None,
                )
                shard_specs[inner.experts.up_proj] = (
                    ("model", "batch"),
                    None,
                    None,
                )
                shard_specs[inner.experts.down_proj] = (
                    ("model", "batch"),
                    None,
                    None,
                )
                for bias_name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
                    b = getattr(inner.experts, bias_name, None)
                    if b is not None:
                        shard_specs[b] = (("model", "batch"), None)

                shared = getattr(mlp, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = (None, "model")
                    shard_specs[shared.up_proj.weight] = (None, "model")
                    shard_specs[shared.down_proj.weight] = ("model", None)
            else:
                shard_specs[mlp.gate_proj.weight] = ("batch", "model")
                shard_specs[mlp.up_proj.weight] = ("batch", "model")
                shard_specs[mlp.down_proj.weight] = ("model", "batch")

        return shard_specs
