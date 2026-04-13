# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 model loader implementation.

Uses a locally modified DeepSeek V3-based model (modeling_deepseek.py) instead of
loading directly from HuggingFace. The modifications adapt the model for
Tenstorrent hardware by replacing cache utilities with a static MLA cache.
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

from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts, enable_sparse_mlp

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3MoE


class ModelVariant(StrEnum):
    """Available Kimi K2 model variants."""

    KIMI_K2_INSTRUCT = "kimi_k2_instruct"


class ModelLoader(ForgeModel):
    """Kimi K2 model loader using the locally modified DeepSeek V3-based Transformer."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_INSTRUCT: LLMModelConfig(
            pretrained_model_name="moonshotai/Kimi-K2-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_INSTRUCT

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. If None, uses the default variant.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the full model depth from config.json (61 layers).
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Kimi-K2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Unused; kept for API compatibility.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _load_config(self, num_layers: Optional[int] = None) -> DeepseekV3Config:
        """Load model configuration from the local config.json.

        Args:
            num_layers: If provided, overrides num_hidden_layers in the config.

        Returns:
            DeepseekV3Config: Populated config stored on ``self.config``.
        """
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
        """Load and return the Kimi K2 model.

        The model is instantiated from the local config.json and the locally
        modified modeling_deepseek.py.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The Kimi K2 model in eval mode.
        """
        config = self._load_config(num_layers=self.num_layers)

        self._load_tokenizer()

        model = DeepseekV3ForCausalLM(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

        # eval() must be called before enable_sparse_mlp so that the original
        # DeepseekV3MoE captured as _original_mlp (via object.__setattr__, outside
        # the registered module tree) has training=False.  model.eval() called
        # after enable_sparse_mlp cannot reach _original_mlp, causing
        # UnboundLocalError in DeepseekV3MoE.forward when _cpu_forward is used.
        model = model.eval()

        # Enable sparse MLP if not already applied. The benchmark infrastructure
        # calls load_model without num_devices, so we apply it here lazily using
        # the live device count.
        has_dense_moe = any(
            isinstance(layer.mlp, DeepseekV3MoE) for layer in model.model.layers
        )
        if has_dense_moe:
            num_devices = xr.global_runtime_device_count()
            mesh_shape, _ = self.get_mesh_config(num_devices)
            enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=model.config)

        return model

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
        if self.config is None:
            self._load_config()
        return torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism.

        Args:
            num_devices: Number of devices to use.

        Returns:
            Tuple of (mesh_shape, axis_names)
        """
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        else:
            raise ValueError(
                f"Kimi K2 is only supported on llmbox (8 devices) and galaxy (32 devices), got {num_devices}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Axis names match those returned by get_mesh_config: ("batch", "model").
        Attention layers use MLA (Multi-head Latent Attention) sharding:
          - q_a_proj / kv_a_proj_with_mqa: row-parallel (shard hidden dim on batch axis)
          - q_b_proj / kv_b_proj: column-parallel (shard output dim on model axis)
          - o_proj: 2D sharded (batch x model)
        MoE layers shard each expert's weights with tensor parallelism.
        Dense MLP layers (first_k_dense_replace=1, so only layer 0) use standard
        column/row parallel sharding.

        Args:
            model: The Kimi K2 DeepseekV3ForCausalLM model instance.

        Returns:
            Dictionary mapping model parameter tensors to their shard specs.
        """

        shard_specs = {}

        # Embedding and output layers
        # embed_tokens.weight: [vocab_size, hidden_size]
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        # norm.weight: [hidden_size]
        shard_specs[model.model.norm.weight] = ("batch",)
        # lm_head.weight: [vocab_size, hidden_size]
        shard_specs[model.lm_head.weight] = ("model", "batch")

        for layer in model.model.layers:
            # MLA attention sharding
            # q_a_proj.weight: [q_lora_rank, hidden_size] — row-parallel input projection
            shard_specs[layer.self_attn.q_a_proj.weight] = (None, "batch")
            # q_b_proj.weight: [num_heads * head_dim, q_lora_rank] — column-parallel
            shard_specs[layer.self_attn.q_b_proj.weight] = ("model", None)
            # q_a_layernorm.weight: [q_lora_rank] — replicated
            shard_specs[layer.self_attn.q_a_layernorm.weight] = (None,)
            # kv_a_proj_with_mqa.weight: [kv_lora_rank + qk_rope_head_dim, hidden_size]
            shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "batch")
            # kv_b_proj.weight: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", None)
            # kv_a_layernorm.weight: [kv_lora_rank] — replicated
            shard_specs[layer.self_attn.kv_a_layernorm.weight] = (None,)
            # o_proj.weight: [hidden_size, num_heads * v_head_dim] — 2D sharded
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            # MLP sharding: MoE for layers >= first_k_dense_replace, dense otherwise
            if isinstance(layer.mlp, A2aSparseMLPWithSharedExperts):
                # Sparse MoE: stacked expert weights sharded across all devices
                a2a = layer.mlp.mlp
                # router gate: [n_routed_experts, hidden_size]
                shard_specs[a2a.router.gate.weight] = (None, "batch")
                # Stacked expert weights: compound-sharded across both mesh axes
                # gate_proj: [E, H, inter], up_proj: [E, H, inter]
                shard_specs[a2a.experts.gate_proj] = (("batch", "model"), None, None)
                shard_specs[a2a.experts.up_proj] = (("batch", "model"), None, None)
                # down_proj: [E, inter, H]
                shard_specs[a2a.experts.down_proj] = (("batch", "model"), None, None)
                # Shared experts (n_shared_experts=1)
                if layer.mlp.shared_experts is not None:
                    shard_specs[layer.mlp.shared_experts.gate_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[layer.mlp.shared_experts.up_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[layer.mlp.shared_experts.down_proj.weight] = (
                        "batch",
                        "model",
                    )
            elif isinstance(layer.mlp, DeepseekV3MoE):
                # Non-sparse MoE: tensor-parallel within each expert
                # gate.weight: [n_routed_experts, hidden_size]
                shard_specs[layer.mlp.gate.weight] = (None, "batch")
                for expert in layer.mlp.experts:
                    # gate_proj.weight: [moe_intermediate_size, hidden_size]
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    # up_proj.weight: [moe_intermediate_size, hidden_size]
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    # down_proj.weight: [hidden_size, moe_intermediate_size]
                    shard_specs[expert.down_proj.weight] = ("batch", "model")
                # Shared experts (n_shared_experts=1)
                if (
                    hasattr(layer.mlp, "shared_experts")
                    and layer.mlp.shared_experts is not None
                ):
                    shard_specs[layer.mlp.shared_experts.gate_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[layer.mlp.shared_experts.up_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[layer.mlp.shared_experts.down_proj.weight] = (
                        "batch",
                        "model",
                    )
            else:
                # Dense MLP (layer 0 only, given first_k_dense_replace=1)
                # gate_proj.weight: [intermediate_size, hidden_size]
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                # up_proj.weight: [intermediate_size, hidden_size]
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                # down_proj.weight: [hidden_size, intermediate_size]
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            # Layer normalization weights: [hidden_size]
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

