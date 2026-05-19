# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 model loader implementation.

Uses a locally modified DeepSeek V3-based model (modeling_deepseek.py) instead of
loading directly from HuggingFace. The modifications adapt the model for
Tenstorrent hardware by replacing cache utilities with a static MLA cache.
"""
import gc
import time
from pathlib import Path
from typing import Optional

import torch
import torch_xla.runtime as xr
from loguru import logger
from transformers import AutoTokenizer

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

from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts, enable_sparse_mlp

from .configuration_deepseek import DeepseekV3Config
from .modified_modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3MoE


class ModelVariant(StrEnum):
    """Available Kimi K2 model variants."""

    KIMI_K2_INSTRUCT_MODIFIED = "kimi_k2_instruct_modified"


class ModelLoader(ForgeModel):
    """Kimi K2 model loader using the locally modified DeepSeek V3-based Transformer."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_INSTRUCT_MODIFIED: LLMModelConfig(
            pretrained_model_name="unsloth/Kimi-K2-Instruct-BF16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_INSTRUCT_MODIFIED

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
        self.model = None

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
        import os
        from .weight_loader import DEFAULT_CHECKPOINT_DIR

        checkpoint_dir = os.environ.get("KIMI_K2_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
        if checkpoint_dir == DEFAULT_CHECKPOINT_DIR and "KIMI_K2_CHECKPOINT_DIR" not in os.environ:
            logger.warning(
                f"[kimi_k2] KIMI_K2_CHECKPOINT_DIR not set, using default: {DEFAULT_CHECKPOINT_DIR}. "
                f"Set KIMI_K2_CHECKPOINT_DIR to override."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
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
        t_start = time.monotonic()

        # --- Stage 1: Config ---
        config = self._load_config(num_layers=self.num_layers)
        logger.info(
            f"[kimi_k2] Config loaded: num_hidden_layers={config.num_hidden_layers}, "
            f"hidden_size={config.hidden_size}, "
            f"num_experts={getattr(config, 'n_routed_experts', '?')}, "
            f"vocab_size={config.vocab_size}"
        )

        # --- Stage 2: Tokenizer ---
        self._load_tokenizer()
        logger.info(f"[kimi_k2] Tokenizer loaded: vocab_size={len(self.tokenizer)}")

        # --- Stage 3: Model construction ---
        # Construct directly in the target dtype to halve peak RSS
        # (avoid fp32 model + bf16 checkpoint held simultaneously).
        init_dtype = dtype_override or torch.bfloat16
        t_construct = time.monotonic()
        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(config)
        # Materialize on CPU in the target dtype (empty — will be overwritten).
        model = model.to_empty(device="cpu").to(init_dtype)
        model_params = sum(p.numel() for p in model.parameters())
        model_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        logger.info(
            f"[kimi_k2] Model constructed in {time.monotonic() - t_construct:.1f}s: "
            f"{model_params / 1e9:.2f}B params, "
            f"{model_bytes / (1024**3):.2f} GB (dtype={init_dtype})"
        )

        # --- Stage 4: Load real checkpoint weights ---
        from . import weight_loader

        layer_ids = list(range(config.num_hidden_layers))
        sd = weight_loader.load_transformer_state_dict(layer_ids)

        # Log key overlap diagnostics before load_state_dict.
        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys())
        overlap = model_keys & sd_keys
        logger.info(
            f"[kimi_k2] Key overlap: model has {len(model_keys)} keys, "
            f"checkpoint has {len(sd_keys)} keys, "
            f"overlap={len(overlap)}"
        )

        # Cast checkpoint tensors to match model dtype before loading to avoid
        # load_state_dict upcasting bf16 checkpoint -> fp32 model params.
        if init_dtype != torch.bfloat16:
            sd = {k: v.to(init_dtype) if v.is_floating_point() else v for k, v in sd.items()}

        t_load = time.monotonic()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            f"[kimi_k2] load_state_dict in {time.monotonic() - t_load:.1f}s: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

        # Log missing keys (these are expected: non-persistent buffers, caches).
        if missing:
            logger.info(f"[kimi_k2] Missing keys ({len(missing)}):")
            for k in sorted(missing):
                logger.debug(f"[kimi_k2]   MISSING: {k}")
            # Summarise by prefix for readability.
            prefixes: dict[str, int] = {}
            for k in missing:
                prefix = k.rsplit(".", 1)[0] if "." in k else k
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            for pfx, cnt in sorted(prefixes.items(), key=lambda x: -x[1])[:20]:
                logger.info(f"[kimi_k2]   missing prefix: {pfx} ({cnt} keys)")

        if unexpected:
            logger.warning(f"[kimi_k2] Unexpected keys ({len(unexpected)}):")
            for k in sorted(unexpected):
                logger.warning(f"[kimi_k2]   UNEXPECTED: {k}")

        del sd
        gc.collect()

        model = model.eval()

        # --- Stage 6: Verify no uninitialized parameters ---
        # After loading real weights + dtype cast, check for params that look
        # uninitialized (all-zero AND were not in the checkpoint overlap).
        loaded_keys = overlap  # keys that were overwritten by checkpoint
        n_suspect = 0
        for name, param in model.named_parameters():
            if name in loaded_keys:
                continue  # populated from checkpoint
            is_zero = param.count_nonzero() == 0
            if is_zero and param.numel() > 1:
                n_suspect += 1
                logger.warning(
                    f"[kimi_k2] SUSPECT unloaded param (all zeros): "
                    f"{name}  shape={list(param.shape)}  dtype={param.dtype}"
                )
        if n_suspect == 0:
            logger.info(
                "[kimi_k2] All non-checkpoint parameters have non-zero values (OK)"
            )
        else:
            logger.warning(
                f"[kimi_k2] {n_suspect} parameters are all-zero and were NOT "
                f"loaded from checkpoint -- these may be uninitialized"
            )

        # --- Stage 7: Enable sparse MLP ---
        has_dense_moe = any(
            isinstance(layer.mlp, DeepseekV3MoE) for layer in model.model.layers
        )
        if has_dense_moe:
            t_sparse = time.monotonic()
            num_devices = xr.global_runtime_device_count()
            mesh_shape, _ = self.get_mesh_config(num_devices)
            logger.info(
                f"[kimi_k2] Enabling sparse MLP: "
                f"mesh_shape={mesh_shape}, num_devices={num_devices}"
            )
            enable_sparse_mlp(
                model, mesh=mesh_shape, cluster_axis=0, config=model.config
            )
            logger.info(
                f"[kimi_k2] Sparse MLP enabled in "
                f"{time.monotonic() - t_sparse:.1f}s"
            )
        else:
            logger.info("[kimi_k2] No dense MoE layers found, skipping sparse MLP")

        self.model = model

        logger.info(
            f"[kimi_k2] load_model complete in {time.monotonic() - t_start:.1f}s"
        )
        return model

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
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
            raise ValueError(f"Kimi K2 is only supported on llmbox and galaxy")

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Args:
            model: The Kimi K2 model instance

        Returns:
            Dictionary mapping model parameters to their shard specifications,
            or None if sharding is not needed for this variant
        """

        shard_specs = {}

        # This expects the cache and inputs to be sharded as follows:
        # compressed_kv: ("batch", None, None, None)
        # k_pe:          ("batch", None, None, None)
        # input_ids:     ("batch", None)

        # Embedding and output layers
        shard_specs[model.model.embed_tokens.weight] = (None, "model")
        shard_specs[model.model.norm.weight] = ("model",)
        shard_specs[model.lm_head.weight] = ("batch", "model")

        for layer in model.model.layers:
            # MLA attention sharding
            shard_specs[layer.self_attn.q_a_proj.weight] = (None, "model")
            shard_specs[layer.self_attn.q_b_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "model")
            shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.o_proj.weight] = (None, "model")

            # MLP sharding: MoE for layers >= first_k_dense_replace, dense otherwise
            if isinstance(layer.mlp, A2aSparseMLPWithSharedExperts):
                # A2aSparseMLP: experts compound-sharded (batch, model)
                mlp_wrapper = layer.mlp
                mlp = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
                shard_specs[mlp.router.gate.weight] = (None, "model")
                shard_specs[mlp.experts.gate_proj] = (
                    ("batch", "model"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.up_proj] = (
                    ("batch", "model"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("batch", "model"),
                    None,
                    None,
                )

                # Shared experts (if present, on wrapper not on inner A2aSparseMLP)
                shared = getattr(mlp_wrapper, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = (None, "model")
                    shard_specs[shared.up_proj.weight] = (None, "model")
                    shard_specs[shared.down_proj.weight] = ("model", None)
            else:
                # Dense MLP (layer 0 only, given first_k_dense_replace=1)
                shard_specs[layer.mlp.gate_proj.weight] = ("batch", "model")
                shard_specs[layer.mlp.up_proj.weight] = ("batch", "model")
                shard_specs[layer.mlp.down_proj.weight] = ("model", "batch")

            shard_specs[layer.input_layernorm.weight] = ("model",)
            shard_specs[layer.post_attention_layernorm.weight] = ("model",)

        return shard_specs
