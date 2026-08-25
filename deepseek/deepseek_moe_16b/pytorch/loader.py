# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek MoE-16B (deepseek-ai/deepseek-moe-16b-chat) model loader implementation.
"""

import torch
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
    """Available DeepSeek MoE-16B model variants for causal language modeling."""

    DEEPSEEK_MOE_16B_CHAT = "DEEPSEEK_MOE_16B_CHAT"


class ModelLoader(ForgeModel):
    """DeepSeek MoE-16B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_MOE_16B_CHAT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-moe-16b-chat",
            max_length=256,
        )
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_MOE_16B_CHAT

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
            model="DeepSeek-MoE-16B",
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
        self._ensure_transformers_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.tokenizer

    @staticmethod
    def _ensure_transformers_compat():
        """Patch APIs removed from newer transformers that DeepSeek MoE remote code still imports.

        deepseek-ai/deepseek-moe-16b-chat modeling_deepseek.py imports
        ``is_torch_fx_available``, which was removed from transformers.utils.import_utils.
        The symbol is only used to optionally FX-wrap an attention-mask helper; returning
        False is safe for our inference path.
        """
        import transformers.utils.import_utils as import_utils

        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = lambda: False

    @staticmethod
    def _rematerialize_rope_buffers(model: torch.nn.Module) -> int:
        """Recompute the rotary embedding caches, which load as uninitialized memory.

        ``DeepseekRotaryEmbedding`` registers ``inv_freq`` / ``cos_cached`` /
        ``sin_cached`` with ``persistent=False`` and fills them in ``__init__``.
        transformers 5.x builds the model on the meta device and only materializes
        tensors that exist in the checkpoint, so these three buffers are allocated
        but never written: they come back as whatever was in memory, including NaN
        and Inf, and differ on every process start.

        ``forward`` also resets ``max_seq_len_cached`` to None after building the
        cache, so the first call rebuilds cos/sin - but from the garbage
        ``inv_freq``, so rebuilding alone does not help.

        The NaNs make CPU SDPA return exact zeros while the device returns finite
        values, so attention diverges in the very first decoder layer and the whole
        comparison is meaningless. Recompute all three from the config.

        Args:
            model: The loaded model whose rotary embedding modules are patched
                   in place.

        Returns:
            int: Number of rotary embedding modules rematerialized.
        """
        n_patched = 0
        for module in model.modules():
            if not (
                hasattr(module, "inv_freq") and hasattr(module, "_set_cos_sin_cache")
            ):
                continue
            inv_freq = 1.0 / (
                module.base
                ** (
                    torch.arange(0, module.dim, 2, dtype=torch.int64).float()
                    / module.dim
                )
            )
            module.register_buffer(
                "inv_freq",
                inv_freq.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype),
                persistent=False,
            )
            # Subclasses (linear / dynamic / yarn scaling) override this and apply
            # their own scaling factor on top of the freshly built inv_freq.
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings,
                device=module.inv_freq.device,
                dtype=next(model.parameters()).dtype,
            )
            n_patched += 1
        return n_patched

    @staticmethod
    def _normalize_rope_scaling(config):
        """Make rope_scaling compatible with DeepSeek MoE remote modeling code.

        transformers 5.x rewrites ``rope_scaling`` into ``rope_parameters`` using
        ``rope_type`` (often injecting ``{"rope_type": "default", ...}`` even when
        the hub config has ``rope_scaling: null``). The remote ``modeling_deepseek.py``
        still checks ``rope_scaling["type"]``, which KeyErrors on the new schema.

        For default / null scaling restore ``None`` so ``_init_rope`` takes the
        unscaled path; otherwise alias ``rope_type`` -> ``type``.

        Args:
            config: The model config, mutated in place when rope_scaling needs
                    rewriting.

        Returns:
            The same config object, for call-site chaining.
        """
        rope = getattr(config, "rope_scaling", None)
        if not isinstance(rope, dict):
            return config

        if "type" in rope:
            return config

        rope_type = rope.get("rope_type", "default")
        if rope_type == "default" and "factor" not in rope:
            config.rope_scaling = None
        else:
            config.rope_scaling = {**rope, "type": rope_type}
        return config

    @staticmethod
    def _patch_moe_infer_min(model: torch.nn.Module) -> None:
        """Make upstream ``moe_infer`` traceable by Dynamo, changing one line.

        Upstream builds the per-expert cumulative offsets as a numpy ndarray

            tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        and then does ``for i, end_idx in enumerate(tokens_per_expert)``. Dynamo
        iterates that ndarray as if it were a tensor and fails with
        ``AttributeError: 'ndarray' object has no attribute 'dim'``. Dropping
        ``.numpy()`` and materializing a plain Python list instead
        (``.cpu().cumsum(0).tolist()``) makes the loop bounds ordinary Python
        ints, which trace cleanly. The ``.cpu()`` is already in upstream and
        syncs the lazy tensor before ``.tolist()``, so the offsets are read
        correctly rather than off an unmaterialized graph.

        Everything else is left exactly as upstream: device-side
        ``argsort`` / ``token_idxs``, in-place ``mul_``, and the un-padded
        ``torch.zeros_like(x)`` accumulation buffer.

        Requires a tt-metal that includes the ``embedding_bw`` zero-init fix
        (tenstorrent/tt-metal#54556, issue #54561). tt-mlir lowers the
        ``scatter_reduce_`` below to ``ttnn.embedding_bw`` with the accumulation
        buffer as the weight table, and that op used to zero only
        ``floor(rows / 32)`` whole tile rows of its output. The buffer here is
        ``seq_len`` rows tall, so any prompt whose length is not a multiple of 32
        left the trailing rows holding stale DRAM that the expert sums were then
        added to. On an equivalent standalone MoE graph (8 experts, top-6, real
        expert MLPs, 9-row buffer) that measured PCC 0.027 against CPU before the
        fix and 0.999986 after; the leaked values depend on what last occupied
        those pages, so the error varies run to run. On an older tt-metal this
        reads as model flakiness - check the runtime before chasing a numerics
        bug in the model.

        ``bincount()`` without ``minlength`` is safe for correctness: experts that
        no token routed to are simply not iterated. It does make the offset list
        length routing-dependent, which can cost extra Dynamo recompiles.

        Args:
            model: The loaded model; every ``DeepseekMoE`` module has its bound
                   ``moe_infer`` replaced in place.
        """

        @torch.no_grad()
        def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
            expert_cache = torch.zeros_like(x)
            idxs = flat_expert_indices.argsort()
            # Only change vs upstream: Python list instead of numpy ndarray.
            tokens_per_expert = flat_expert_indices.bincount().cpu().cumsum(0).tolist()
            token_idxs = idxs // self.num_experts_per_tok
            for i, end_idx in enumerate(tokens_per_expert):
                start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
                if start_idx == end_idx:
                    continue
                expert = self.experts[i]
                exp_token_idx = token_idxs[start_idx:end_idx]
                expert_tokens = x[exp_token_idx]
                expert_out = expert(expert_tokens)
                expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
                expert_cache.scatter_reduce_(
                    0,
                    exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                    expert_out,
                    reduce="sum",
                )
            return expert_cache

        for module in model.modules():
            if type(module).__name__ == "DeepseekMoE":
                module.moe_infer = moe_infer.__get__(module, type(module))

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek MoE-16B model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DeepSeek MoE-16B model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()
        else:
            self._ensure_transformers_compat()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        config = self._normalize_rope_scaling(config)
        # KV cache path in remote modeling_deepseek.py is incompatible with TT
        # compile / graph capture; disable before constructing the model.
        config.use_cache = False

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self._rematerialize_rope_buffers(model)
        self._patch_moe_infer_min(model)
        model.eval()
        model.config.use_cache = False
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DeepSeek MoE-16B model with this instance's variant settings.

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
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style tensor-parallel shard spec.

        deepseek-moe-16b-chat uses plain multi-head attention, not DeepSeek-V2
        MLA, so Q/K/V column-shard over heads and O row-shards. Every expert is
        tensor-parallel like a dense MLP; the router stays replicated.

        Args:
            model: The loaded causal-LM model.

        Returns:
            dict: Weight tensor -> (row_axis, col_axis) mesh axis names.
        """
        shard_specs = {}
        for layer in model.model.layers:
            # deepseek-moe-16b-chat uses standard multi-head attention (q/k/v/o),
            # not DeepSeek-V2 MLA. Column-shard Q/K/V over heads; row-shard O.
            sa = layer.self_attn
            shard_specs[sa.q_proj.weight] = ("model", "batch")
            shard_specs[sa.k_proj.weight] = ("model", "batch")
            shard_specs[sa.v_proj.weight] = ("model", "batch")
            shard_specs[sa.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # Sparse MoE: experts are an nn.ModuleList of DeepseekMLP (separate
                # gate/up/down Linears), not fused 3D gate_up_proj/down_proj.
                # Tensor-parallel each expert like a dense MLP. The router (gate)
                # must see all experts, so it stays replicated.
                for expert in mlp.experts:
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    shard_specs[expert.down_proj.weight] = ("batch", "model")

                shared = getattr(mlp, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = ("model", "batch")
                    shard_specs[shared.up_proj.weight] = ("model", "batch")
                    shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                # Dense MLP (layer 0 only; MoE starts after first_k_dense_replace).
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
