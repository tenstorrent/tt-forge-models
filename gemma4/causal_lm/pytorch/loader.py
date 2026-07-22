# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 26B-A4B causal-LM (text) loader.

``google/gemma-4-26B-A4B-it`` is a ``Gemma4ForConditionalGeneration``
any-to-any (text + vision) model. This loader brings up the **text-only
causal-LM prefill path** (``input_ids`` + ``attention_mask`` only); the vision
tower is instantiated but never entered because ``load_inputs`` returns a dict
keyed by name, leaving ``pixel_values`` unset.

Architecture (text decoder, ``text_config``):
    * 30 decoder layers, hidden 2816, tied embeddings, vocab 262144.
    * Attention is heterogeneous by layer type: 25 ``sliding_attention`` layers
      (16 q-heads / 8 kv-heads, head_dim 256) and 5 ``full_attention`` layers
      (16 q-heads / 2 kv-heads, head_dim 512, ``k == v`` so ``v_proj is None``).
    * Each layer carries BOTH a dense MLP (SwiGLU-ish, intermediate 2112) and a
      128-expert top-8 MoE block with fused expert weights
      (``experts.gate_up_proj`` ``(128, 1408, 2816)`` and
      ``experts.down_proj`` ``(128, 2816, 704)``).

``ModelLoaderPrefill`` adds the strategy-parameterized ``load_shard_spec`` used
by the prefill TP path (Megatron column->row + MoE intermediate sharding, plus
an FSDP variant). See ``load_shard_spec`` for the layout and the divisibility
guarantees at the 2x2 (TP=2, "SP"=2) mesh.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel, ForgePrefillModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma4 causal-LM variants."""

    GEMMA_4_26B_A4B_IT = "26B-A4B-it"


class ModelLoader(ForgeModel):
    """Gemma4 26B-A4B causal-LM (text) loader."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B_IT

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Gemma 4",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Gemma4 model, driven through its text-only causal-LM path.

        Args:
            dtype_override: Optional torch dtype for the weights. The checkpoint
                ships in bfloat16.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Gemma4ForConditionalGeneration.__init__ does not accept ``use_cache``
        # as a kwarg, so set it on the config (on the text sub-config too).
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False
        if hasattr(config, "text_config"):
            config.text_config.use_cache = False
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def unpack_forward_output(self, fwd_output):
        """Gemma4ForConditionalGeneration returns a custom output type not in
        the generic handler table; unpack its logits explicitly."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Text-only causal-LM inputs. Returns a dict so the tensors bind by
        name and ``pixel_values`` stays unset (text path only)."""
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()
        self.tokenizer.padding_side = "right"

        if getattr(self.tokenizer, "chat_template", None):
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt or self.sample_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            input_text = prompt or self.sample_text

        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    def get_mesh_config(self, num_devices: int):
        """Default 1-D mesh ``(1, num_devices)`` on ``("batch", "model")``.

        The strategy-aware prefill path overrides the mesh from test metadata
        (e.g. ``(2, 2)`` for TP=2 + input-sharding); this default covers the
        non-prefill TP path. Query heads are sharded on the model axis, so
        ``num_attention_heads`` must divide it.
        """
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        assert (
            text_cfg.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def _shard_spec_for_layers(self, model, strategy, batch_axis):
        """Shared Megatron/FSDP shard map over the text decoder layers.

        Column-parallel (shard out_features on ``model``) for q/k/v_proj, dense
        MLP gate/up, and the fused-expert intermediate; row-parallel for o_proj,
        dense down_proj, and the fused-expert output. Embeddings, lm_head (tied),
        RMSNorms, per-head q/k norms and the router are left replicated. On the
        ``full_attention`` layers ``v_proj is None`` (value reuses key) and is
        skipped.

        strategy:
            "megatron" -> weights sharded on the ``model`` axis only (non-model
                dim is ``None``); ``batch_axis`` is ignored.
            "fsdp"     -> the non-model dim is additionally sharded on
                ``batch_axis`` (the data axis), and the fused-expert num_experts
                dim (axis 0) is sharded on ``batch_axis``.
        """
        other = None if strategy == "megatron" else batch_axis
        expert_leader = None if strategy == "megatron" else batch_axis

        shard_specs = {}
        for layer in model.model.language_model.layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", other)
            shard_specs[attn.k_proj.weight] = ("model", other)
            v_proj = getattr(attn, "v_proj", None)
            if v_proj is not None and v_proj.weight is not None:
                shard_specs[v_proj.weight] = ("model", other)
            shard_specs[attn.o_proj.weight] = (other, "model")

            # Dense MLP (present on every layer alongside the MoE block).
            shard_specs[layer.mlp.gate_proj.weight] = ("model", other)
            shard_specs[layer.mlp.up_proj.weight] = ("model", other)
            shard_specs[layer.mlp.down_proj.weight] = (other, "model")

            # Fused MoE experts: gate_up_proj (E, 2*I, H) column-parallel on the
            # intermediate axis (1); down_proj (E, H, I) row-parallel on the
            # intermediate axis (2). Under FSDP the expert axis (0) is sharded
            # on the data axis as well.
            shard_specs[layer.experts.gate_up_proj] = (expert_leader, "model", None)
            shard_specs[layer.experts.down_proj] = (expert_leader, None, "model")
        return shard_specs

    def load_shard_spec(self, model):
        """Default (non-prefill) Megatron TP map on a ``("batch", "model")`` mesh."""
        return self._shard_spec_for_layers(model, strategy="megatron", batch_axis="batch")


class ModelLoaderPrefill(ModelLoader, ForgePrefillModel):
    """Prefill-focused loader: strategy-parameterized weight shard specs for the
    prefill TP sweep (meshes / strategies / batch / sequence length)."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B_IT: ModelLoader._VARIANTS[
            ModelVariant.GEMMA_4_26B_A4B_IT
        ],
    }
    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B_IT

    def load_shard_spec(self, model, strategy="megatron", batch_axis="batch"):
        if strategy not in ("megatron", "fsdp"):
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")
        return self._shard_spec_for_layers(model, strategy=strategy, batch_axis=batch_axis)
