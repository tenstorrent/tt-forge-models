# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 MoE loader (text decoder, causal-LM / prefill path).

google/gemma-4-26B-A4B-it is a sparse-MoE image-text-to-text model
(``Gemma4ForConditionalGeneration``). This loader drives the **text decoder**
only: it loads the full model, then wraps ``model.model.language_model`` +
``model.lm_head`` in a thin module so:

  * the multimodal merge glue in ``Gemma4Model.forward`` is bypassed (a clean
    text-only prefill graph), and
  * the wrapper's ``.config`` **is** the ``text_config`` object, so the runner's
    ``inject_custom_moe`` (which sets ``config._experts_implementation``) reaches
    the ``Gemma4TextExperts`` modules (they are built from ``text_config``).

Architecture (text_config): hidden 2816, 30 layers, GQA 16q/8kv head_dim 256,
dense MLP intermediate 2112 (runs every layer), **MoE every layer**: 128 experts
top-8, expert intermediate 704, stored as 3D batched tensors
(``experts.gate_up_proj`` [E, 2*704, H], ``experts.down_proj`` [E, H, 704]).
Experts are 88% of the 25.8B params. The data-dependent expert-loop forward is
replaced on device by the ``tt_moe`` expert-parallel backend
(``inject_custom_moe: true`` in the test config).
"""

from typing import Optional

import torch
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
from ....tools.utils import cast_input_to_type, pad_inputs


class ModelVariant(StrEnum):
    """Available Gemma 4 MoE variants."""

    GEMMA_4_26B_A4B = "26B-A4B"


class _Gemma4TextDecoder(torch.nn.Module):
    """Thin text-decoder wrapper: ``language_model`` + ``lm_head`` -> logits.

    Bypasses the multimodal merge; exposes ``.config == text_config`` so
    ``inject_custom_moe`` dispatches the ``tt_moe`` experts backend correctly.
    """

    def __init__(self, full_model):
        super().__init__()
        self.model = full_model.model.language_model  # Gemma4TextModel (.layers)
        self.lm_head = full_model.lm_head
        self.config = full_model.config.text_config
        self.config.use_cache = False

    def tie_weights(self):
        # Gemma 4 ties lm_head to the token embeddings; re-tie after .to(device).
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        return self.lm_head(hidden)


class ModelLoader(ForgeModel):
    """Gemma 4 MoE text-decoder loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B

    sample_text = "Why is the sky blue?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.seq_len = None
        self.num_layers = num_layers
        # EP axis is chosen in get_mesh_config so load_shard_spec stays consistent
        # with the tt_moe backend's auto cluster axis (first mesh axis with size>1).
        self._expert_mesh_axis = "model"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma4-MoE",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Load the Gemma 4 MoE text decoder as a clean prefill wrapper."""
        name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        layers = num_layers if num_layers is not None else self.num_layers
        if layers is not None:
            config = AutoConfig.from_pretrained(name)
            text_cfg = getattr(config, "text_config", config)
            text_cfg.num_hidden_layers = layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        full = AutoModelForCausalLM.from_pretrained(name, **model_kwargs).eval()
        full.config.use_cache = False
        if hasattr(full.config, "text_config"):
            full.config.text_config.use_cache = False

        decoder = _Gemma4TextDecoder(full).eval()
        self.config = decoder.config
        self.model = decoder
        return decoder

    def load_inputs(self, dtype_override=None, batch_size=1, prompt: Optional[str] = None):
        """Text-only prefill inputs (batch=1 single user), padded to max_length."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(prompt or self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len
        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask

        # Keep only the decoder's forward kwargs.
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        cfg = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        self.config = getattr(cfg, "text_config", cfg)
        return self.config

    def get_mesh_config(self, num_devices: int):
        """Megatron TP x expert-parallel mesh for the 4-chip qb2.

        User-requested layout is tp=2, sp=2 -> a (2, 2) mesh: the "model" axis is
        tensor-parallel (attention + dense MLP); the "batch" axis carries the
        second parallelism dimension, used here as **expert parallelism** for the
        sparse MoE (128 experts are 88% of the params, so they must be
        distributed). The tt_moe backend dispatches along a single cluster axis
        (the first mesh axis with size>1), so the experts are sharded on exactly
        that axis to stay consistent. Falls back to (1, N) for odd counts.
        """
        cfg = self.config if self.config is not None else self.load_config()
        text_cfg = getattr(cfg, "text_config", cfg)

        if num_devices % 2 == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            mesh_shape = (1, num_devices)

        # Expert-parallel axis == tt_moe cluster axis == first axis with size>1.
        if mesh_shape[0] > 1:
            self._expert_mesh_axis = "batch"
            ep = mesh_shape[0]
        else:
            self._expert_mesh_axis = "model"
            ep = mesh_shape[1]

        tp = mesh_shape[1]
        assert text_cfg.num_attention_heads % tp == 0, (
            f"num_attention_heads={text_cfg.num_attention_heads} not divisible by tp={tp}"
        )
        assert text_cfg.num_key_value_heads % tp == 0, (
            f"num_key_value_heads={text_cfg.num_key_value_heads} not divisible by tp={tp}"
        )
        assert text_cfg.num_experts % ep == 0, (
            f"num_experts={text_cfg.num_experts} not divisible by ep={ep}"
        )
        return mesh_shape, ("batch", "model")

    def _megatron_shard_spec(self, model, batch_axis="batch"):
        """Megatron column->row FFN sharding + expert-parallel MoE sharding.

        Dense MLP (runs every layer): gate/up column-sharded on "model" (tp),
        down row-sharded -> Megatron tensor parallelism on the model axis.
        MoE experts: expert dim sharded on ``self._expert_mesh_axis`` (== the
        tt_moe cluster axis) for expert parallelism.

        Attention is **replicated**: this Gemma-4 MoE uses GQA with an
        alternative-attention variant where V shares K's projection (``v_proj``
        is ``None`` on the full-attention layers) and later layers share KV, and
        head-sharding Q crashes the ``repeat_kv`` reshard under Shardy (the same
        limitation the sibling diffusiongemma loader documents). Attention is a
        small fraction of the params, so replication is cheap and correct.
        lm_head/embeddings are tied and left replicated.
        """
        expert_axis = self._expert_mesh_axis
        shard_specs = {}

        base = model.model  # Gemma4TextModel
        for layer in base.layers:
            # Dense shared MLP -> tensor-parallel (Megatron column->row).
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and getattr(mlp, "gate_proj", None) is not None:
                shard_specs[mlp.gate_proj.weight] = ("model", None)
                shard_specs[mlp.up_proj.weight] = ("model", None)
                shard_specs[mlp.down_proj.weight] = (None, "model")

            # Sparse MoE experts -> expert-parallel on the cluster axis.
            experts = getattr(layer, "experts", None)
            if experts is not None and getattr(experts, "gate_up_proj", None) is not None:
                # 3D batched experts: [E, 2I, H] / [E, H, I]; shard the expert dim.
                shard_specs[experts.gate_up_proj] = (expert_axis, None, None)
                shard_specs[experts.down_proj] = (expert_axis, None, None)

        return shard_specs

    def load_shard_spec(self, model):
        """Default (non-prefill TP) shard spec: Megatron + expert parallelism."""
        return self._megatron_shard_spec(model, batch_axis="batch")


class ModelLoaderPrefill(ModelLoader, ForgePrefillModel):
    """Prefill-focused loader for the multi-chip tensor-parallel prefill path.

    Exposes the strategy-parameterized ``load_shard_spec(model, strategy,
    batch_axis)`` the runner uses to sweep mesh / strategy / seq / batch.
    """

    # A ~full-length prompt so a seq_len=128 prefill is mostly real tokens (little
    # padding) -- padding positions otherwise dominate and depress the PCC.
    prefill_text = (
        "The sky appears blue because of a phenomenon called Rayleigh scattering. "
        "As sunlight reaches Earth's atmosphere, it collides with gas molecules "
        "that scatter shorter blue wavelengths far more strongly than longer red "
        "ones, so blue light is redirected across the sky in every direction. "
        "When we look up, we see this scattered blue light coming from all around. "
        "Near sunrise and sunset the light travels through much more atmosphere, "
        "the blue is scattered away, and the remaining reds and oranges dominate "
        "the sky, which is why sunsets glow with warm colors rather than blue."
    )

    def load_inputs_prefill(self, dtype_override=None, batch_size=1, seq_len=128):
        """Prefill inputs (single user) padded to ``seq_len``."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.prefill_text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        padded_input_ids, _ = pad_inputs(inputs["input_ids"], seq_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], seq_len)
        self.seq_len = seq_len
        result = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        if dtype_override is not None:
            for key in result:
                result[key] = cast_input_to_type(result[key], dtype_override)

        return result

    def load_shard_spec(self, model, strategy="megatron", batch_axis="batch"):
        """Weight shard spec parameterized by ``strategy`` / ``batch_axis``.

        Only ``megatron`` is meaningful for this MoE (attention/dense are
        tensor-parallel; experts are expert-parallel across the cluster axis).
        An ``fsdp`` alias maps to the same spec.
        """
        if strategy not in ("megatron", "fsdp"):
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")
        # Experts are expert-parallel on the COMPOUND (batch, model) mesh axis --
        # the same layout the tt_moe backend's own get_tt_moe_shard_specs uses.
        # On a (1, N) mesh (batch size 1) this collapses to N-way EP on the model
        # axis == the tt_moe cluster axis, which is the supported/proven topology.
        # (A 2x2 mesh compiles but the tt_moe reshape is not supported when TP and
        # EP ride different axes -- see the bringup report.)
        self._expert_mesh_axis = (batch_axis, "model")
        return self._megatron_shard_spec(model, batch_axis=batch_axis)
