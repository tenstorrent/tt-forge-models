# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 model loader implementation for causal language modeling.

google/gemma-4-12B is a Gemma4UnifiedForConditionalGeneration (any-to-any:
text + vision + audio). This loader brings up the *text-only* causal-LM path
(input_ids + attention_mask only), which is the tractable first target for
single-device bringup. Vision/audio components are out of scope here.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma4 model variants for causal LM."""

    GEMMA_4_12B = "12B"


class ModelLoader(ForgeModel):
    """Gemma4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_12B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-12B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_12B

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Gemma 4",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current causal_lm variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma4 causal_lm model instance (text-only path).

        Args:
            dtype_override: Optional torch dtype to load weights in. The model
                ships in bfloat16; when not provided, transformers uses the
                checkpoint's native dtype.

        Returns:
            torch.nn.Module: The Gemma4 unified model instance, driven through
            its causal language modeling (text) path.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer()
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Gemma4UnifiedForConditionalGeneration.__init__ does not accept
        # ``use_cache`` as a kwarg (unlike most causal-LM heads), so set it on
        # the config — on the text decoder sub-config when present.
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

    def unpack_forward_output(self, fwd_output):
        """Extract the logits tensor from the Gemma4 unified model output.

        Gemma4UnifiedForConditionalGeneration returns a custom
        ``Gemma4UnifiedCausalLMOutputWithPast`` that is not registered in the
        generic handler table, so unpack it explicitly here.
        """
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for FSDP-style TP.

        The Gemma4 unified text decoder is a standard causal-LM stack, so it
        uses Pattern B (batch + model axes). Query heads (and the MLP) are
        sharded on the model axis, so ``num_attention_heads`` must be divisible
        by it. KV projections are left replicated (see ``load_shard_spec``):
        the 8 global layers carry a single global KV head (``attention_k_eq_v``
        with ``num_global_key_value_heads == 1``) that cannot be split across
        the mesh, so no KV-head divisibility constraint is imposed here.
        """
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        n_heads = text_cfg.num_attention_heads
        assert (
            n_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map for the Gemma4 unified text decoder.

        Column-parallel (shard out_features on the model axis) for q_proj and
        the MLP gate/up projections; row-parallel for o_proj and down_proj.

        KV projections are intentionally **replicated** (omitted from the map):
        Gemma4's global ``full_attention`` layers use ``attention_k_eq_v`` so
        ``v_proj is None`` (value reuses key) and carry only a single global KV
        head — a single head cannot be sharded across the model axis, and
        mixing sharded/replicated KV per layer-type is fragile on a first
        compile. Replicating all KV is the standard GQA-TP fallback and keeps
        every query head correctly grouped on each chip. ``k_proj``/``v_proj``
        are therefore skipped (and guarded for absence/None).

        Per-projection RMSNorms (q_norm/k_norm/v_norm), layernorms, embeddings
        and lm_head are left replicated. Vision/audio towers are unused on the
        text-only path and are replicated.
        """
        shard_specs = {}
        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")
            # k_proj/v_proj replicated (skipped). On Gemma4 global layers
            # v_proj is None and k_proj holds a single unsharddable KV head.
        return shard_specs

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample text inputs for the Gemma4 model.

        Returns a dict (not a list) so the harness passes the tensors as
        keyword arguments. Gemma4UnifiedForConditionalGeneration.forward has
        ``pixel_values`` as its second positional parameter, so a positional
        ``[input_ids, attention_mask]`` would bind the attention mask to
        ``pixel_values`` and wrongly trigger the vision tower. Keying by name
        keeps ``pixel_values`` None and stays on the text-only path.

        Returns:
            dict: {"input_ids", "attention_mask"} tensors for the text path.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()
        # google/gemma-4-12B is a base (non-instruct) checkpoint and ships
        # without a chat template, so fall back to plain prompt text when no
        # template is available.
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
