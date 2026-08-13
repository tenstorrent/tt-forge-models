# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AllenAI Molmo 72B 0924 model loader implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import CausalLMOutputWithPast
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


def _patch_molmo_remote_code(pretrained_model_name: str) -> None:
    """Patch Molmo remote code for current transformers + torch.compile.

    - tie_weights: newer transformers passes kwargs the remote no-op rejects.
    - ModelOutput: bare ModelOutput drops ``.logits`` under torch.compile; return
      CausalLMOutputWithPast from Molmo and adapt MolmoForCausalLM accordingly.
    """
    causal_cls = get_class_from_dynamic_module(
        "modeling_molmo.MolmoForCausalLM",
        pretrained_model_name,
    )
    molmo_cls = get_class_from_dynamic_module(
        "modeling_molmo.Molmo",
        pretrained_model_name,
    )

    if not getattr(causal_cls.tie_weights, "_molmo_kwargs_patched", False):

        def tie_weights(self, *args, **kwargs):
            return None

        tie_weights._molmo_kwargs_patched = True
        causal_cls.tie_weights = tie_weights

    if getattr(molmo_cls.forward, "_molmo_compile_patched", False):
        return

    orig_molmo_forward = molmo_cls.forward

    def molmo_forward(self, *args, **kwargs):
        out = orig_molmo_forward(self, *args, **kwargs)
        return CausalLMOutputWithPast(
            logits=out["logits"],
            past_key_values=out["attn_key_values"],
            hidden_states=out["hidden_states"],
        )

    molmo_forward._molmo_compile_patched = True
    molmo_cls.forward = molmo_forward

    def causal_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        attention_bias=None,
        response_mask=None,
        images=None,
        image_masks=None,
        image_input_idx=None,
        subsegment_ids=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        loss_masks=None,
        use_cache=None,
        last_logits_only=None,
        output_attentions=None,
        output_hidden_states=None,
        append_last_valid_logits=None,
        return_dict=None,
        cache_position=None,
    ):
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in Molmo")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.embedding_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, outputs.past_key_values, hidden_states)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    causal_forward._molmo_compile_patched = True
    causal_cls.forward = causal_forward


class _SplitSwiGLUFeedForwardProj(nn.Module):
    """Drop-in replacement for Molmo's fused SwiGLU ``ff_proj`` projection.

    Molmo packs the SwiGLU up/gate projections into a single ``ff_proj`` Linear
    whose output is later cut by ``SwiGLU.chunk(2, dim=-1)`` into
    ``x=up`` (first half) and ``gate`` (second half), returning
    ``F.silu(gate) * up``.

    Under tensor parallel that fused weight is column-sharded contiguously, so
    the ``up`` half lands on one set of devices and the ``gate`` half on
    another; the per-shard ``chunk`` then mis-pairs up/gate on every device and
    PCC collapses. Splitting into two independent Linears makes each a clean
    column-parallel matmul (matching every other dense SwiGLU model in the repo)
    and turns the SwiGLU multiply into an elementwise op over the already
    identically-sharded intermediate dim. This module folds the SwiGLU
    activation in, so the block's separate ``act`` is replaced with Identity.
    """

    def __init__(self, up: nn.Linear, gate: nn.Linear):
        super().__init__()
        self.up = up
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.up(x)


def _split_molmo_ff_proj(model) -> None:
    """Unfuse each block's SwiGLU ``ff_proj`` into separate up/gate Linears.

    Numerically identical to the original fused ``ff_proj`` + ``SwiGLU``, but TP
    safe (see ``_SplitSwiGLUFeedForwardProj``). Idempotent and applied to every
    transformer block; the biggest weight in the model stays sharded, so there
    is no memory regression.
    """
    for block in model.model.transformer.blocks:
        ff_proj = block.ff_proj
        if isinstance(ff_proj, _SplitSwiGLUFeedForwardProj):
            continue

        out_features, in_features = ff_proj.weight.shape
        half = out_features // 2  # SwiGLU: up = first half, gate = second half.
        has_bias = ff_proj.bias is not None
        weight = ff_proj.weight
        requires_grad = weight.requires_grad

        def _make_linear(row_slice: slice) -> nn.Linear:
            linear = nn.Linear(
                in_features,
                half,
                bias=has_bias,
                dtype=weight.dtype,
                device=weight.device,
            )
            linear.weight = nn.Parameter(
                weight[row_slice].detach().clone(), requires_grad=requires_grad
            )
            if has_bias:
                linear.bias = nn.Parameter(
                    ff_proj.bias[row_slice].detach().clone(),
                    requires_grad=requires_grad,
                )
            return linear

        up = _make_linear(slice(0, half))
        gate = _make_linear(slice(half, out_features))

        block.ff_proj = _SplitSwiGLUFeedForwardProj(up, gate)
        # The SwiGLU activation is now folded into ff_proj.
        block.act = nn.Identity()


class ModelVariant(StrEnum):
    """Available AllenAI Molmo 72B 0924 model variants for causal language modeling."""

    MOLMO_72B_0924 = "Molmo-72B-0924"


class ModelLoader(ForgeModel):
    """AllenAI Molmo 72B 0924 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MOLMO_72B_0924: LLMModelConfig(
            pretrained_model_name="allenai/Molmo-72B-0924",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO_72B_0924

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
            model="molmo",
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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AllenAI Molmo 72B 0924 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The AllenAI Molmo 72B 0924 model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        _patch_molmo_remote_code(pretrained_model_name)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        # Prefill-only path: BufferCache KV/RoPE paths are not TP-safe under compile.
        model.config.use_cache = False
        # Unfuse the SwiGLU ff_proj so it is safe to column-shard under TP
        # (numerically identical for single-device runs).
        _split_molmo_ff_proj(model)
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AllenAI Molmo 72B 0924 model with this instance's variant settings.

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
        """Tensor-parallel shard spec for Molmo.

        Attention input projection (``att_proj``) is fused GQA ``[Q|K|V]`` with
        unequal widths (Q=d_model, K=V=n_kv*head_dim). Naive column-TP on that
        fused weight mis-partitions Q/K/V across devices and collapses PCC, so
        it (and the vocab embedding) stay replicated. MLP + attn out + LM head
        use standard Megatron column/row splits.

        ``ff_proj`` is Molmo's fused SwiGLU up/gate projection; it is unfused
        into separate ``up``/``gate`` Linears at load time (see
        ``_split_molmo_ff_proj``) so each can be column-sharded independently
        instead of being cut on a sharded axis by the SwiGLU ``chunk``.
        """
        molmo = model.model
        shard_specs = {}

        for block in molmo.transformer.blocks:
            # Replicate fused att_proj; row-parallel attn output.
            shard_specs[block.attn_out.weight] = ("batch", "model")

            # Unfused SwiGLU: column-parallel up/gate, then row-parallel ff_out.
            shard_specs[block.ff_proj.up.weight] = ("model", "batch")
            shard_specs[block.ff_proj.gate.weight] = ("model", "batch")
            shard_specs[block.ff_out.weight] = ("batch", "model")

        # LM head when weight tying is disabled (Molmo-72B).
        if hasattr(molmo.transformer, "ff_out"):
            shard_specs[molmo.transformer.ff_out.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
