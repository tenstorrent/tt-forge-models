# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gpt-oss model loader implementation for causal language modeling tasks.
"""
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils.quantization_config import Mxfp4Config
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available gpt-oss model variants."""

    GPT_OSS_20B = "20B"
    GPT_OSS_120B = "120B"


class _ReplayModel(torch.nn.Module):
    """Runs a single decoder layer with frozen input for isolated layer testing.

    Accepts the same (input_ids, attention_mask) interface as the full model
    so it can be compiled and sharded with the same infrastructure.

    Modes:
        "full"              - Run the entire decoder layer (attention + MoE).
        "attention_only"    - Run only input_layernorm → self_attn → residual add.
                              Also captures the post-attention hidden state to
                              ``_post_attn_capture_path`` (CPU only) so that a
                              subsequent ``moe_only`` run can use it as frozen input.
        "moe_only"          - Run only post_attention_layernorm → mlp → residual add.
                              ``frozen_hidden_states`` should be the post-attention
                              state captured by a prior ``attention_only`` run.
        "attention_no_sink" - Same as attention_only but manually reimplements the
                              attention forward with standard softmax, bypassing
                              the GPT-OSS sink-token mechanism (concat → softmax
                              → drop). Compare PCC with attention_only to isolate
                              whether sink tokens cause precision loss.
        "attention_fp32"    - Same as attention_only but casts the entire self_attn
                              module to float32 (weights + computation). Tests
                              whether bf16 precision in QK^T, RoPE, or attention@V
                              causes the PCC drop.
    """

    VALID_MODES = (
        "full", "attention_only", "moe_only",
        "attention_no_sink", "attention_fp32",
        "attn_proj_only", "attn_qkt_only", "attn_softmax_only",
        "attn_v_only", "attention_manual",
    )

    def __init__(self, decoder_layer, norm, lm_head, rotary_emb,
                 frozen_hidden_states, config=None, mode="full"):
        super().__init__()
        assert mode in self.VALID_MODES, (
            f"Invalid replay mode '{mode}', must be one of {self.VALID_MODES}"
        )
        self.layer = decoder_layer
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.config = config
        self.mode = mode
        self.frozen_input = torch.nn.Parameter(
            frozen_hidden_states, requires_grad=False
        )
        # Set by ModelLoader for attention_only mode to save intermediate state
        self._post_attn_capture_path = None
        # Set by ModelLoader for attn_v_only mode
        self._attn_intermediates_path = None

    def _prepare_attention_inputs(self, hidden_states, attention_mask):
        """Build position_ids, causal mask and rotary embeddings."""
        batch_size, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)
        cache_position = torch.arange(
            seq_len, device=hidden_states.device, dtype=torch.long
        )
        mask_kwargs = {
            "config": self.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        layer_attention_mask = causal_mask_mapping[self.layer.attention_type]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return position_ids, layer_attention_mask, position_embeddings

    @torch.compiler.disable  # Avoid dynamo fake-tensor shape mismatch in rotary (capture vs workload)
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.mode == "attention_only":
            return self._forward_attention_only(attention_mask)
        elif self.mode == "moe_only":
            return self._forward_moe_only()
        elif self.mode == "attention_no_sink":
            return self._forward_attention_no_sink(attention_mask)
        elif self.mode == "attention_fp32":
            return self._forward_attention_fp32(attention_mask)
        elif self.mode == "attn_proj_only":
            return self._forward_attn_proj_only(attention_mask)
        elif self.mode == "attn_qkt_only":
            return self._forward_attn_qkt_only(attention_mask)
        elif self.mode == "attn_softmax_only":
            return self._forward_attn_softmax_only(attention_mask)
        elif self.mode == "attn_v_only":
            return self._forward_attn_v_only()
        elif self.mode == "attention_manual":
            return self._forward_attention_manual(attention_mask)
        else:
            return self._forward_full(attention_mask)

    # ------------------------------------------------------------------ full
    def _forward_full(self, attention_mask):
        hidden_states = self.frozen_input
        position_ids, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        layer_out = self.layer(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # -------------------------------------------------------- attention_only
    def _forward_attention_only(self, attention_mask):
        hidden_states = self.frozen_input
        position_ids, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        # Attention sublayer (mirrors decoder layer internals)
        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)
        attn_out = self.layer.self_attn(
            hidden_states=normed,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = attn_out[0] if isinstance(attn_out, (tuple, list)) else attn_out
        hidden_states = residual + hidden_states

        # Capture post-attention state (CPU golden run only) for moe_only replay
        if hidden_states.device.type == "cpu" and self._post_attn_capture_path:
            torch.save(hidden_states.detach().clone(), self._post_attn_capture_path)
            print(f"[capture] Saved post-attention state ({hidden_states.shape}) "
                  f"→ {self._post_attn_capture_path}")

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # -------------------------------------------------------- attention_fp32
    def _forward_attention_fp32(self, attention_mask):
        """Entire forward in float32 (model must be .float()'d by create_replay_model)."""
        hidden_states = self.frozen_input  # already f32
        position_ids, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        attn_out = self.layer.self_attn(
            hidden_states=normed,
            attention_mask=layer_attention_mask.float()
                if layer_attention_mask is not None else None,
            position_ids=position_ids,
            position_embeddings=tuple(p.float() for p in position_embeddings),
        )
        hidden_states = attn_out[0] if isinstance(attn_out, (tuple, list)) else attn_out
        hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # -------------------------------------------------------- attn_proj_only
    def _forward_attn_proj_only(self, attention_mask):
        """Only projection matmuls: layernorm → q_proj → o_proj → residual.

        Tests whether the Linear matmul kernels on device introduce precision
        loss, without any attention score computation (QK^T, softmax, attn@V).
        """
        hidden_states = self.frozen_input
        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        # q_proj: [batch, seq, hidden] → [batch, seq, num_heads * head_dim]
        q = self.layer.self_attn.q_proj(normed)
        # o_proj: [batch, seq, num_heads * head_dim] → [batch, seq, hidden]
        out = self.layer.self_attn.o_proj(q)

        hidden_states = residual + out
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # --------------------------------------------------------- attn_qkt_only
    def _forward_attn_qkt_only(self, attention_mask):
        """Q/K/V proj → RoPE → QK^T → reduce → O proj (skip softmax & attn@V).

        Isolates the QK^T batched matmul. If PCC drops here vs attn_proj_only,
        the QK^T matmul kernel is the precision bottleneck.
        """
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

        hidden_states = self.frozen_input
        _, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        self_attn = self.layer.self_attn
        input_shape = normed.shape[:-1]  # (batch, seq_len)
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        # Q/K/V proj + reshape
        query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        # GQA repeat
        n_rep = self_attn.num_key_value_groups
        if n_rep > 1:
            b, nkv, s, hd = key_states.shape
            key_states = key_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)

        # ★ QK^T — this is the op under test ★
        # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq]
        # → [batch, heads, seq, seq]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self_attn.scaling

        # Reduce [batch, heads, seq, seq] → [batch, seq, heads*head_dim]
        # mean over last dim (key seq), then transpose heads back
        reduced = attn_weights.mean(dim=-1)  # [batch, heads, seq]
        # Expand to head_dim so o_proj gets the right input shape
        reduced = reduced.unsqueeze(-1).expand_as(query_states)  # [batch, heads, seq, head_dim]
        reduced = reduced.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        reduced = reduced.reshape(*input_shape, -1)  # [batch, seq, heads*head_dim]

        # O proj
        out = self_attn.o_proj(reduced)

        hidden_states = residual + out
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ----------------------------------------------------- attn_softmax_only
    def _forward_attn_softmax_only(self, attention_mask):
        """QK^T → mask → softmax+sink → reduce (skip attn@V).

        Same as eager_attention_forward up through softmax+sink+drop,
        then reduces with mean instead of matmul with V.
        If PCC drops here vs attn_qkt_only → softmax+sink is the bottleneck.
        """
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

        hidden_states = self.frozen_input
        _, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        self_attn = self.layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        n_rep = self_attn.num_key_value_groups
        if n_rep > 1:
            b, nkv, s, hd = key_states.shape
            key_states = key_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)

        # QK^T
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self_attn.scaling

        # Causal mask
        if layer_attention_mask is not None:
            causal_mask = layer_attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # V states (needed for attn_v_only capture)
        value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        if n_rep > 1:
            b2, nkv2, s2, hd2 = value_states.shape
            value_states = value_states[:, :, None, :, :].expand(b2, nkv2, n_rep, s2, hd2) \
                .reshape(b2, nkv2 * n_rep, s2, hd2)

        # ★ Softmax WITH sink (mirrors eager_attention_forward) ★
        sinks = self_attn.sinks.reshape(1, -1, 1, 1).expand(
            query_states.shape[0], -1, query_states.shape[-2], -1,
        )
        combined = torch.cat([attn_weights, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = torch.nn.functional.softmax(combined, dim=-1, dtype=combined.dtype)
        scores = probs[..., :-1]  # drop sink dim

        # Capture softmax scores + V for attn_v_only mode (CPU only)
        if scores.device.type == "cpu" and self._attn_intermediates_path:
            torch.save({
                "scores": scores.detach().clone(),
                "value_states": value_states.detach().clone(),
            }, self._attn_intermediates_path)
            print(f"[capture] Saved attn intermediates (scores={scores.shape}, "
                  f"V={value_states.shape}) → {self._attn_intermediates_path}")

        # Reduce [batch, heads, seq, seq] → [batch, seq, heads*head_dim]
        reduced = scores.mean(dim=-1)  # [batch, heads, seq]
        reduced = reduced.unsqueeze(-1).expand_as(query_states)
        reduced = reduced.transpose(1, 2).contiguous()
        reduced = reduced.reshape(*input_shape, -1)

        out = self_attn.o_proj(reduced)
        hidden_states = residual + out
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ----------------------------------------------------------- attn_v_only
    def _forward_attn_v_only(self):
        """Only attn_weights @ V matmul + O proj (frozen scores & V as input).

        Captures from attn_softmax_only must exist. frozen_scores and
        frozen_value_states are set as nn.Parameters by create_replay_model.
        """
        self_attn = self.layer.self_attn
        residual = self.frozen_input
        input_shape = residual.shape[:-1]  # (batch, seq_len)

        # ★ The single op under test ★
        # scores: [batch, heads, seq, seq]  @  V: [batch, heads, seq, head_dim]
        attn_output = torch.matmul(self.frozen_scores, self.frozen_value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # O proj
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ------------------------------------------------------- attention_manual
    def _forward_attention_manual(self, attention_mask):
        """Full attention WITH sinks, manually decomposed (no self_attn module call).

        Identical computation to attention_only but every op is explicit.
        If PCC ~0.94 here too → cumulative bf16 error amplification.
        If PCC 0.99+ → compiler graph fusion/optimization is the cause.
        """
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

        hidden_states = self.frozen_input
        _, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        self_attn = self.layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        # Q/K/V proj
        query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        # GQA repeat
        n_rep = self_attn.num_key_value_groups
        if n_rep > 1:
            b, nkv, s, hd = key_states.shape
            key_states = key_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)
            value_states = value_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)

        # QK^T
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self_attn.scaling

        # Causal mask
        if layer_attention_mask is not None:
            causal_mask = layer_attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax WITH sink (identical to eager_attention_forward)
        sinks = self_attn.sinks.reshape(1, -1, 1, 1).expand(
            query_states.shape[0], -1, query_states.shape[-2], -1,
        )
        combined = torch.cat([attn_weights, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = torch.nn.functional.softmax(combined, dim=-1, dtype=combined.dtype)
        scores = probs[..., :-1]  # drop sink dim

        # Attention @ V
        attn_output = torch.matmul(scores, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # O proj
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self_attn.o_proj(attn_output)

        # Residual + norm + lm_head
        hidden_states = residual + attn_output
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ------------------------------------------------------ attention_no_sink
    def _forward_attention_no_sink(self, attention_mask):
        """Attention with standard softmax — sink token mechanism bypassed."""
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

        hidden_states = self.frozen_input
        _, layer_attention_mask, position_embeddings = \
            self._prepare_attention_inputs(hidden_states, attention_mask)

        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        # --- Manual attention forward (mirrors GptOssAttention.forward) ---
        self_attn = self.layer.self_attn
        input_shape = normed.shape[:-1]  # (batch, seq_len)
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        # Q/K/V projections → (batch, heads, seq, head_dim)
        query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        # GQA: repeat K/V heads to match Q head count
        n_rep = self_attn.num_key_value_groups
        if n_rep > 1:
            b, nkv, s, hd = key_states.shape
            key_states = key_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)
            value_states = value_states[:, :, None, :, :].expand(b, nkv, n_rep, s, hd) \
                .reshape(b, nkv * n_rep, s, hd)

        # QK^T * scaling
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self_attn.scaling

        # Causal mask
        if layer_attention_mask is not None:
            causal_mask = layer_attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ★ Standard softmax — NO sink concat/drop ★
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=attn_weights.dtype,
        )

        # Attention @ V
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # O projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self_attn.o_proj(attn_output)

        # Residual + norm + lm_head
        hidden_states = residual + attn_output
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ------------------------------------------------------------- moe_only
    def _forward_moe_only(self):
        # frozen_input is the post-attention state (after 1st residual add)
        hidden_states = self.frozen_input

        # MoE sublayer (mirrors decoder layer internals)
        residual = hidden_states
        normed = self.layer.post_attention_layernorm(hidden_states)
        mlp_out = self.layer.mlp(normed)
        hidden_states = mlp_out[0] if isinstance(mlp_out, (tuple, list)) else mlp_out
        hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


class ModelLoader(ForgeModel):
    """gpt-oss model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GPT_OSS_20B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-20b",
            max_length=256,
        ),
        ModelVariant.GPT_OSS_120B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-120b",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B

    # Sample messages for inference
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = None,
        moe_fp32: bool = False,
        capture_after_layer: Optional[int] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
            moe_fp32: If True, MoE (A2aSparseMLP) parameters and computation run in fp32
                      while the rest of the model stays in bf16. In replay mode
                      (GPT_OSS_REPLAY_LAYER set), fp32 is applied only to the replay target layer
                      to reduce CPU DRAM usage.
            capture_after_layer: If set, registers a forward hook on this layer index
                      (0-based) to capture its output hidden_states during CPU inference.
                      Use with create_replay_model() to test a subsequent layer in isolation.
        """
        super().__init__(variant)
        self.config = None
        self.tokenizer = None
        self.num_layers = num_layers
        self.moe_fp32 = moe_fp32
        self.capture_after_layer = capture_after_layer
        self.captured_output = None
        self.mlp_type = "sparse"

        # Environment variable overrides for use with existing test runners
        # (where __init__ kwargs can't be passed through).
        #   GPT_OSS_NUM_LAYERS=3
        #   GPT_OSS_MOE_FP32=1   (replay mode: applies only to GPT_OSS_REPLAY_LAYER)
        #   GPT_OSS_CAPTURE_AFTER_LAYER=1   (saves to GPT_OSS_CAPTURE_PATH)
        #   GPT_OSS_REPLAY_LAYER=2          (loads from GPT_OSS_CAPTURE_PATH)
        #   GPT_OSS_CAPTURE_PATH=/tmp/gpt_oss_captured.pt
        #   GPT_OSS_REPLAY_MODE=full|attention_only|moe_only
        #   GPT_OSS_POST_ATTN_PATH=/tmp/gpt_oss_post_attn.pt
        if self.num_layers is None and os.environ.get("GPT_OSS_NUM_LAYERS"):
            self.num_layers = int(os.environ["GPT_OSS_NUM_LAYERS"])
        if not self.moe_fp32 and os.environ.get("GPT_OSS_MOE_FP32"):
            self.moe_fp32 = True
        if self.capture_after_layer is None and os.environ.get("GPT_OSS_CAPTURE_AFTER_LAYER"):
            self.capture_after_layer = int(os.environ["GPT_OSS_CAPTURE_AFTER_LAYER"])
        self._capture_path = os.environ.get("GPT_OSS_CAPTURE_PATH", "/tmp/gpt_oss_captured.pt")
        self._replay_layer = (
            int(os.environ["GPT_OSS_REPLAY_LAYER"])
            if os.environ.get("GPT_OSS_REPLAY_LAYER")
            else None
        )
        self._replay_mode = os.environ.get("GPT_OSS_REPLAY_MODE", "full")
        self._post_attn_capture_path = os.environ.get(
            "GPT_OSS_POST_ATTN_PATH", "/tmp/gpt_oss_post_attn.pt"
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="GPT-OSS",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the gpt-oss model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16 as default.

        Returns:
            torch.nn.Module: The gpt-oss model instance for causal language modeling.
        """
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config with modifications
        quantization_config = Mxfp4Config(dequantize=True)
        self.load_config()

        # Prepare model kwargs
        model_kwargs = {
            "config": self.config,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        # Set dtype - default to bfloat16 if not specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        from tt_torch.sparse_mlp import A2aSparseMLP
        if self.mlp_type == "a2a_sparse" :
            for layer in model.model.layers:
                mesh, _ = self.get_mesh_config(32)
                cluster_axis = 0
                layer.mlp = A2aSparseMLP(
                    layer.mlp,
                    num_experts=self.config.num_local_experts,
                    num_experts_per_tok=self.config.num_experts_per_tok,
                    num_devices=mesh[0] * mesh[1],
                    dispatch_devices=mesh[cluster_axis],
                    cluster_axis=cluster_axis,
                    config=self.config,
                )
        if self.moe_fp32:
            # In replay mode, cast only the replay target layer to fp32 to avoid
            # doubling MoE memory for every loaded layer on CPU.
            if self._replay_layer is not None:
                if not (0 <= self._replay_layer < len(model.model.layers)):
                    raise ValueError(
                        f"GPT_OSS_REPLAY_LAYER={self._replay_layer} is out of range for "
                        f"{len(model.model.layers)} loaded layers"
                    )
                target_layer_indices = [self._replay_layer]
                print(
                    f"[moe_fp32] Replay mode: applying fp32 to layer {self._replay_layer} only"
                )
            else:
                target_layer_indices = list(range(len(model.model.layers)))
                print(
                    f"[moe_fp32] Applying fp32 to all loaded layers "
                    f"(count={len(target_layer_indices)})"
                )

            for layer_idx in target_layer_indices:
                mlp = model.model.layers[layer_idx].mlp
                mlp.float()

                def _make_fp32_wrapper(orig_fwd):
                    def _fp32_forward(hidden_states):
                        out, scores = orig_fwd(hidden_states.float())
                        return out.to(torch.bfloat16), scores
                    return _fp32_forward

                mlp.forward = _make_fp32_wrapper(mlp.forward)

        if self.capture_after_layer is not None:
            target = model.model.layers[self.capture_after_layer]
            _loader_ref = self

            def _capture_hook(module, args, output):
                # Decoder layer output is a tensor [batch, seq, hidden] for GPT-OSS.
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                hidden = hidden.detach().clone()
                if hidden.device.type != "cpu":
                    return
                _loader_ref.captured_output = hidden
                torch.save(hidden, _loader_ref._capture_path)
                print(f"[capture] Saved layer {_loader_ref.capture_after_layer} output "
                      f"({hidden.shape}) → {_loader_ref._capture_path}")

            target.register_forward_hook(_capture_hook)

        self.model = model

        if self._replay_layer is not None:
            assert os.path.exists(self._capture_path), (
                f"Replay requested but capture file not found: {self._capture_path}\n"
                f"Run with GPT_OSS_CAPTURE_AFTER_LAYER={self._replay_layer - 1} first."
            )
            capture_data = torch.load(self._capture_path, weights_only=True)
            if isinstance(capture_data, dict):
                self.captured_output = capture_data["hidden_states"]
            else:
                self.captured_output = capture_data
            if not isinstance(self.captured_output, torch.Tensor):
                raise TypeError(
                    f"Captured data must be a torch.Tensor, got {type(self.captured_output)}"
                )
            if self.captured_output.ndim != 3:
                raise RuntimeError(
                    "Captured hidden_states must be 3D [batch, seq, hidden], "
                    f"got shape {tuple(self.captured_output.shape)}. "
                    "Please delete the capture file and recapture."
                )
            print(f"[replay] Loaded captured hidden_states ({self.captured_output.shape}) "
                  f"from {self._capture_path}")
            print(f"[replay] Creating single-layer model for layer {self._replay_layer} "
                  f"(mode={self._replay_mode})")

            # For moe_only mode, override frozen input with post-attention state
            if self._replay_mode == "moe_only":
                assert os.path.exists(self._post_attn_capture_path), (
                    f"MoE-only replay requires post-attention state.\n"
                    f"Run with GPT_OSS_REPLAY_MODE=attention_only first to capture it.\n"
                    f"Expected file: {self._post_attn_capture_path}"
                )
                post_attn = torch.load(self._post_attn_capture_path, weights_only=True)
                print(f"[replay] MoE-only: loaded post-attention state ({post_attn.shape}) "
                      f"from {self._post_attn_capture_path}")
                self.captured_output = post_attn

            replay = self.create_replay_model(layer_idx=self._replay_layer)
            self.model = replay
            return replay

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the gpt-oss model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           This is currently not used as tokenized inputs are integers.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Align replay inputs with captured hidden_states shape when replaying.
        target_batch = 64
        target_max_length = 128
        if (
            self._replay_layer is not None
            and isinstance(self.captured_output, torch.Tensor)
            and self.captured_output.ndim == 3
        ):
            target_batch = int(self.captured_output.shape[0])
            target_max_length = int(self.captured_output.shape[1])

        # Create tokenized inputs
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=target_max_length,
        )
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            # if the model uses sliding window attention, match sliding window value to input size so it
            # does not go out of bounds when updating the cache
            # Issue: https://github.com/tenstorrent/tt-xla/issues/3186
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        # inputs = {k: v.expand(target_batch, -1) for k, v in inputs.items()}

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism.

        Args:
            num_devices: Number of devices to use for tensor parallelism

        Returns:
            Tuple of (mesh_shape, axis_names)
        """
        # Support different mesh configurations based on number of devices
        if num_devices == 32:  # Galaxy
            mesh_shape = (8, 4)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        else:
            raise ValueError(f"Gpt-oss is only supported on llmbox and galaxy")

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Args:
            model: The gpt-oss model instance

        Returns:
            Dictionary mapping model parameters to their shard specifications,
            or None if sharding is not needed for this variant
        """
        if isinstance(model, _ReplayModel):
            return self.load_replay_shard_spec(model)

        shard_specs = {}

        # Embedding and output layers
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.model.norm.weight] = ("batch",)

        # lm_head sharding causes 20B hang: https://github.com/tenstorrent/tt-xla/issues/3484
        if self._variant and self._variant == ModelVariant.GPT_OSS_120B:
            shard_specs[model.lm_head.weight] = ("model", "batch")
        else:
            shard_specs[model.lm_head.weight] = (None, None)

        # Apply tensor parallel sharding to each transformer layer
        for layer in model.model.layers:
            # Attention layer sharding
            # q_proj weight shape: [2880, 4096]
            # Sharded column-wise (head-parallel): [2880, 4096/num_devices]
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)

            # k_proj weight shape: [2880, 512]
            # Sharded column-wise (head-parallel): [2880, 512/num_devices]
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)

            # v_proj weight shape: [2880, 512]
            # Sharded column-wise (head-parallel): [2880, 512/num_devices]
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)

            # o_proj weight shape: [4096, 2880]
            # Sharded row-wise: [4096/num_devices, 2880]
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = ("batch",)

            # sinks shape: [4096]
            # Local replication per device (row-wise)
            shard_specs[layer.self_attn.sinks] = (None,)

            # MLP layer sharding
            # Router weight is replicated with batch sharding
            shard_specs[layer.mlp.router.weight] = (None, "batch")

            if self.mlp_type == "a2a_sparse":
                shard_specs[layer.mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)
                shard_specs[layer.mlp.experts.down_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.down_proj_bias] = (("model", "batch"), None)
            else :
                # Shard experts across devices
                # For example: 32 experts / 8 devices = 4 experts per device
                # [num_experts, hidden_size, 2 * expert_dim]
                shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
                # [num_experts, 2 * expert_dim]
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
                # [num_experts, expert_dim, hidden_size]
                shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
                # [num_experts, hidden_size]
                shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")

            # Layer normalization weights
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def create_replay_model(self, layer_idx=None):
        """Create a single-layer model that replays the captured hidden_states.

        Must be called after running CPU inference with capture_after_layer set.

        Args:
            layer_idx: Index of the layer to test in isolation.
                       Defaults to capture_after_layer + 1.

        Returns:
            _ReplayModel: A model that runs only the target layer with frozen input.
        """
        assert self.captured_output is not None, (
            "No captured output. Run CPU inference first with capture_after_layer set."
        )
        if layer_idx is None:
            layer_idx = self.capture_after_layer + 1
        assert layer_idx < len(self.model.model.layers), (
            f"layer_idx {layer_idx} out of range (model has {len(self.model.model.layers)} layers)"
        )

        replay = _ReplayModel(
            decoder_layer=self.model.model.layers[layer_idx],
            norm=self.model.model.norm,
            lm_head=self.model.lm_head,
            rotary_emb=self.model.model.rotary_emb,
            frozen_hidden_states=self.captured_output,
            config=self.model.config,
            mode=self._replay_mode,
        )
        if self._replay_mode == "attention_only":
            replay._post_attn_capture_path = self._post_attn_capture_path
        if self._replay_mode == "attention_fp32":
            replay.float()  # Cast ALL params: self_attn, layernorm, norm, lm_head, frozen_input
            print(f"[attention_fp32] Cast entire model to float32 "
                  f"(frozen_input={replay.frozen_input.dtype}, "
                  f"lm_head={replay.lm_head.weight.dtype})")
        attn_intermediates_path = os.environ.get(
            "GPT_OSS_ATTN_INTERMEDIATES_PATH", "/tmp/gpt_oss_attn_intermediates.pt"
        )
        if self._replay_mode == "attn_softmax_only":
            replay._attn_intermediates_path = attn_intermediates_path
        if self._replay_mode == "attn_v_only":
            assert os.path.exists(attn_intermediates_path), (
                f"attn_v_only requires captured intermediates.\n"
                f"Run with GPT_OSS_REPLAY_MODE=attn_softmax_only first.\n"
                f"Expected: {attn_intermediates_path}"
            )
            data = torch.load(attn_intermediates_path, weights_only=True)
            replay.frozen_scores = torch.nn.Parameter(
                data["scores"], requires_grad=False,
            )
            replay.frozen_value_states = torch.nn.Parameter(
                data["value_states"], requires_grad=False,
            )
            print(f"[attn_v_only] Loaded scores={data['scores'].shape}, "
                  f"V={data['value_states'].shape}")
        replay.eval()
        return replay

    def load_replay_shard_spec(self, replay_model):
        """Load shard specifications for a _ReplayModel.

        Args:
            replay_model: A _ReplayModel instance from create_replay_model().

        Returns:
            Dictionary mapping model parameters to their shard specifications.
        """
        shard_specs = {}
        return shard_specs  # Start with empty and add only relevant params based on mode
        mode = replay_model.mode

        shard_specs[replay_model.frozen_input] = (None, None, "batch")
        shard_specs[replay_model.norm.weight] = ("batch",)

        if self._variant and self._variant == ModelVariant.GPT_OSS_120B:
            shard_specs[replay_model.lm_head.weight] = ("model", "batch")
        else:
            shard_specs[replay_model.lm_head.weight] = (None, None)

        layer = replay_model.layer

        # Attention weights (skip for moe_only — not used in forward)
        if mode != "moe_only":
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = ("batch",)
            shard_specs[layer.self_attn.sinks] = (None,)

        # MLP weights (skip for attention-only modes — not used in forward)
        # Frozen attn intermediates for attn_v_only (replicate across devices)
        if mode == "attn_v_only":
            shard_specs[replay_model.frozen_scores] = (None, None, None, None)
            shard_specs[replay_model.frozen_value_states] = (None, None, None, None)

        if mode not in ("attention_only", "attention_no_sink", "attention_fp32",
                        "attn_proj_only", "attn_qkt_only", "attn_softmax_only",
                        "attn_v_only", "attention_manual"):
            shard_specs[layer.mlp.router.weight] = (None, "batch")
            if self.mlp_type == "a2a_sparse":
                shard_specs[layer.mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)
                shard_specs[layer.mlp.experts.down_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.down_proj_bias] = (("model", "batch"), None)
            else:
                shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
                shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
                shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")

        shard_specs[layer.input_layernorm.weight] = ("batch",)
        shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the gpt-oss model with this instance's variant.

        Returns:
            The configuration object for the gpt-oss model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config