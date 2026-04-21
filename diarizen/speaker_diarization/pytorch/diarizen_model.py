# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone DiariZen WavLM-Conformer model implementation.

Implements the pruned WavLM + Conformer architecture used by
BUT-FIT/diarizen-wavlm-large-s80-md without requiring the upstream
diarizen library (which is not available on PyPI).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# WavLM feature extractor
# ---------------------------------------------------------------------------


class WavLMConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, use_group_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=False
        )
        if use_group_norm:
            self.layer_norm = nn.GroupNorm(out_ch, out_ch, affine=True)
        else:
            self.layer_norm = nn.LayerNorm(out_ch, elementwise_affine=True)
        self.use_group_norm = use_group_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_group_norm:
            x = self.layer_norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.layer_norm(x)
            x = x.transpose(-1, -2)
        x = F.gelu(x)
        return x


class WavLMFeatureExtractor(nn.Module):
    def __init__(self, conv_specs):
        super().__init__()
        # conv_specs: list of (in_ch, out_ch, kernel, stride)
        layers = []
        for i, (in_ch, out_ch, kernel, stride) in enumerate(conv_specs):
            layers.append(
                WavLMConvLayer(in_ch, out_ch, kernel, stride, use_group_norm=(i == 0))
            )
        self.conv_layers = nn.ModuleList(layers)
        last_ch = conv_specs[-1][1]
        self.dummy_weight = nn.Parameter(torch.zeros(last_ch))

    def forward(self, x):
        # x: [batch, samples]
        x = x.unsqueeze(1)  # [batch, 1, samples]
        for layer in self.conv_layers:
            x = layer(x)
        return x.transpose(-1, -2)  # [batch, time, channels]


# ---------------------------------------------------------------------------
# WavLM encoder components
# ---------------------------------------------------------------------------


class WavLMFeatureProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.projection = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_dim, kernel_size=128, groups=16):
        super().__init__()
        conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        # x: [batch, time, hidden_dim]
        x = x.transpose(-1, -2)  # [batch, hidden, time]
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[:, :, : -self.num_remove]
        x = F.gelu(x)
        x = x.transpose(-1, -2)  # [batch, time, hidden]
        return x


class WavLMAttention(nn.Module):
    """WavLM attention with variable per-layer q/k/v dim and optional rel-pos."""

    def __init__(self, hidden_dim, num_heads, qkv_dim, has_rel_attn=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, qkv_dim)
        self.k_proj = nn.Linear(hidden_dim, qkv_dim)
        self.v_proj = nn.Linear(hidden_dim, qkv_dim)
        self.out_proj = nn.Linear(qkv_dim, hidden_dim)

        # GRU relative position gating (always present when attention exists)
        self.gru_rel_pos_const = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(64, 8)

        if has_rel_attn:
            # rel_attn_embed shape matches [qkv_dim, num_heads] in weights
            self.rel_attn_embed = nn.Embedding(qkv_dim, num_heads)
        else:
            self.rel_attn_embed = None

    def forward(self, x):
        # x: [batch, seq, hidden]
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # [bsz, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [bsz, heads, seq, head_dim]
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.qkv_dim)
        out = self.out_proj(out)
        return out


class WavLMFeedForward(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_dim, intermediate_dim)
        self.output_dense = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = F.gelu(x)
        x = self.output_dense(x)
        return x


class WavLMEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, qkv_dim, ffn_dim, has_rel_attn=False):
        super().__init__()
        self.has_attention = qkv_dim > 0
        if self.has_attention:
            self.attention = WavLMAttention(
                hidden_dim, num_heads, qkv_dim, has_rel_attn
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = WavLMFeedForward(hidden_dim, ffn_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.has_attention:
            residual = x
            x = self.layer_norm(x)
            x = self.attention(x)
            x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = x + residual
        return x


class WavLMEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, layer_configs):
        super().__init__()
        self.feature_projection = WavLMFeatureProjection(feat_dim, hidden_dim)
        self.transformer = _WavLMTransformer(hidden_dim, layer_configs)

    def forward(self, x):
        x = self.feature_projection(x)
        hidden_states = self.transformer(x)
        return hidden_states


class _WavLMTransformer(nn.Module):
    def __init__(self, hidden_dim, layer_configs):
        super().__init__()
        self.pos_conv_embed = WavLMPositionalConvEmbedding(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                WavLMEncoderLayer(
                    hidden_dim,
                    cfg["num_heads"],
                    cfg["qkv_dim"],
                    cfg["ffn_dim"],
                    cfg.get("has_rel_attn", False),
                )
                for cfg in layer_configs
            ]
        )

    def forward(self, x):
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)
        all_hidden = [x]
        for layer in self.layers:
            x = layer(x)
            all_hidden.append(x)
        return all_hidden  # list of 25 tensors [batch, seq, hidden]


class WavLMModel(nn.Module):
    def __init__(self, conv_specs, feat_dim, hidden_dim, layer_configs):
        super().__init__()
        self.feature_extractor = WavLMFeatureExtractor(conv_specs)
        self.encoder = WavLMEncoder(feat_dim, hidden_dim, layer_configs)

    def forward(self, x):
        # x: [batch, samples]
        features = self.feature_extractor(x)  # [batch, time, feat_dim]
        hidden_states = self.encoder(features)  # list of 25 [batch, time, hidden]
        return hidden_states


# ---------------------------------------------------------------------------
# Conformer
# ---------------------------------------------------------------------------


class ConformerFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.ln_norm = nn.LayerNorm(dim)
        self.w_1 = nn.Linear(dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        x = self.ln_norm(x)
        x = F.silu(self.w_1(x))
        x = self.w_2(x)
        return x


class ConformerMHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_norm = nn.LayerNorm(dim)
        self.mha = _ConformerMHACore(dim, num_heads)

    def forward(self, x):
        x = self.ln_norm(x)
        x = self.mha(x)
        return x


class _ConformerMHACore(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.linearQ = nn.Linear(dim, dim)
        self.linearK = nn.Linear(dim, dim)
        self.linearV = nn.Linear(dim, dim)
        self.linearO = nn.Linear(dim, dim)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        q = self.linearQ(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linearK(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linearV(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.linearO(out)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.ln_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.bn_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: [batch, seq, dim]
        x = self.ln_norm(x)
        x = x.transpose(-1, -2)  # [batch, dim, seq]
        x = self.pointwise_conv1(x)  # [batch, dim*2, seq]
        x = F.glu(x, dim=1)  # [batch, dim, seq]
        x = self.depthwise_conv(x)
        x = self.bn_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(-1, -2)  # [batch, seq, dim]
        return x


class ConformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, conv_kernel_size=31):
        super().__init__()
        self.ffn1 = ConformerFeedForward(dim, ffn_dim)
        self.mha = ConformerMHA(dim, num_heads)
        self.conv = ConformerConvModule(dim, conv_kernel_size)
        self.ffn2 = ConformerFeedForward(dim, ffn_dim)
        self.ln_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mha(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.ln_norm(x)
        return x


class Conformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, ffn_dim, conv_kernel_size=31):
        super().__init__()
        self.conformer_layer = nn.ModuleList(
            [
                ConformerLayer(dim, num_heads, ffn_dim, conv_kernel_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.conformer_layer:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Full DiariZen model
# ---------------------------------------------------------------------------


class DiariZenSegmentationModel(nn.Module):
    """
    Segmentation model from the DiariZen diarization pipeline.

    Architecture: pruned WavLM → weighted hidden-state sum → projection
    → LayerNorm → Conformer → linear classifier.
    """

    CONV_SPECS = [
        (1, 512, 10, 5),
        (512, 153, 3, 2),
        (153, 224, 3, 2),
        (224, 255, 3, 2),
        (255, 302, 3, 2),
        (302, 368, 2, 2),
        (368, 211, 2, 2),
    ]
    FEAT_DIM = 211
    HIDDEN_DIM = 1024
    NUM_WAVLM_LAYERS = 24
    CONFORMER_DIM = 256
    CONFORMER_HEADS = 4
    CONFORMER_FFN_DIM = 1024
    CONFORMER_LAYERS = 4
    CONFORMER_KERNEL = 31
    NUM_CLASSES = 11

    # Per-layer (qkv_dim, ffn_dim, has_rel_attn); qkv_dim=0 means no attention
    LAYER_CONFIGS = [
        {"num_heads": 16, "qkv_dim": 320, "ffn_dim": 1092, "has_rel_attn": True},
        {"num_heads": 16, "qkv_dim": 192, "ffn_dim": 925},
        {"num_heads": 16, "qkv_dim": 384, "ffn_dim": 759},
        {"num_heads": 16, "qkv_dim": 384, "ffn_dim": 646},
        {"num_heads": 16, "qkv_dim": 320, "ffn_dim": 745},
        {"num_heads": 16, "qkv_dim": 320, "ffn_dim": 615},
        {"num_heads": 16, "qkv_dim": 192, "ffn_dim": 684},
        {"num_heads": 16, "qkv_dim": 320, "ffn_dim": 958},
        {"num_heads": 16, "qkv_dim": 256, "ffn_dim": 286},
        {"num_heads": 16, "qkv_dim": 0, "ffn_dim": 294},  # no attention
        {"num_heads": 16, "qkv_dim": 128, "ffn_dim": 406},
        {"num_heads": 16, "qkv_dim": 256, "ffn_dim": 377},
        {"num_heads": 16, "qkv_dim": 0, "ffn_dim": 463},  # no attention
        {"num_heads": 16, "qkv_dim": 192, "ffn_dim": 542},
        {"num_heads": 16, "qkv_dim": 128, "ffn_dim": 298},
        {"num_heads": 16, "qkv_dim": 64, "ffn_dim": 236},
        {"num_heads": 16, "qkv_dim": 0, "ffn_dim": 96},  # no attention
        {"num_heads": 16, "qkv_dim": 0, "ffn_dim": 104},  # no attention
        {"num_heads": 16, "qkv_dim": 64, "ffn_dim": 134},
        {"num_heads": 16, "qkv_dim": 128, "ffn_dim": 211},
        {"num_heads": 16, "qkv_dim": 384, "ffn_dim": 473},
        {"num_heads": 16, "qkv_dim": 576, "ffn_dim": 1011},
        {"num_heads": 16, "qkv_dim": 640, "ffn_dim": 1770},
        {"num_heads": 16, "qkv_dim": 512, "ffn_dim": 1316},
    ]

    def __init__(self):
        super().__init__()
        self.wavlm_model = WavLMModel(
            self.CONV_SPECS, self.FEAT_DIM, self.HIDDEN_DIM, self.LAYER_CONFIGS
        )
        # Weighted sum over 25 hidden states (feature proj output + 24 layers)
        self.weight_sum = nn.Linear(25, 1, bias=False)
        self.proj = nn.Linear(self.HIDDEN_DIM, self.CONFORMER_DIM)
        self.lnorm = nn.LayerNorm(self.CONFORMER_DIM)
        self.conformer = Conformer(
            self.CONFORMER_LAYERS,
            self.CONFORMER_DIM,
            self.CONFORMER_HEADS,
            self.CONFORMER_FFN_DIM,
            self.CONFORMER_KERNEL,
        )
        self.classifier = nn.Linear(self.CONFORMER_DIM, self.NUM_CLASSES)

    def forward(self, x):
        # x: [batch, 1, samples] (the pipeline feeds mono audio with channel dim)
        x = x.squeeze(1)  # [batch, samples]
        hidden_states = self.wavlm_model(x)  # list of 25 [batch, time, 1024]

        # Stack and apply learned weighted sum
        stacked = torch.stack(hidden_states, dim=-1)  # [batch, time, 1024, 25]
        weights = F.softmax(self.weight_sum.weight, dim=-1)  # [1, 25]
        x = (stacked * weights).sum(dim=-1)  # [batch, time, 1024]

        x = self.proj(x)
        x = self.lnorm(x)
        x = self.conformer(x)
        x = self.classifier(x)
        return x

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DiariZenSegmentationModel":
        from huggingface_hub import hf_hub_download

        model = cls()
        weights_path = hf_hub_download(model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Expected for rel_attn_embed layers (only layer 0 has it in weights)
            non_rel_missing = [k for k in missing if "rel_attn_embed" not in k and "gru_rel" not in k]
            if non_rel_missing:
                raise RuntimeError(f"Missing keys: {non_rel_missing}")
        return model
