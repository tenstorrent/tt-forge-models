# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper classes for the Vox-Profile narrow-accent Whisper classifier.

Vendored from https://github.com/tiantiaf0627/vox-profile-release
(src/model/accent/whisper_accent.py) with CUDA-specific calls replaced by
device-agnostic equivalents so that the model can be loaded and executed
in CPU-only bring-up environments. The gradient-reversal branch is
unused by the ``tiantiaf/whisper-large-v3-narrow-accent`` config and has
been pruned accordingly.
"""

import copy

import loralib as lora
import torch
import transformers.models.whisper.modeling_whisper as whisper
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoFeatureExtractor, WhisperModel
from transformers.activations import ACT2FN


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = whisper.WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config

        if layer_idx > config.encoder_layers // 2:
            if self.config.finetune_method in ("lora", "combined"):
                self.fc1 = lora.Linear(
                    self.embed_dim, config.encoder_ffn_dim, r=config.lora_rank
                )
                self.fc2 = lora.Linear(
                    config.encoder_ffn_dim, self.embed_dim, r=config.lora_rank
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WhisperWrapper(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/tiantiaf0627/vox-profile-release",
):
    def __init__(
        self,
        pretrain_model="whisper_large",
        output_class_num=4,
        hidden_dim=256,
        finetune_method="lora",
        lora_rank=16,
        freeze_params=True,
        use_conv_output=True,
        apply_gradient_reversal=False,
        num_dataset=4,
    ):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "openai/whisper-tiny", chunk_length=15
        )
        self.pretrain_model = pretrain_model
        backbone_sources = {
            "whisper_tiny": "openai/whisper-tiny",
            "whisper_base": "openai/whisper-base",
            "whisper_small": "openai/whisper-small",
            "whisper_medium": "openai/whisper-medium",
            "whisper_large": "openai/whisper-large-v3",
        }
        backbone_id = backbone_sources[self.pretrain_model]
        if self.pretrain_model == "whisper_large":
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                backbone_id, chunk_length=15
            )
        self.backbone_model = WhisperModel.from_pretrained(
            backbone_id,
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
            max_source_positions=750,
        )
        self.embed_positions = copy.deepcopy(
            self.backbone_model.encoder.embed_positions.weight
        )
        self.embed_positions.requires_grad = False

        state_dict = self.backbone_model.state_dict()
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = finetune_method
        self.model_config.lora_rank = lora_rank
        self.finetune_method = finetune_method
        self.apply_gradient_reversal = apply_gradient_reversal
        self.use_conv_output = use_conv_output

        if self.finetune_method == "lora":
            self.backbone_model.encoder.layers = nn.ModuleList(
                [
                    WhisperEncoderLayer(self.model_config, layer_idx)
                    for layer_idx in range(self.model_config.encoder_layers)
                ]
            )
            msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        self.freeze_params = freeze_params
        if self.freeze_params and self.finetune_method != "lora":
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = False
        elif self.freeze_params and self.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                p.requires_grad = name in msg.missing_keys
        else:
            for name, p in self.backbone_model.named_parameters():
                trainable = not any(
                    tag in name
                    for tag in ("decoder", "conv1", "conv2", "embed_positions")
                )
                p.requires_grad = trainable

        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )

        if use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, x, length=None, return_feature=False):
        max_audio_len = 15 * 16000
        device = x.device

        if length is not None:
            batched = [x[idx].detach().cpu().numpy() for idx in range(len(length))]
            features = self.feature_extractor(
                batched,
                return_tensors="pt",
                sampling_rate=16000,
                max_length=max_audio_len,
            ).input_features.to(device)
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
        else:
            features = self.feature_extractor(
                x[0].detach().cpu(),
                return_tensors="pt",
                sampling_rate=16000,
                max_length=max_audio_len,
            ).input_features.to(device)
            length = self._get_feat_extract_output_lengths(torch.tensor([len(x[0])]))

        self.backbone_model.encoder.embed_positions = (
            self.backbone_model.encoder.embed_positions.from_pretrained(
                self.embed_positions[:750]
            )
        )

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states
        features = torch.stack(hidden_states, dim=0)[-1]

        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        if length is not None and features.shape[0] == length.shape[0]:
            pooled = []
            for snt_id in range(features.shape[0]):
                actual_size = int(length[snt_id])
                pooled.append(torch.mean(features[snt_id, :actual_size, ...], dim=0))
            features = torch.stack(pooled)
        else:
            features = torch.mean(features, dim=1)

        predicted = self.out_layer(features)
        if return_feature:
            return predicted, features
        return predicted

    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths
