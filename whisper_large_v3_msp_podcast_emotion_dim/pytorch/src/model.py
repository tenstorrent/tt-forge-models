# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal port of the vox-profile ``WhisperWrapper`` used by
``tiantiaf/whisper-large-v3-msp-podcast-emotion-dim``.

Reference: https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/model/emotion/whisper_emotion_dim.py
"""
import copy

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import WhisperModel


class WhisperWrapper(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/tiantiaf0627/vox-profile-release",
):
    """Whisper-large-v3 encoder with dimensional emotion regression heads.

    The checkpoint is trained with ``finetune_method="finetune"`` so the
    LoRA-aware ``WhisperEncoderLayer`` from the upstream implementation is not
    exercised and is intentionally omitted here. The forward pass mirrors the
    inference path of ``WhisperWrapper.forward`` and returns the three
    dimensional emotion scalars (arousal, valence, dominance).
    """

    def __init__(
        self,
        pretrain_model="whisper_large",
        hidden_dim=256,
        finetune_method="finetune",
        lora_rank=16,
        freeze_params=True,
        output_class_num=9,
        use_conv_output=True,
        detailed_class_num=17,
        predict_gender=False,
    ):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.use_conv_output = use_conv_output
        self.predict_gender = predict_gender

        self.backbone_model = WhisperModel.from_pretrained(
            "openai/whisper-large-v3",
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
            max_source_positions=750,
        )
        self.embed_positions = copy.deepcopy(
            self.backbone_model.encoder.embed_positions.weight
        )
        self.embed_positions.requires_grad = False
        self.model_config = self.backbone_model.config

        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )

        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))

        self.emotion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        self.detailed_out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, detailed_class_num),
        )
        self.arousal_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.valence_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.dominance_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if self.predict_gender:
            self.gender_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, input_features):
        # Mirror the upstream forward: reinstall the 15-second positional
        # embeddings before running the encoder.
        self.backbone_model.encoder.embed_positions = nn.Embedding.from_pretrained(
            self.embed_positions[:750]
        )

        hidden_states = self.backbone_model.encoder(
            input_features, output_hidden_states=True
        ).hidden_states
        features = torch.stack(hidden_states, dim=0)[-1]

        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        features = torch.mean(features, dim=1)

        arousal = self.arousal_layer(features)
        valence = self.valence_layer(features)
        dominance = self.dominance_layer(features)
        return arousal, valence, dominance
