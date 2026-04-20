# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper Large V3 MSP-Podcast Emotion model loader for audio classification
(speech emotion recognition).

tiantiaf/whisper-large-v3-msp-podcast-emotion is a LoRA fine-tuned Whisper
Large V3 backbone with classification heads for categorical emotion,
detailed emotion, and arousal/valence/dominance regression. It is the
top-performing SAILER pipeline entry from the INTERSPEECH 2025 Speech
Emotion Challenge on the MSP-Podcast dataset.
"""

from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Whisper Large V3 MSP-Podcast Emotion model variants."""

    LARGE_V3_MSP_PODCAST_EMOTION = "Large_V3_MSP_Podcast_Emotion"


class ModelLoader(ForgeModel):
    """Whisper Large V3 MSP-Podcast Emotion model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_MSP_PODCAST_EMOTION: ModelConfig(
            pretrained_model_name="tiantiaf/whisper-large-v3-msp-podcast-emotion",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_MSP_PODCAST_EMOTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Large V3 MSP-Podcast Emotion",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        feature_extractor_kwargs = {}
        if dtype_override is not None:
            feature_extractor_kwargs["dtype"] = dtype_override

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            "openai/whisper-large-v3", chunk_length=15, **feature_extractor_kwargs
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from huggingface_hub import PyTorchModelHubMixin
        from transformers import WhisperConfig, WhisperModel

        class LoRALinear(nn.Module):
            """LoRA-adapted linear layer matching loralib parameter names."""

            def __init__(self, in_features, out_features, rank):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(out_features, in_features))
                self.bias = nn.Parameter(torch.empty(out_features))
                self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            def forward(self, x):
                return F.linear(x, self.weight, self.bias) + x @ self.lora_A.transpose(
                    0, 1
                ) @ self.lora_B.transpose(0, 1)

        class WhisperWrapper(nn.Module, PyTorchModelHubMixin):
            """Whisper backbone with LoRA fine-tuning and emotion heads."""

            def __init__(
                self,
                pretrain_model="whisper_large",
                hidden_dim=256,
                finetune_method="lora",
                lora_rank=16,
                freeze_params=True,
                output_class_num=9,
                use_conv_output=True,
                detailed_class_num=17,
            ):
                super().__init__()

                backbone_config = WhisperConfig.from_pretrained(
                    "openai/whisper-large-v3", max_source_positions=750
                )
                self.backbone_model = WhisperModel(backbone_config)

                encoder_layers = backbone_config.encoder_layers
                encoder_dim = backbone_config.d_model
                encoder_ffn_dim = backbone_config.encoder_ffn_dim

                # Apply LoRA to the feed-forward layers in the deeper half of the
                # encoder, matching the reference vox-profile WhisperEncoderLayer.
                if finetune_method == "lora":
                    half = encoder_layers // 2
                    for i in range(encoder_layers):
                        if i > half:
                            layer = self.backbone_model.encoder.layers[i]
                            layer.fc1 = LoRALinear(
                                encoder_dim, encoder_ffn_dim, lora_rank
                            )
                            layer.fc2 = LoRALinear(
                                encoder_ffn_dim, encoder_dim, lora_rank
                            )

                # External copy of positional embeddings kept for state-dict
                # compatibility with the published checkpoint.
                self.embed_positions = nn.Parameter(
                    torch.zeros(backbone_config.max_source_positions, encoder_dim),
                    requires_grad=False,
                )

                # Weighted-layer parameter retained for state-dict compatibility.
                if use_conv_output:
                    num_weight_layers = backbone_config.num_hidden_layers + 1
                    self.weights = nn.Parameter(
                        torch.ones(num_weight_layers) / num_weight_layers
                    )
                else:
                    num_weight_layers = backbone_config.num_hidden_layers
                    self.weights = nn.Parameter(torch.zeros(num_weight_layers))

                # Point-wise Conv1d projection stack.
                self.model_seq = nn.Sequential(
                    nn.Conv1d(encoder_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                )

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

            def forward(self, input_features):
                encoder_outputs = self.backbone_model.encoder(
                    input_features, output_hidden_states=True
                )
                features = encoder_outputs.hidden_states[-1]

                # Conv1d expects (batch, channels, time).
                features = features.transpose(1, 2)
                features = self.model_seq(features)
                features = features.transpose(1, 2)

                features = features.mean(dim=1)

                emotion = self.emotion_layer(features)
                detailed_emotion = self.detailed_out_layer(features)
                arousal = self.arousal_layer(features)
                valence = self.valence_layer(features)
                dominance = self.dominance_layer(features)

                return emotion, detailed_emotion, arousal, valence, dominance

        model = WhisperWrapper.from_pretrained(
            self._variant_config.pretrained_model_name,
            strict=False,
        )

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz. The feature
        # extractor pads to the configured 15-second chunk length.
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return {"input_features": inputs.input_features}
