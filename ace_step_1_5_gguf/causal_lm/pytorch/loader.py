# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 GGUF model loader implementation.

ACE-Step is a diffusion-based music generation model using a DiT architecture.
The GGUF checkpoint's "acestep-dit" architecture is not supported by
transformers' GGUF loader, so we load the model using trust_remote_code from
a HuggingFace repo that ships the custom model classes.

We test only the decoder (AceStepDiTModel) because the full model's forward
uses @torch.no_grad and internal randomness that prevent graph tracing.
"""
import torch
from transformers import AutoConfig, AutoModel
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

SOURCE_REPO = "ACE-Step/acestep-v15-turbo-shift3"


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 GGUF model variants."""

    ACE_STEP_1_5_TURBO_GGUF = "TURBO_GGUF"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 GGUF model loader.

    Loads the DiT decoder sub-model for compilation testing.
    """

    _VARIANTS = {
        ModelVariant.ACE_STEP_1_5_TURBO_GGUF: LLMModelConfig(
            pretrained_model_name=SOURCE_REPO,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ACE_STEP_1_5_TURBO_GGUF

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ACE-Step 1.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        torch_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        full_model = AutoModel.from_config(
            config, trust_remote_code=True, torch_dtype=torch_dtype
        )

        decoder = full_model.decoder.eval()
        self.config = config
        self.model = decoder
        return decoder

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.config is None:
            self.load_config()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        seq_len = self._variant_config.max_length
        if seq_len % self.config.patch_size != 0:
            seq_len = seq_len - (seq_len % self.config.patch_size)

        hidden_dim = self.config.audio_acoustic_hidden_dim
        context_dim = self.config.in_channels - hidden_dim
        encoder_seq_len = 16

        inputs = {
            "hidden_states": torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype),
            "timestep": torch.full((batch_size,), 0.5, dtype=dtype),
            "timestep_r": torch.full((batch_size,), 0.5, dtype=dtype),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "encoder_hidden_states": torch.randn(
                batch_size, encoder_seq_len, self.config.hidden_size, dtype=dtype
            ),
            "encoder_attention_mask": torch.ones(
                batch_size, encoder_seq_len, dtype=torch.long
            ),
            "context_latents": torch.randn(
                batch_size, seq_len, context_dim, dtype=dtype
            ),
        }
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
