# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IndexTTS-1.5 model loader implementation for text-to-speech tasks.
"""
import os

import torch
import torch.nn as nn
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


class IndexTTS15GPTWrapper(nn.Module):
    """Wrapper around the IndexTTS-1.5 GPT (UnifiedVoice) backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces mel token logits.
    """

    def __init__(self, inference_model):
        super().__init__()
        # Store inference_model (a GPT2InferenceModel, i.e. nn.Module) directly
        # so that .to(device) properly moves all transformer parameters.
        # IndexTTS is a plain Python class (not nn.Module), so wrapping it
        # directly would leave parameters on CPU when moved to XLA.
        self.inference_model = inference_model

    def forward(self, inputs_embeds):
        outputs = self.inference_model.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.inference_model.lm_head(hidden_states)
        return logits


class ModelVariant(StrEnum):
    """Available IndexTTS-1.5 model variants."""

    INDEXTTS_1_5 = "IndexTTS-1.5"


class ModelLoader(ForgeModel):
    """IndexTTS-1.5 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.INDEXTTS_1_5: ModelConfig(
            pretrained_model_name="IndexTeam/IndexTTS-1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDEXTTS_1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IndexTTS-1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import sys
        import types

        from huggingface_hub import snapshot_download

        # transformers ≥5.0 removed model_parallel_utils; provide a stub so
        # indextts can import without error (only used in parallelize() calls).
        if "transformers.utils.model_parallel_utils" not in sys.modules:
            _stub = types.ModuleType("transformers.utils.model_parallel_utils")
            _stub.assert_device_map = lambda *a, **kw: None
            _stub.get_device_map = lambda *a, **kw: {}
            sys.modules["transformers.utils.model_parallel_utils"] = _stub

        from indextts.infer import IndexTTS

        model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name,
        )
        tts = IndexTTS(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            is_fp16=False,
            use_cuda_kernel=False,
        )
        # IndexTTS is a plain Python class (not nn.Module), so its internal
        # modules are invisible to .to(device). Extract inference_model (an
        # nn.Module) so device placement reaches all parameters.
        inference_model = tts.gpt.inference_model
        model = IndexTTS15GPTWrapper(inference_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # GPT backbone hidden_size=1280, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1280, dtype=dtype)
        return (inputs_embeds,)
