# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NextStep-1.1 model loader implementation for text-to-image generation.
"""

import os
import sys
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ...base import ForgeModel


def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
    """Standard RoPE init for transformers 5.x compatibility (replaces the removed 'default' ROPE_INIT_FUNCTIONS key)."""
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device)
            / head_dim
        )
    )
    return inv_freq, 1.0


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
    """Available NextStep-1.1 model variants."""

    NEXTSTEP_1_1 = "NextStep-1.1"


class ModelLoader(ForgeModel):
    """NextStep-1.1 model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.NEXTSTEP_1_1: ModelConfig(
            pretrained_model_name="stepfun-ai/NextStep-1.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEXTSTEP_1_1

    sample_text = (
        'A realistic photograph of a wall with "TOWARD AUTOREGRESSIVE IMAGE '
        'GENERATION" displayed.'
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NextStep-1.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def _setup_remote_code_path(self, pretrained_model_name):
        snapshot_path = snapshot_download(
            pretrained_model_name,
            ignore_patterns=["*.safetensors", "*.pt", "assets/*"],
        )
        for subdir in ["models", "utils", "vae"]:
            subdir_path = os.path.join(snapshot_path, subdir)
            if os.path.isdir(subdir_path):
                init_file = os.path.join(subdir_path, "__init__.py")
                if not os.path.exists(init_file):
                    open(init_file, "w").close()
        if snapshot_path not in sys.path:
            sys.path.insert(0, snapshot_path)

        # transformers 5.x removed "default" from ROPE_INIT_FUNCTIONS; patch it back in
        if "default" not in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._setup_remote_code_path(pretrained_model_name)

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # transformers 5.x moved rope_theta into rope_parameters; backfill for model code expecting config.rope_theta
        if not hasattr(config, "rope_theta") and getattr(
            config, "rope_parameters", None
        ):
            config.rope_theta = config.rope_parameters.get("rope_theta", 10000.0)

        model = AutoModel.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()

        # NextStep has no forward() method; add one that embeds input_ids and delegates to forward_model
        import types

        def _forward(self_model, input_ids=None, attention_mask=None, **kwargs):
            inputs_embeds = self_model.prepare_inputs_embeds(input_ids)
            return self_model.forward_model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )

        model.forward = types.MethodType(_forward, model)

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
        )

        return inputs
