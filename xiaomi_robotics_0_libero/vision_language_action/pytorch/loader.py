# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Xiaomi-Robotics-0-LIBERO (MiBoT) model loader implementation for
vision-language-action prediction on the LIBERO benchmark.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import get_file


class MiBoTInferenceWrapper(torch.nn.Module):
    """Wraps MiBoTForActionGeneration so the forward seed is injected from a fixed
    integer (MiBoT's forward expects ``kwargs['seed']`` for deterministic diffusion
    sampling)."""

    def __init__(self, model: torch.nn.Module, seed: int = 0):
        super().__init__()
        self.model = model
        self.seed = seed

    def forward(self, **kwargs):
        return self.model(seed=self.seed, **kwargs)


class ModelVariant(StrEnum):
    """Available Xiaomi-Robotics-0-LIBERO (MiBoT) model variants."""

    LIBERO = "libero"


class ModelLoader(ForgeModel):
    """Loader for Xiaomi-Robotics-0-LIBERO (MiBoT) action prediction model."""

    _VARIANTS = {
        ModelVariant.LIBERO: ModelConfig(
            pretrained_model_name="XiaomiRobotics/Xiaomi-Robotics-0-LIBERO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIBERO

    sample_prompt = "pick up the red block"
    robot_type = "libero_all"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Xiaomi-Robotics-0-LIBERO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    @staticmethod
    def _patch_rope_init_functions():
        # transformers>=5.0 removed "default" from ROPE_INIT_FUNCTIONS; the
        # model's trust_remote_code still does a dict lookup for that key.
        from transformers import modeling_rope_utils

        if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
            import torch

            def _default_rope(config=None, device=None, seq_len=None, **rope_kwargs):
                if rope_kwargs:
                    base, dim = rope_kwargs["base"], rope_kwargs["dim"]
                else:
                    base = config.rope_theta
                    factor = getattr(config, "partial_rotary_factor", 1.0)
                    head_dim = getattr(
                        config,
                        "head_dim",
                        config.hidden_size // config.num_attention_heads,
                    )
                    dim = int(head_dim * factor)
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).float().to(device)
                        / dim
                    )
                )
                return inv_freq, 1.0

            modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope

    def load_model(self, *, dtype_override=None, **kwargs):
        self._patch_rope_init_functions()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model_config = model.config

        if self.processor is None:
            self._load_processor()

        return MiBoTInferenceWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vlm_inputs = self.processor(
            text=[text] * batch_size,
            images=[image] * batch_size,
            return_tensors="pt",
        )

        cfg = self.model_config
        state_dim = cfg.state_dim if cfg is not None else 32
        state_length = cfg.state_length if cfg is not None else 1
        state = torch.zeros(batch_size, state_length, state_dim, dtype=torch.float32)
        action_mask = self.processor.get_action_mask(
            self.robot_type, batch_size=batch_size
        )

        if dtype_override is not None:
            if "pixel_values" in vlm_inputs:
                vlm_inputs["pixel_values"] = vlm_inputs["pixel_values"].to(
                    dtype_override
                )
            state = state.to(dtype_override)
            action_mask = action_mask.to(dtype_override)

        inputs = {
            "state": state,
            "action_mask": action_mask,
            **{k: v for k, v in vlm_inputs.items()},
        }
        return inputs

    def unpack_forward_output(self, fwd_output):
        """``MiBoTForActionGeneration`` returns ``ActionGenerationOutput(actions=...)``."""
        if hasattr(fwd_output, "actions"):
            return fwd_output.actions
        return fwd_output
