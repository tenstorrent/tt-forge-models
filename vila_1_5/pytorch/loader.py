# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VILA1.5 model loader implementation for multimodal conditional generation.

VILA1.5 is a vision-language model combining a SigLIP vision encoder
with a ShearedLLaMA language model backbone, connected via an MLP
downsample projector.
"""

import json
import os
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class LlavaLlamaConfig(PreTrainedConfig):
    model_type = "llava_llama"

    def __init__(
        self,
        hidden_size: int = 2560,
        mm_hidden_size: int = 1152,
        mm_vision_select_layer: int = -2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.mm_vision_select_layer = mm_vision_select_layer


class _VILAProjector(nn.Module):
    """mlp_downsample projector: Identity → LayerNorm → Linear → GELU → Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LlavaLlamaModel(PreTrainedModel):
    """VILA1.5 multimodal model: SigLIP vision + mlp_downsample projector + LLaMA LM."""

    config_class = LlavaLlamaConfig

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # VILA has no MoE experts; skip sys.modules lookup in transformers >= 5.2
        return False

    def __init__(self, config: LlavaLlamaConfig):
        super().__init__(config)
        self.vision_tower: Optional[SiglipVisionModel] = None
        self.mm_projector: Optional[_VILAProjector] = None
        self.language_model: Optional[LlamaForCausalLM] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "LlavaLlamaModel":
        kwargs.pop("trust_remote_code", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        local_dir = snapshot_download(pretrained_model_name_or_path)

        with open(os.path.join(local_dir, "config.json")) as f:
            raw = json.load(f)

        config = LlavaLlamaConfig(
            hidden_size=raw.get("hidden_size", 2560),
            mm_hidden_size=raw.get("mm_hidden_size", 1152),
            mm_vision_select_layer=raw.get("mm_vision_select_layer", -2),
        )

        model = cls(config)
        dtype_kw = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

        model.language_model = LlamaForCausalLM.from_pretrained(
            os.path.join(local_dir, "llm"), **dtype_kw
        )
        model.vision_tower = SiglipVisionModel.from_pretrained(
            os.path.join(local_dir, "vision_tower"), **dtype_kw
        )

        # mlp_downsample projects from mm_hidden_size * 4 (after pixel_unshuffle) to hidden_size
        in_dim = config.mm_hidden_size * 4
        model.mm_projector = _VILAProjector(in_dim, config.hidden_size)
        proj_weights = safetensors.torch.load_file(
            os.path.join(local_dir, "mm_projector", "model.safetensors")
        )
        model.mm_projector.load_state_dict(proj_weights, strict=False)

        if torch_dtype is not None:
            model = model.to(torch_dtype)

        return model

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_tower(pixel_values, output_hidden_states=True)
        # Select from configured layer (default: second-to-last)
        features = vision_out.hidden_states[self.config.mm_vision_select_layer]
        B, N, D = features.shape

        # Reshape to spatial grid and apply pixel_unshuffle for mlp_downsample
        H = W = int(N**0.5)
        spatial = features.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        # Pad to even spatial dims so pixel_unshuffle(2) divides exactly
        spatial = F.pad(spatial, (0, W % 2, 0, H % 2))
        spatial = F.pixel_unshuffle(spatial, 2)  # (B, D*4, H//2, W//2)
        _, D4, Hd, Wd = spatial.shape
        patches = spatial.permute(0, 2, 3, 1).reshape(B, Hd * Wd, D4)
        return self.mm_projector(patches)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self._encode_images(pixel_values)
            B, n_vis, _ = image_features.shape
            # Prepend visual tokens to the text sequence
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)
            if attention_mask is not None:
                vis_mask = torch.ones(
                    B,
                    n_vis,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )


class ModelVariant(StrEnum):
    """Available VILA1.5 model variants."""

    VILA_1_5_3B = "3B"


class ModelLoader(ForgeModel):
    """VILA1.5 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.VILA_1_5_3B: ModelConfig(
            pretrained_model_name="Efficient-Large-Model/VILA1.5-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VILA_1_5_3B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VILA1.5 model loader."""
        super().__init__(variant)
        self._local_dir: Optional[str] = None
        self._image_processor: Optional[SiglipImageProcessor] = None
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VILA1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_local(self):
        if self._local_dir is None:
            self._local_dir = snapshot_download(
                self._variant_config.pretrained_model_name
            )

    def _load_processor(self):
        self._ensure_local()
        self._image_processor = SiglipImageProcessor.from_pretrained(
            os.path.join(self._local_dir, "vision_tower")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self._local_dir, "llm")
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VILA1.5 model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlavaLlamaModel.from_pretrained(str(model_name), **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VILA1.5."""
        if self._image_processor is None or self._tokenizer is None:
            self._load_processor()

        text_inputs = self._tokenizer(self.sample_text, return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        pixel_values = self._image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if dtype_override:
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        if batch_size > 1:
            result = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in result.items()
            }

        return result
