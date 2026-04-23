# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
import numpy as np
import torch
from PIL import Image
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
    """Available Surya Recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Surya Recognition model loader for OCR text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="vikp/surya_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._foundation_predictor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="surya_rec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_surya_for_transformers5():
        # surya-ocr 0.17.1 has four incompatibilities with transformers 5.2+:
        # 1. ROPE_INIT_FUNCTIONS no longer contains 'default'; add it back.
        # 2. SuryaModel._tied_weights_keys is a list but transformers 5.2+ expects
        #    a dict mapping target weight name → source weight name.
        # 3. SuryaModel.tie_weights() doesn't accept recompute_mapping kwarg.
        #    Also uses _tie_or_clone_weights which was removed.
        # 4. SuryaModel.__init__ doesn't call self.post_init(), which is required
        #    by transformers 5.2+ to initialize all_tied_weights_keys.
        # 5. Qwen2_5_VisionRotaryEmbedding stores inv_freq as a plain attribute;
        #    transformers 5.2+ uses meta device init so plain attrs stay on meta
        #    and .to(device) skips them. Fix: register_buffer so it gets moved.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from surya.common.surya import SuryaModel
        from surya.common.surya.encoder import Qwen2_5_VisionRotaryEmbedding

        if "default" not in ROPE_INIT_FUNCTIONS:

            def _compute_default(config, device=None, seq_len=None):
                base = getattr(config, "rope_theta", 10000.0)
                dim = config.hidden_size // config.num_attention_heads
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).float().to(device)
                        / dim
                    )
                )
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS["default"] = _compute_default

        if isinstance(SuryaModel._tied_weights_keys, list):
            SuryaModel._tied_weights_keys = {
                "lm_head.weight": "embedder.token_embed.weight"
            }

        def _patched_tie_weights(self, **kwargs):
            # transformers 5.2+ removed _tie_or_clone_weights; tie directly
            self.lm_head.weight = self.embedder.token_embed.weight

        SuryaModel.tie_weights = _patched_tie_weights

        _orig_rot_emb_init = Qwen2_5_VisionRotaryEmbedding.__init__

        def _patched_rot_emb_init(self, dim: int, theta: float = 10000.0) -> None:
            _orig_rot_emb_init(self, dim, theta)
            if isinstance(self.inv_freq, torch.Tensor):
                inv_freq = self.inv_freq
                del self.inv_freq
                self.register_buffer("inv_freq", inv_freq, persistent=False)

        Qwen2_5_VisionRotaryEmbedding.__init__ = _patched_rot_emb_init

        _orig_surya_init = SuryaModel.__init__

        def _patched_surya_init(self, config, **kwargs):
            _orig_surya_init(self, config, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        SuryaModel.__init__ = _patched_surya_init

    def load_model(self, *, dtype_override=None, **kwargs):
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self._patch_surya_for_transformers5()
        self._foundation_predictor = FoundationPredictor(device="cpu")
        RecognitionPredictor(self._foundation_predictor)
        model = self._foundation_predictor.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from surya.common.surya.schema import TaskNames

        image = np.array(Image.new("RGB", (896, 196), color=(255, 255, 255)))
        fp = self._foundation_predictor
        model = fp.model

        batch_input = fp.prepare_input(
            task_names=[TaskNames.ocr_without_boxes] * batch_size,
            images=[image] * batch_size,
            input_text=[None] * batch_size,
            math_modes=[False] * batch_size,
        )
        processed = fp.processor(batch_input, padding_side="left", device=model.device)

        input_ids = processed["input_ids"].to(dtype=torch.long)
        attention_mask = processed["attention_mask"].to(dtype=torch.long)
        position_ids = processed["position_ids"].to(dtype=torch.long)
        image_tiles = processed["image_tiles"].to(dtype=model.dtype)
        grid_thw = processed["grid_thw"].to(dtype=torch.long)

        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(
                pixel_values=image_tiles,
                grid_thw=grid_thw,
                encoder_chunk_size=4096,
                valid_batch_size=batch_size,
            )

        cache_position = (
            torch.arange(input_ids.shape[1], dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        if dtype_override is not None:
            image_embeddings = image_embeddings.to(dtype_override)

        return {
            "input_ids": input_ids,
            "image_embeddings": image_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
