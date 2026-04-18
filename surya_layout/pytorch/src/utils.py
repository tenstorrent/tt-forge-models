# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image

os.environ.setdefault("TORCH_DEVICE", "cpu")

import torch._dynamo

_orig_mark_static = torch._dynamo.mark_static_address
torch._dynamo.mark_static_address = lambda *a, **kw: None

from surya.common.surya.decoder.config import SuryaDecoderConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

_original_decoder_config_init = SuryaDecoderConfig.__init__


def _patched_decoder_config_init(self, *args, **kwargs):
    kwargs.setdefault("pad_token_id", 0)
    _original_decoder_config_init(self, *args, **kwargs)


SuryaDecoderConfig.__init__ = _patched_decoder_config_init

if "default" not in ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(config=None, device=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


class SuryaLayoutWrapper(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from surya.foundation import FoundationPredictor

        self._fp = FoundationPredictor(device=device)
        self.vision_encoder = self._fp.model.vision_encoder

        self.eval()
        self.vision_encoder.eval()
        for _, param in self.vision_encoder.named_parameters():
            param.requires_grad = False

    def forward(self, image_tiles: torch.Tensor):
        return self.vision_encoder.embed_images(
            image_batch=image_tiles,
            grid_thw=self._grid_thw.unsqueeze(0).to(image_tiles.device),
        )

    def preprocess_image(self, image: Image.Image):
        from surya.input.processing import convert_if_not_rgb

        img = convert_if_not_rgb([image])[0]
        processed = self._fp.processor.image_processor(img)
        batch_input = self._fp.prepare_input(
            task_names=["layout"],
            images=[processed],
            input_text=[""],
            math_modes=[False],
        )
        proc_result = self._fp.processor(
            batch_input,
            padding_side="left",
            device="cpu",
        )
        image_tiles = proc_result["image_tiles"]
        grid_thw = proc_result["grid_thw"]
        self._grid_thw = grid_thw
        return image_tiles.unsqueeze(0)


def save_outputs_layout(co_out, images, result_path):
    os.makedirs(result_path, exist_ok=True)
    results = {"embeddings_shape": list(co_out.shape)}
    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    logger.info(f"Wrote results to {result_path}")
