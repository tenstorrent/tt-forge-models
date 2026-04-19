# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import transformers.utils.output_capturing as oc

_original_reset = oc.CompileableContextVar.reset


def _safe_reset(self, token):
    if token is None:
        self.global_var = None
        self.compiling = False
    else:
        _original_reset(self, token)


oc.CompileableContextVar.reset = _safe_reset


class SamModelWrapper(nn.Module):
    """Wrapper that calls SAM sub-components directly to avoid the
    top-level SamModel.forward which is incompatible with torch.compile."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_points):
        image_positional_embeddings = self.model.get_image_wide_positional_embeddings()
        batch_size = pixel_values.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        vision_outputs = self.model.vision_encoder(pixel_values)
        image_embeddings = vision_outputs[0]

        input_labels = torch.ones_like(
            input_points[:, :, :, 0], dtype=torch.int, device=input_points.device
        )

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=None,
            input_masks=None,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            attention_similarity=None,
            target_embedding=None,
        )

        return iou_predictions, low_res_masks
