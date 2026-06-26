# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper exposing only the dots.ocr ``dots_vit`` vision tower, returning the
merged patch embeddings as a single tensor.

Device note (n150, tt-xla): the tower does not pass on device as written. The
patch-embed ``Conv2d`` (3->1536 ch, kernel=stride=14) trips a tt-metal conv2d
runtime assert ("Reader indices buffer page size 520 exceeds worst-case CB size
256", conv2d_op_program_factory_common.cpp:311). The conv is equivalent to a
linear projection of a single 14x14 patch; substituting that linear lets the
tower compile and execute, but the bf16 result then diverges from the CPU
golden (PCC ~0.64) - the deep RMSNorm/residual stack is bf16-sensitive and the
fp32-accumulation knob is not settable through the test runner (issue #2861).
This wrapper keeps the faithful conv so the loader represents the real model;
see the bringup report for the full diagnosis.
"""
import torch


class VisionTowerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower

    def forward(self, pixel_values, image_grid_thw):
        # Returns merged vision embeddings: [num_merged_patches, hidden_size].
        return self.vision_tower(pixel_values, image_grid_thw)
