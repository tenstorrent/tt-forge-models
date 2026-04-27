# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Cadrille model wrapper for extracting logits from model outputs.

The Qwen2-VL-2B visual encoder (embed_dim=1280, depth=32) exceeds TT L1
memory limits and cannot be compiled for TT hardware.
Strategy:
  1. Keep the visual encoder on CPU by overriding _apply() to temporarily
     remove it from the module tree during device transfers.
  2. Use @torch._dynamo.disable on _precompute_embeddings to create a graph
     break so the visual encoder runs eagerly on CPU outside the TT graph.
  3. Only the language model is compiled and run on TT.
"""

import torch
import torch._dynamo


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Clone embed_tokens weight to CPU, stored outside nn.Module tree so
        # _apply() won't move it to TT device.
        embed_weight = model.model.language_model.embed_tokens.weight.detach().clone()
        self.__dict__["_embed_weight_cpu"] = embed_weight

    def _apply(self, fn, recurse=True):
        # Temporarily remove visual encoder from module tree during any device
        # transfer to prevent its weights from being moved to TT device.
        visual = self.model.model._modules.pop("visual", None)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            if visual is not None:
                self.model.model._modules["visual"] = visual
        return result

    @torch._dynamo.disable
    def _precompute_embeddings(self, pixel_values, image_grid_thw, input_ids, attention_mask):
        """Run visual encoder on CPU and return merged inputs_embeds + position_ids.

        @torch._dynamo.disable creates a graph break: torch.compile skips this
        method so the visual encoder runs eagerly on CPU. Results are moved to
        the target device before returning so the compiled TT graph receives
        proper XLA tensors.
        """
        visual = self.model.model.visual  # stays on CPU via _apply() override
        embed_weight_cpu = self.__dict__["_embed_weight_cpu"]

        pixel_values_cpu = pixel_values.cpu()
        image_grid_thw_cpu = image_grid_thw.cpu()
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu()

        with torch.no_grad():
            pixel_values_typed = pixel_values_cpu.type(visual.dtype)
            vision_out = visual(pixel_values_typed, grid_thw=image_grid_thw_cpu, return_dict=True)
            split_sizes = (
                image_grid_thw_cpu.prod(-1) // visual.spatial_merge_size**2
            ).tolist()
            image_embeds = torch.cat(
                torch.split(vision_out.pooler_output, split_sizes), dim=0
            )

        inputs_embeds = torch.nn.functional.embedding(input_ids_cpu, embed_weight_cpu)

        image_token_id = self.model.config.image_token_id
        image_mask = (input_ids_cpu == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(inputs_embeds.dtype)
        )

        position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids_cpu,
            image_grid_thw=image_grid_thw_cpu,
            video_grid_thw=None,
            attention_mask=attention_mask_cpu,
        )

        target_device = attention_mask.device
        return inputs_embeds.to(target_device), position_ids.to(target_device)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs):
        inputs_embeds, position_ids = self._precompute_embeddings(
            pixel_values, image_grid_thw, input_ids, attention_mask
        )
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            image_grid_thw=None,
            use_cache=False,
        )
        return outputs.logits
