# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
AdaVaR model wrapper for extracting logits from model outputs.

The Qwen2.5-VL-7B visual encoder (hidden_size=1280, 32 blocks) exceeds TT L1
memory (1,745,920 B > 1,572,864 B limit) and cannot be compiled for TT hardware.
Strategy:
  1. Keep the visual encoder on CPU by overriding _apply() to temporarily remove it
     from the model's module tree during any device transfer (to_device() calls
     nn.Module.to() which calls _apply(), plus torch.compile's internal _apply() calls).
  2. Use @torch._dynamo.disable on _precompute_embeddings to create a true graph break,
     preventing torch.compile from compiling the visual encoder into the TT graph.
  3. Only the language model part is compiled for TT.
"""

import torch
import torch._dynamo


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Clone embed_tokens weight to CPU before model is moved to TT device.
        # Stored in __dict__ (bypasses nn.Module registration) so _apply() won't move it.
        embed_weight = model.model.language_model.embed_tokens.weight.detach().clone()
        self.__dict__["_embed_weight_cpu"] = embed_weight

    def _apply(self, fn, recurse=True):
        # Temporarily remove visual from model's module tree so any device-transfer fn
        # (from nn.Module.to(), torch.compile internals, etc.) won't move its weights to TT.
        # nn.Module.to() calls _apply() — overriding _apply() covers all paths including
        # compiled_executable.to(device) which bypasses our to() override.
        visual = self.model.model._modules.pop("visual", None)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            if visual is not None:
                self.model.model._modules["visual"] = visual
        return result

    @torch._dynamo.disable
    def _precompute_embeddings(self, pixel_values, image_grid_thw, input_ids, attention_mask):
        """Run visual encoder and compute merged inputs_embeds + position_ids.

        @torch._dynamo.disable creates a true graph break: torch.compile skips compiling
        this method for TT. The visual encoder (too large for TT L1) runs eagerly on CPU.
        All CPU computation happens here, and the results are moved to the target device
        (attention_mask.device) before returning, so the compiled TT forward receives
        proper XLA tensors rather than CPU tensors.
        After this returns, compilation resumes for the language model forward.
        """
        visual = self.model.model.visual  # CPU (protected by _apply() override)
        embed_weight_cpu = self.__dict__["_embed_weight_cpu"]  # CPU clone

        # Move inputs to CPU (eager TT sync happens here for any pending ops)
        pixel_values_cpu = pixel_values.cpu()
        image_grid_thw_cpu = image_grid_thw.cpu()
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu()

        # Run visual encoder on CPU
        with torch.no_grad():
            pixel_values_typed = pixel_values_cpu.type(visual.dtype)
            vision_out = visual(pixel_values_typed, grid_thw=image_grid_thw_cpu, return_dict=True)
            split_sizes = (
                image_grid_thw_cpu.prod(-1) // visual.spatial_merge_size**2
            ).tolist()
            image_embeds = torch.cat(
                torch.split(vision_out.pooler_output, split_sizes), dim=0
            )

        # Compute text embeddings on CPU using pre-cloned embed weight
        inputs_embeds = torch.nn.functional.embedding(input_ids_cpu, embed_weight_cpu)

        # Merge image embeddings into text embeddings at image token positions
        image_token_id = self.model.config.image_token_id
        image_mask = (input_ids_cpu == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(inputs_embeds.dtype)
        )

        # Compute 3D RoPE position IDs on CPU (uses .tolist() internally)
        position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids_cpu,
            image_grid_thw=image_grid_thw_cpu,
            video_grid_thw=None,
            attention_mask=attention_mask_cpu,
        )

        # Move to target device (TT in compiled mode, CPU in golden mode) before returning.
        # This ensures the compiled TT graph receives proper XLA tensors rather than CPU
        # tensors, which would be force-converted by the TT backend with precision loss.
        target_device = attention_mask.device
        return inputs_embeds.to(target_device), position_ids.to(target_device)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # Run visual encoder + embedding merge eagerly (disabled from TT compilation).
        # Returns tensors already on the target device (TT in compiled mode).
        inputs_embeds, position_ids = self._precompute_embeddings(
            pixel_values, image_grid_thw, input_ids, attention_mask
        )

        # Forward with pixel_values=None skips visual encoder inside model.forward,
        # only compiles/runs the language model on TT.
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
