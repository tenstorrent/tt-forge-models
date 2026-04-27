# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
AIN model wrapper for extracting logits from model outputs.
"""

import copy
import torch


def _xla_int_to_cpu(tensor):
    """Transfer an integer tensor from XLA to CPU using float32 as intermediary.

    TT XLA cannot transfer int32/int64 tensors from device to host (error code 13).
    Float32 transfers succeed. For integer values < 2^24 = 16,777,216 (token IDs,
    grid dimensions, attention masks), int→float32 is lossless.
    """
    return tensor.float().cpu().long()


@torch.compiler.disable
def _rot_pos_emb_cpu(grid_thw, spatial_merge_size, rotary_pos_emb_cpu):
    """Run rot_pos_emb entirely on CPU to avoid XLA arange/repeat failures.

    Qwen2VL's rot_pos_emb calls torch.arange(h) where h is extracted from
    the image_grid_thw input tensor. When grid_thw is on the XLA (TT) device,
    torch.arange(xla_tensor) dispatches via __torch_function__ and fails with
    error code 13. This function runs the entire computation on CPU and moves
    the result back to the original device.
    """
    original_device = grid_thw.device
    if original_device.type != "cpu":
        # Use float32 intermediary: TT XLA can't transfer int32/int64 to CPU
        grid_thw_cpu = _xla_int_to_cpu(grid_thw)
    else:
        grid_thw_cpu = grid_thw.long()

    pos_ids = []
    for t, h, w in grid_thw_cpu:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw_cpu[:, 1:].max()
    rotary_pos_emb_full = rotary_pos_emb_cpu(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

    if original_device.type != "cpu":
        rotary_pos_emb = rotary_pos_emb.to(original_device)
    return rotary_pos_emb


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._patch_visual_cpu()
        self._patch_rot_pos_emb()
        self._patch_get_image_features()
        self._patch_get_rope_index()

    def _patch_visual_cpu(self):
        """Keep the visual encoder on CPU to avoid integer XLA operations.

        Qwen2VLVisionTransformer.forward computes cu_seqlens via
        torch.repeat_interleave with integer tensors. TT XLA may not support
        this, and the subsequent split_sizes / rot_pos_emb computations also
        require integer CPU access. Keeping the entire visual encoder on CPU
        avoids all of these issues.

        Removes visual from Qwen2VLModel._modules so Module.to('xla') never
        migrates it, then stores the CPU copy as a plain __dict__ attribute
        via object.__setattr__.
        """
        qwen2vl_model = self.model.model  # Qwen2VLModel
        visual_cpu = copy.deepcopy(qwen2vl_model.visual).cpu()
        # Remove from _modules so to('xla') skips it
        qwen2vl_model._modules.pop("visual", None)
        # Store as non-module attribute; bypasses Module.__setattr__
        object.__setattr__(qwen2vl_model, "visual", visual_cpu)

    def _patch_rot_pos_emb(self):
        """CPU-offload rot_pos_emb to avoid XLA arange failures.

        Stores a CPU copy of visual.rotary_pos_emb outside _modules so
        Module.to('xla') never migrates it. The patched rot_pos_emb runs
        entirely on CPU and moves its result back to XLA.
        """
        visual = self.model.model.visual  # CPU visual (via __dict__ after _patch_visual_cpu)
        rotary_pos_emb_cpu = copy.deepcopy(visual.rotary_pos_emb).cpu()
        object.__setattr__(self, "_rotary_pos_emb_cpu", rotary_pos_emb_cpu)

        spatial_merge_size = visual.spatial_merge_size
        _cpu_ref = rotary_pos_emb_cpu

        def patched_rot_pos_emb(grid_thw):
            return _rot_pos_emb_cpu(grid_thw, spatial_merge_size, _cpu_ref)

        visual.rot_pos_emb = patched_rot_pos_emb

    def _patch_get_image_features(self):
        """CPU-offload visual feature extraction to avoid integer XLA operations.

        The visual encoder (rot_pos_emb, cu_seqlens) uses integer tensors that
        TT XLA cannot transfer from device to host. Running the visual encoder
        on CPU (moved there by _patch_visual_cpu) and computing split_sizes on
        CPU avoids all integer XLA operations in the vision path.
        """
        qwen2vl_model = self.model.model  # Qwen2VLModel

        @torch.compiler.disable
        def patched_get_image_features(pixel_values, image_grid_thw=None, **kwargs):
            visual = qwen2vl_model.visual  # CPU visual encoder
            spatial_merge_size = visual.spatial_merge_size

            # Transfer float32 tensor to CPU (XLA float32 → CPU works)
            pixel_values_cpu = pixel_values.cpu()

            # Transfer integer tensor to CPU via float32 intermediary
            if image_grid_thw.device.type != "cpu":
                grid_thw_cpu = _xla_int_to_cpu(image_grid_thw)
            else:
                grid_thw_cpu = image_grid_thw.long()

            # Run visual encoder entirely on CPU
            pixel_values_typed = pixel_values_cpu.type(visual.dtype)
            vision_outputs = visual(
                pixel_values_typed, grid_thw=grid_thw_cpu, return_dict=True, **kwargs
            )

            # Compute split_sizes from CPU int64 grid_thw (avoids .tolist() on XLA)
            split_sizes = [
                int(s)
                for s in (grid_thw_cpu.prod(-1) // spatial_merge_size**2).tolist()
            ]
            image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
            vision_outputs.pooler_output = image_embeds
            return vision_outputs

        qwen2vl_model.get_image_features = patched_get_image_features

    def _patch_get_rope_index(self):
        """CPU-offload rope index computation to avoid int64 XLA→CPU transfers.

        get_rope_index uses input_ids.tolist() and image_grid_thw.item() which
        require device→host transfers. TT XLA cannot transfer int64 tensors to
        CPU (error code 13). We transfer via float32 intermediary and run the
        entire rope index computation on CPU, then return results to XLA.
        """
        qwen2vl_model = self.model.model  # Qwen2VLModel
        original_get_rope_index = qwen2vl_model.get_rope_index

        @torch.compiler.disable
        def patched_get_rope_index(
            input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            **kwargs,
        ):
            original_device = input_ids.device

            if original_device.type != "cpu":
                input_ids_cpu = _xla_int_to_cpu(input_ids)
                grid_thw_cpu = (
                    _xla_int_to_cpu(image_grid_thw)
                    if image_grid_thw is not None
                    else None
                )
                attn_mask_cpu = (
                    _xla_int_to_cpu(attention_mask)
                    if attention_mask is not None
                    else None
                )
            else:
                input_ids_cpu = input_ids.long()
                grid_thw_cpu = (
                    image_grid_thw.long() if image_grid_thw is not None else None
                )
                attn_mask_cpu = (
                    attention_mask.long() if attention_mask is not None else None
                )

            pos_ids, rope_deltas = original_get_rope_index(
                input_ids_cpu,
                image_grid_thw=grid_thw_cpu,
                video_grid_thw=video_grid_thw,
                attention_mask=attn_mask_cpu,
                **kwargs,
            )

            if original_device.type != "cpu":
                pos_ids = pos_ids.to(original_device)
                rope_deltas = rope_deltas.to(original_device)

            return pos_ids, rope_deltas

        qwen2vl_model.get_rope_index = patched_get_rope_index

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            **kwargs,
        }
        outputs = self.model(**inputs)
        return outputs.logits
