# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tongyi-MAI/MAI-UI-8B model loader implementation for image to text.

MAI-UI-8B is a Qwen3-VL based foundation GUI agent for GUI grounding and
mobile/web navigation tasks.
"""

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_qwen3vl_for_tt_device(model=None):
    """Patch Qwen3 VL methods that call .tolist() on device tensors.

    The test runner moves all input tensors to TT device, but the VisionModel
    and get_rope_index methods call .tolist() on grid_thw / input_ids tensors for
    Python control flow. TT device does not support eager tensor reads — they
    trigger a device sync that fails with INTERNAL Error code: 13. Moving these
    metadata tensors to CPU before the .tolist() calls avoids the sync while
    keeping all actual vision and language computations on TT device.

    fast_pos_embed_interpolate additionally calls torch.tensor(python_list,
    device=self.pos_embed.weight.device). When the model is on TT device,
    creating a TT-device tensor from a Python list fails during XLA graph
    extraction. We pre-capture pos_embed.weight as a CPU tensor before the
    model is moved to TT (by passing model at call time), then use it to
    avoid any eager device tensor construction inside the function.

    model: Qwen3VLForConditionalGeneration on CPU, used to capture
           pos_embed.weight before the model is moved to TT device.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        return

    orig_rot_pos = modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_vl.Qwen3VLModel.get_rope_index
    orig_get_image = modeling_qwen3_vl.Qwen3VLModel.get_image_features

    # Pre-capture pos_embed.weight as a CPU tensor while the model is still on
    # CPU. After the model is moved to TT, we cannot read this weight in eager
    # mode (TT device sync fails). The closure holds a permanent CPU reference.
    _pos_embed_weight_cpu = None
    if model is not None:
        try:
            _pos_embed_weight_cpu = (
                model.model.visual.pos_embed.weight.detach().clone().cpu()
            )
        except AttributeError:
            pass

    @torch.compiler.disable
    def _patched_fast_pos(self, grid_thw):
        # fast_pos_embed_interpolate calls .tolist() (Error code: 13 on TT) and
        # torch.tensor(python_list, device=pos_embed.weight.device) which also
        # fails when device is TT. @torch.compiler.disable causes a graph break
        # before this function; it then runs in eager Python mode.
        # Reimplement the function body entirely on CPU using the pre-captured
        # pos_embed weight. Return the CPU tensor; the resumed Dynamo graph for
        # the continuation of forward handles the host→device transfer implicitly.
        cpu_weight = _pos_embed_weight_cpu
        if cpu_weight is None:
            cpu_weight = self.pos_embed.weight.detach().cpu()

        grid_thw_list = grid_thw.cpu().tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        weight_tensor = torch.tensor(weight_list, dtype=cpu_weight.dtype)
        pos_embeds = (
            torch.nn.functional.embedding(idx_tensor, cpu_weight)
            * weight_tensor[:, :, None]
        )
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)

        # Transfer CPU result to XLA device using send_cpu_data_to_device,
        # which uses the PJRT buffer API and does NOT trigger a standalone
        # extract_compiled_graph/sync() call (unlike .to(device) via torch.compile).
        orig_device = self.pos_embed.weight.device
        if str(orig_device) != "cpu":
            import torch_xla.core.xla_model as xm
            [patch_pos_embeds] = xm.send_cpu_data_to_device(
                [patch_pos_embeds], orig_device
            )
        return patch_pos_embeds

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu())

    def _patched_get_rope(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = orig_get_rope(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    def _patched_get_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return orig_get_image(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            **kwargs,
        )

    # _deepstack_process uses hidden_states[visual_pos_masks, :] to gather
    # image token positions (dynamic output shape) then scatters back.
    # @torch.compiler.disable causes a graph break before this function so it
    # runs in eager Python mode. We do the boolean-indexed update on CPU tensors
    # (safe in eager mode) and return the result via send_cpu_data_to_device,
    # which uses the PJRT buffer API and does NOT trigger a standalone sync.
    @torch.compiler.disable
    def _patched_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        orig_device = hidden_states.device
        hs_cpu = hidden_states.detach().cpu().clone()
        vm_cpu = visual_pos_masks.cpu()
        ve_cpu = visual_embeds.cpu().to(hs_cpu.dtype)
        hs_cpu[vm_cpu] = hs_cpu[vm_cpu] + ve_cpu
        if str(orig_device) != "cpu":
            import torch_xla.core.xla_model as xm
            [hs_cpu] = xm.send_cpu_data_to_device([hs_cpu], orig_device)
        return hs_cpu

    modeling_qwen3_vl.Qwen3VLTextModel._deepstack_process = _patched_deepstack_process

    # get_placeholder_mask calls torch_compilable_check with
    # inputs_embeds[special_image_mask] — a boolean-masked gather with
    # data-dependent output shape that XLA cannot compile. The check is a
    # runtime validation that the CPU run already verified; skip it.
    def _patched_get_placeholder_mask(
        self, input_ids, inputs_embeds, image_features=None, video_features=None
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.video_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        special_image_mask = (
            special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        special_video_mask = (
            special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        return special_image_mask, special_video_mask

    # Qwen3VLModel.forward calls inputs_embeds.masked_scatter(image_mask, image_embeds)
    # to insert image embeddings into the sequence. XLA lowers masked_scatter via
    # SORT + set-dimension-size + SCATTER, and tt-mlir generates a CumSumOp that
    # requires 7.65 GB of DRAM — crashing on the TT device. Replace it with a
    # @torch.compiler.disable CPU path (same pattern as _patched_deepstack_process)
    # and patch forward to call it. The 3D mask[..., 0] slice used for visual_pos_masks
    # and deepstack is preserved unchanged.
    @torch.compiler.disable
    def _cpu_masked_scatter(inputs_embeds, mask_3d, embeds):
        orig_device = inputs_embeds.device
        hs = inputs_embeds.detach().cpu().clone()
        mask_2d = mask_3d[..., 0].detach().cpu()
        em = embeds.detach().cpu().to(hs.dtype)
        hs[mask_2d] = em
        if str(orig_device) != "cpu":
            import torch_xla.core.xla_model as xm
            [hs] = xm.send_cpu_data_to_device([hs], orig_device)
        return hs

    _Qwen3VLModelOutputWithPast = modeling_qwen3_vl.Qwen3VLModelOutputWithPast

    def _patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = _cpu_masked_scatter(inputs_embeds, image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = _cpu_masked_scatter(inputs_embeds, video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return _Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_vl.Qwen3VLModel.get_rope_index = _patched_get_rope
    modeling_qwen3_vl.Qwen3VLModel.get_image_features = _patched_get_image
    modeling_qwen3_vl.Qwen3VLModel.get_placeholder_mask = _patched_get_placeholder_mask
    modeling_qwen3_vl.Qwen3VLModel.forward = _patched_model_forward


class ModelVariant(StrEnum):
    """Available MAI-UI-8B model variants for image to text."""

    MAI_UI_8B = "mai_ui_8b"


class ModelLoader(ForgeModel):
    """MAI-UI-8B model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MAI_UI_8B: LLMModelConfig(
            pretrained_model_name="Tongyi-MAI/MAI-UI-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAI_UI_8B

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    # Standard pixel limits for Qwen VL models to stay within hardware L1 budget
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mai_ui_8b",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Patch .tolist() and torch.tensor(list, device=TT) calls AFTER loading
        # so pos_embed.weight can be captured while the model is still on CPU.
        _patch_qwen3vl_for_tt_device(model=model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
