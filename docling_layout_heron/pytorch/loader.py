# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Docling Layout Heron model loader implementation for document layout detection.
"""
import torch
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


def _patch_rtdetrv2_for_tt_device(model):
    """
    Patch RTDetrV2MultiscaleDeformableAttention.forward to avoid torch.long
    integer arithmetic on the TT device.

    The vanilla forward performs:
        (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length
    where spatial_shapes is a torch.long tensor on the TT device.  TT hardware
    does not support int64 element-wise arithmetic, so this equality check
    evaluates to False and raises ValueError.

    The fix replaces the check with an equivalent Python-integer calculation
    over spatial_shapes_list (a plain list of (height, width) tuples that is
    always passed alongside the tensor) which never touches the TT device.
    """
    import types
    from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
        RTDetrV2MultiscaleDeformableAttention,
        multi_scale_deformable_attention_v2,
    )
    import torch.nn.functional as F

    def _patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs,
    ):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        # Perform the alignment check using Python-integer arithmetic over
        # spatial_shapes_list so that no torch.long computation runs on the
        # TT device (TT hardware doesn't support int64 arithmetic).
        if spatial_shapes_list is not None:
            total_hw = sum(int(h) * int(w) for h, w in spatial_shapes_list)
            if total_hw != int(sequence_length):
                raise ValueError(
                    "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
                )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = (
                self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            )
            offset = (
                sampling_offsets
                * n_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = (
                reference_points[:, :, None, :, :2] + offset
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, "
                f"but got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attention_v2(
            value,
            spatial_shapes_list,
            sampling_locations,
            attention_weights,
            self.n_points_list,
            self.method,
        )

        output = self.output_proj(output)
        return output, attention_weights

    # Patch every RTDetrV2MultiscaleDeformableAttention instance in the model
    for module in model.modules():
        if isinstance(module, RTDetrV2MultiscaleDeformableAttention):
            module.forward = types.MethodType(_patched_forward, module)

    return model


class ModelVariant(StrEnum):
    """Available Docling Layout Heron model variants."""

    DOCLING_LAYOUT_HERON = "docling_layout_heron"
    DOCLING_LAYOUT_HERON_101 = "docling_layout_heron_101"


class ModelLoader(ForgeModel):
    """Docling Layout Heron model loader for document layout detection."""

    _VARIANTS = {
        ModelVariant.DOCLING_LAYOUT_HERON: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-heron",
        ),
        ModelVariant.DOCLING_LAYOUT_HERON_101: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-heron-101",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOCLING_LAYOUT_HERON

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Docling Layout Heron",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import RTDetrImageProcessor

        self.processor = RTDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import RTDetrV2ForObjectDetection

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = RTDetrV2ForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        _patch_rtdetrv2_for_tt_device(model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        import requests
        from PIL import Image

        if self.processor is None:
            self._load_processor()

        url = "https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo/resolve/main/example_images/annual_rep_14.png"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
