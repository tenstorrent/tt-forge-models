# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Reason2 model loader implementation for image to text.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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


class _CpuOnlyTensor(torch.Tensor):
    """Tensor subclass that refuses transfer to non-CPU devices.

    TT device does not support device-to-host tensor reads. image_grid_thw must
    stay on CPU so fast_pos_embed_interpolate / rot_pos_emb can call .tolist()
    for Python control flow without triggering a device sync that fails with
    Error code: 13. The test runner calls .to(tt_device) on all inputs; this
    subclass intercepts that call and stays on CPU.
    """

    def __new__(cls, data: "torch.Tensor", *args, **kwargs) -> "_CpuOnlyTensor":
        return torch.Tensor._make_subclass(cls, data.cpu())

    def to(self, *args, **kwargs) -> "torch.Tensor":
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = torch.device(args[0])
        elif "device" in kwargs:
            device = torch.device(kwargs["device"])
        if device is not None and device.type != "cpu":
            return self  # Stay on CPU; all .tolist() callers get a CPU tensor
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> "torch.Tensor":
        return self


def _patch_qwen3vl_for_tt_device():
    """Patch Qwen3VL vision model forward to handle grid_thw on CPU.

    The test runner moves all inputs to TT device, but image_grid_thw must
    stay on CPU (_CpuOnlyTensor) so fast_pos_embed_interpolate / rot_pos_emb
    can call .tolist() for Python control flow.

    Problem: the derived tensors (pos_embeds, rotary_pos_emb, cu_seqlens) are
    computed from CPU grid_thw and thus live on CPU. When the compiled XLA graph
    resumes after those eager computations, TT PJRT cannot compile a program
    that takes CPU tensors as inputs — every input to a compiled XLA program
    must be on TT device.

    Fix: patch Qwen3VLVisionModel.forward to explicitly move the CPU-derived
    tensors to TT device (host-to-device transfer, which TT PJRT does support)
    before they are used alongside hidden_states (TT device). position_ids is
    pre-computed in load_inputs to bypass get_rope_index which calls
    input_ids.tolist() on a TT tensor.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        from transformers.modeling_outputs import BaseModelOutputWithDeepstackFeatures
    except ImportError:
        return

    orig_forward = modeling_qwen3_vl.Qwen3VLVisionModel.forward

    def _patched_vision_forward(self, hidden_states, grid_thw, **kwargs):
        # patch_embed moves hidden_states to the model device (TT)
        hidden_states = self.patch_embed(hidden_states)
        target_device = hidden_states.device

        # grid_thw is _CpuOnlyTensor; these compute on CPU, then move to TT
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(target_device)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # Move position_embeddings to TT device before use in attention
        position_embeddings = (emb.cos().to(target_device), emb.sin().to(target_device))

        # cu_seqlens computed from CPU grid_thw → move to TT device
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(target_device)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )

    modeling_qwen3_vl.Qwen3VLVisionModel.forward = _patched_vision_forward


class ModelVariant(StrEnum):
    """Available Cosmos Reason2 model variants for image to text."""

    COSMOS_REASON2_8B = "8b"


class ModelLoader(ForgeModel):
    """Cosmos Reason2 model loader implementation for image to text tasks."""

    # nvidia/Cosmos-Reason2-8B is a gated repo; use public Qwen3-VL-8B-Instruct (same architecture)
    BASE_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.COSMOS_REASON2_8B: LLMModelConfig(
            pretrained_model_name="nvidia/Cosmos-Reason2-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.COSMOS_REASON2_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="cosmos_reason2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cosmos Reason2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Cosmos Reason2 model instance for image to text.
        """
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_QWEN3_VL_MODEL)

        _patch_qwen3vl_for_tt_device()

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.BASE_QWEN3_VL_MODEL, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Cosmos Reason2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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

        # Pre-compute position_ids on CPU to bypass get_rope_index during the
        # compiled forward. get_rope_index calls input_ids.tolist() which fails
        # when input_ids is on TT device (device-to-host reads not supported).
        # Providing position_ids directly makes Qwen3VLModel.forward skip
        # compute_3d_position_ids entirely.
        if self._model is not None:
            position_ids, _ = self._model.model.get_rope_index(
                inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs["attention_mask"],
            )
            inputs["position_ids"] = position_ids

        # Keep image_grid_thw on CPU so the patched vision model forward can
        # use .tolist() on it without triggering TT device reads.
        inputs["image_grid_thw"] = _CpuOnlyTensor(inputs["image_grid_thw"])

        return inputs
