# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 model loader implementation for image to text.
"""

import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AwqConfig,
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


def _patch_vision_pos_embed(model):
    """Patch the vision encoder's pos embed to avoid repeat().

    The TT-XLA runtime translates repeat(n, 1) to concatenation, which fails
    when n=1 (zero extra copies = zero arguments to cat). We replace repeat
    with cat for n>1 and skip entirely for n=1.
    """
    import types

    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is None:
        return

    @torch.compiler.disable
    def patched_fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_cpu = grid_thw.detach().cpu()
        embed_weight = self.pos_embed.weight.detach().cpu()
        grid_thw_list = grid_thw_cpu.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
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
        weight_tensor = torch.tensor(weight_list, dtype=embed_weight.dtype)
        pos_embeds = embed_weight[idx_tensor] * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )
        merge_size = self.config.spatial_merge_size
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            if t > 1:
                pos_embed = torch.cat([pos_embed] * t, dim=0)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    visual.fast_pos_embed_interpolate = types.MethodType(
        patched_fast_pos_embed_interpolate, visual
    )


class ModelVariant(StrEnum):
    """Available Qwen 3 model variants for image to text."""

    QWEN_3_VL_2B_INSTRUCT = "2b_instruct"
    QWEN_3_VL_2B_INSTRUCT_FP8 = "2b_instruct_fp8"
    QWEN_3_VL_2B_THINKING = "2b_thinking"
    QWEN_3_VL_4B_INSTRUCT = "4b_instruct"
    QWEN_3_VL_4B_INSTRUCT_FP8 = "4b_instruct_fp8"
    QWEN_3_VL_4B_THINKING = "4b_thinking"
    QWEN_3_VL_4B_THINKING_FP8 = "4b_thinking_fp8"
    QWEN_3_VL_8B_INSTRUCT = "8b_instruct"
    QWEN_3_VL_8B_INSTRUCT_FP8 = "8b_instruct_fp8"
    QWEN_3_VL_8B_INSTRUCT_AWQ = "8b_instruct_awq"
    QWEN_3_VL_30B_A3B_INSTRUCT = "30b_a3b_instruct"
    QWEN_3_VL_30B_A3B_INSTRUCT_MLX_4BIT = "30b_a3b_instruct_mlx_4bit"
    QWEN_3_VL_32B_INSTRUCT = "32b_instruct"
    QWEN_3_VL_30B_A3B_THINKING = "30b_a3b_thinking"
    UNSLOTH_QWEN_3_VL_4B_INSTRUCT = "unsloth_4b_instruct"


class ModelLoader(ForgeModel):
    """Qwen 3 model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_VL_2B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-2B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_2B_INSTRUCT_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-2B-Instruct-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_2B_THINKING: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-2B-Thinking",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_INSTRUCT_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Instruct-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_THINKING: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Thinking",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_THINKING_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Thinking-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
            max_length=128,
        ),
        ModelVariant.UNSLOTH_QWEN_3_VL_4B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-4B-Instruct",
            max_length=128,
        ),
    }

    # Variants that use the MoE architecture
    _MOE_VARIANTS = {
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT,
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_MLX_4BIT,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_2B_INSTRUCT

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
        group = (
            ModelGroup.VULCAN
            if variant
            in (
                ModelVariant.QWEN_3_VL_4B_INSTRUCT_FP8,
                ModelVariant.QWEN_3_VL_8B_INSTRUCT,
                ModelVariant.QWEN_3_VL_8B_INSTRUCT_FP8,
                ModelVariant.QWEN_3_VL_8B_INSTRUCT_AWQ,
                ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT,
                ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_MLX_4BIT,
                ModelVariant.QWEN_3_VL_32B_INSTRUCT,
                ModelVariant.UNSLOTH_QWEN_3_VL_4B_INSTRUCT,
            )
            else ModelGroup.RED
        )
        return ModelInfo(
            model="qwen_v3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3 model instance for image to text.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "torch_dtype": dtype_override
            if dtype_override is not None
            else torch.float32,
        }

        if self._variant == ModelVariant.QWEN_3_VL_8B_INSTRUCT_AWQ:
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        # MLX/AWQ repos may not ship a processor; fall back to the base model
        if self._variant == ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_MLX_4BIT:
            processor_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        elif self._variant == ModelVariant.QWEN_3_VL_8B_INSTRUCT_AWQ:
            processor_name = "Qwen/Qwen3-VL-8B-Instruct"
        else:
            processor_name = pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(processor_name)

        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if self._variant in self._MOE_VARIANTS
            else Qwen3VLForConditionalGeneration
        )
        model = model_cls.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        _patch_vision_pos_embed(model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
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
        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)
        return inputs
