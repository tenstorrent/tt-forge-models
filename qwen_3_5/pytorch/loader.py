# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for multimodal modeling.
"""

from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import cast_input_to_type


def _patched_vision_attn_forward(
    self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
):
    """Monkey-patched Qwen3VLVisionAttention.forward that replaces the non-flash
    split/loop path with a single attention call using a block-diagonal mask.

    The original uses lengths.tolist() + torch.split + a Python loop over chunks,
    which triggers aten._local_scalar_dense graph breaks in every vision block (27×).
    This version builds a block-diagonal mask from cu_seqlens using compilable ops
    (searchsorted, where) so each token only attends within its own sequence segment.
    Mathematically equivalent, fully compilable.
    """
    import sys

    mod = sys.modules[type(self).__module__]
    apply_rotary_pos_emb_vision = mod.apply_rotary_pos_emb_vision
    eager_attention_forward = mod.eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    # Block-diagonal mask: each token attends only within its sequence segment.
    # Uses searchsorted to map positions → segment ids without .tolist().
    positions = torch.arange(seq_length, device=cu_seqlens.device)
    seg_ids = torch.searchsorted(cu_seqlens, positions, right=True) - 1
    block_mask = seg_ids.unsqueeze(0) == seg_ids.unsqueeze(1)
    attention_mask = torch.full(
        (seq_length, seq_length),
        torch.finfo(query_states.dtype).min,
        device=query_states.device,
        dtype=query_states.dtype,
    )
    attention_mask[block_mask] = 0
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        is_causal=False,
        **kwargs,
    )

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


def _patched_get_image_features(
    self, pixel_values, image_grid_thw=None, **kwargs
):
    """Monkey-patched Qwen3_5Model.get_image_features that replaces
    .tolist() + torch.split with torch.tensor_split using cumulative indices.
    Eliminates the aten._local_scalar_dense graph break, fully compilable.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    vision_output = self.visual(
        pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
    )
    image_embeds = vision_output.pooler_output
    split_indices = (
        (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2)
        .cumsum(0)[:-1]
    )
    image_embeds = torch.tensor_split(image_embeds, split_indices, dim=0)
    vision_output.pooler_output = image_embeds
    return vision_output


class ModelVariant(StrEnum):
    """Available Qwen 3.5 multimodal model variants."""

    QWEN_3_5_27B = "Qwen/Qwen3.5-27B"
    QWEN_3_5_35B_A3B = "Qwen/Qwen3.5-35B-A3B"
    QWEN_3_5_122B_A10B = "Qwen/Qwen3.5-122B-A10B"
    QWEN_3_5_397B_A17B = "Qwen/Qwen3.5-397B-A17B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_27B),
        ),
        ModelVariant.QWEN_3_5_35B_A3B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_35B_A3B),
        ),
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_122B_A10B),
        ),
        ModelVariant.QWEN_3_5_397B_A17B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_397B_A17B),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = "What animal is on the candy?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        group = ModelGroup.RED

        return ModelInfo(
            model="qwen_3_5_multimodal",
            variant=variant,
            group=group,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.5 model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Patch .tolist() calls with compilable alternatives on the actual classes
        # used by the loaded model (avoids class-name guessing in auto-generated code).
        # Vision attention: block-diagonal mask instead of split/loop (27× graph breaks).
        # get_image_features: tensor_split with cumulative indices instead of .tolist().
        attn_cls = type(model.model.visual.blocks[0].attn)
        attn_cls.forward = _patched_vision_attn_forward
        type(model.model).get_image_features = _patched_get_image_features

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the Qwen 3.5 multimodal model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_url = image_url or self.sample_image_url
        text_prompt = prompt or self.sample_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names) where mesh_shape is (batch_dim, model_dim)
                   and mesh_axis_names are ("batch", "model").
        """
        if num_devices == 32:  # Galaxy
            mesh_shape = (8, 4)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.visual.blocks:
            shard_specs[layer.attn.qkv.weight] = ("model", "batch")
            shard_specs[layer.attn.proj.weight] = ("batch", "model")
            shard_specs[layer.mlp.linear_fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.linear_fc2.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            if hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[expert.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")

        return shard_specs
