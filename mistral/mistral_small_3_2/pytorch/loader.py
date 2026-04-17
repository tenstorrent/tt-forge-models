# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 3.2 model loader implementation for multimodal vision-language modeling.
"""

from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Mistral Small 3.2 model variants."""

    MISTRAL_SMALL_3_2_24B_INSTRUCT = "unsloth/Mistral-Small-3.2-24B-Instruct-2506"


class ModelLoader(ForgeModel):
    """Mistral Small 3.2 model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mistral_small_3_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        from transformers import AutoProcessor

        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    @staticmethod
    def _patch_pixtral_attention():
        """Patch PixtralAttention to avoid ambiguous reshape with zero-element tensors.

        torch.compile with fake tensors fails on reshape(batch, 0, -1) because -1
        is ambiguous when the tensor has 0 elements. Replace -1 with embed_dim.

        Patches at the class level so torch._dynamo sees the fix during tracing.
        """
        from transformers.models.pixtral.modeling_pixtral import (
            PixtralAttention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def _patched_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            output_attentions=False,
            **kwargs,
        ):
            batch_size, patches, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                batch_size, patches, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                batch_size, patches, self.num_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size, patches, self.num_heads, self.head_dim
            ).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=0
            )

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(
                batch_size, patches, self.embed_dim
            ).contiguous()
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights

        PixtralAttention.forward = _patched_forward

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mistral Small 3.2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Mistral Small 3.2 model instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        self._patch_pixtral_attention()

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
        """Load and return sample inputs for the Mistral Small 3.2 model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from PIL import Image
        from ....tools.utils import cast_input_to_type, get_file

        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = cast_input_to_type(
                    inputs["pixel_values"], dtype_override
                )

        # Convert image_sizes from a tensor to Python int tuples so torch._dynamo
        # uses concrete integer values rather than fake-tensor zeros during tracing.
        # Without this, the pixtral patch merger receives h=0,w=0 (from 0//patch_size)
        # which causes unfold to fail on a [1, hidden, 0, 0] tensor.
        if "image_sizes" in inputs and inputs["image_sizes"] is not None:
            inputs["image_sizes"] = [
                tuple(int(v) for v in size) for size in inputs["image_sizes"]
            ]

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
        shard_specs = {}

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        for layer in model.model.vision_tower.transformer.layers:
            # Feed-forward (PixtralMLP)
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

            # Attention (PixtralAttention)
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        return shard_specs
