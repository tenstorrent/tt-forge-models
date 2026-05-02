# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Ministral-3-8B-Instruct-2512-4bit model loader implementation for
multimodal vision-language modeling.

Note: The mlx-community/Ministral-3-8B-Instruct-2512-4bit model is an MLX-quantized
variant of mistralai/Ministral-3-8B-Instruct-2512. Since MLX models cannot be loaded
directly with transformers, this loader uses the base mistralai/Ministral-3-8B-Instruct-2512
model.
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    """Available mlx-community/Ministral-3-8B-Instruct-2512-4bit model variants."""

    MINISTRAL_3_8B_INSTRUCT_2512_4BIT = "Ministral-3-8B-Instruct-2512-4bit"


class ModelLoader(ForgeModel):
    """mlx-community/Ministral-3-8B-Instruct-2512-4bit model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_4BIT: LLMModelConfig(
            pretrained_model_name="mistralai/Ministral-3-8B-Instruct-2512",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_4BIT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ministral-3-8B-Instruct-2512-4bit",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ministral-3-8B-Instruct-2512 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ministral-3-8B-Instruct-2512 model instance.
        """
        import sys
        import types
        import torch
        from transformers import Mistral3ForConditionalGeneration

        # transformers 5.x imports triton unconditionally in finegrained_fp8.py even on CPU;
        # stub it out so loading succeeds — actual FP8 kernels are skipped on non-CUDA systems
        if "triton" not in sys.modules:
            try:
                import triton  # noqa: F401
            except ImportError:

                def _jit(*args, **kwargs):
                    return args[0] if args and callable(args[0]) else lambda fn: fn

                class _AnyAttr:
                    def __getattr__(self, name):
                        return _AnyAttr()

                    def __call__(self, *a, **k):
                        return _AnyAttr()

                mock_triton = types.ModuleType("triton")
                mock_triton.jit = _jit
                mock_triton.language = _AnyAttr()
                sys.modules["triton"] = mock_triton
                sys.modules["triton.language"] = mock_triton.language

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

        model.eval()

        # Patch generate_block_attention_mask globally to avoid in-place XLA tensor
        # mutation.  The original does causal_mask[start:end, start:end] = 0 on a TT
        # tensor inside a for loop (iterating TT tensor elements → Dynamo graph break →
        # torch_xla.sync() → INTERNAL Error code: 13).
        # Fix: for a single image the mask is all-zeros; for multiple images build on CPU.
        import transformers.models.pixtral.modeling_pixtral as _pixtral_mod

        def _fixed_generate_block_attention_mask(patch_embeds_list, tensor):
            dtype = tensor.dtype
            device = tensor.device
            seq_len = tensor.shape[1]
            if len(patch_embeds_list) == 1:
                return torch.zeros(
                    (tensor.shape[0], 1, seq_len, seq_len), dtype=dtype, device=device
                )
            d_min = torch.finfo(dtype).min
            causal_mask_cpu = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype)
            start = 0
            for block_len in patch_embeds_list:
                end = start + int(block_len)
                causal_mask_cpu[start:end, start:end] = 0.0
                start = end
            return causal_mask_cpu.to(device)[None, None, :, :].expand(
                tensor.shape[0], 1, -1, -1
            )

        _pixtral_mod.generate_block_attention_mask = _fixed_generate_block_attention_mask

        # Patch get_image_features on model.model (Mistral3Model) to compute
        # split_sizes on CPU in int64.  On TT, int64 arithmetic uses bfloat16
        # internally: bfloat16(2310) == 2320, which makes split_with_sizes fail
        # with "expects sum 2310, got split_sizes=[2320]".
        def _patched_get_image_features(
            self,
            pixel_values,
            image_sizes,
            vision_feature_layer=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            if vision_feature_layer is None:
                vision_feature_layer = self.config.vision_feature_layer
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            image_outputs = self.vision_tower(
                pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            else:
                hs_pool = [
                    image_outputs.hidden_states[layer_idx]
                    for layer_idx in vision_feature_layer
                ]
                selected_image_feature = torch.cat(hs_pool, dim=-1)
            image_features = self.multi_modal_projector(
                selected_image_feature.squeeze(0), image_sizes
            )
            downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
            split_sizes = (
                torch.as_tensor(image_sizes, dtype=torch.int64).cpu() // downsample_ratio
            ).prod(dim=-1).tolist()
            image_features = torch.split(image_features.squeeze(0), split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        model.model.get_image_features = types.MethodType(
            _patched_get_image_features, model.model
        )

        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the Ministral-3-8B-Instruct-2512 model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from PIL import Image
        from ...tools.utils import cast_input_to_type, get_file

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
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

        return shard_specs
