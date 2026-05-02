# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RedHatAI Mistral Small 3.1 24B quantized W4A16 model loader implementation for multimodal vision-language modeling.
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
    """Available Mistral Small 3.1 24B quantized W4A16 model variants."""

    MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_QUANTIZED_W4A16 = (
        "24B_Instruct_2503_Quantized_W4A16"
    )


class ModelLoader(ForgeModel):
    """RedHatAI Mistral Small 3.1 24B quantized W4A16 model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_QUANTIZED_W4A16: LLMModelConfig(
            pretrained_model_name="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_QUANTIZED_W4A16

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
            model="Mistral Small 3.1 24B Quantized W4A16",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        # transformers 5.x: PixtralImageProcessor is loaded as fast processor by default;
        # use_fast=False maintains slow processor behaviour matching the saved checkpoint.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=False
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        from transformers import Mistral3ForConditionalGeneration, AutoConfig

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # The checkpoint was saved with old-style flat paths (language_model.*, vision_tower.*,
        # multi_modal_projector.*).  transformers 5.x wraps them under self.model.*, but the
        # compressed-tensors quantization config ignore list still uses old paths.  Fix the
        # ignore list so the quantizer correctly skips the vision tower, projector, and lm_head.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if (
            hasattr(config, "quantization_config")
            and isinstance(config.quantization_config, dict)
            and "ignore" in config.quantization_config
        ):
            updated_ignore = []
            for path in config.quantization_config["ignore"]:
                if path.startswith("vision_tower."):
                    updated_ignore.append("model." + path)
                elif path.startswith("multi_modal_projector."):
                    updated_ignore.append("model." + path)
                elif path == "language_model.lm_head":
                    # Maps to outer lm_head and inner model.language_model.lm_head
                    updated_ignore.append("lm_head")
                    updated_ignore.append("model.language_model.lm_head")
                else:
                    updated_ignore.append(path)
            config.quantization_config["ignore"] = updated_ignore

        model_kwargs = {"device_map": "cpu", "torch_dtype": dtype}
        model_kwargs |= kwargs

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )

        # Dequantize pack-quantized INT4 layers to standard nn.Linear for TT device.
        _dequantize_compressed_linear(model, dtype)

        # Patch get_image_features on Mistral3Model to compute split_sizes on CPU.
        # On TT, int64 arithmetic promotes to bfloat16 internally so prod() of large
        # patch counts (e.g. 2310) rounds to the wrong value; keeping it on CPU avoids this.
        _patch_get_image_features(model)

        # Replace generate_block_attention_mask with a functional version.
        # The stock implementation uses in-place XLA tensor assignment inside a Python loop
        # which causes a Dynamo graph break and INTERNAL Error code 13 on TT.
        _patch_generate_block_attention_mask()

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
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
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


def _dequantize_compressed_linear(model, dtype):
    """Replace pack-quantized Linear modules with standard nn.Linear."""
    import torch.nn as nn
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor

    for parent_module in list(model.modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not (
                isinstance(child_module, nn.Linear)
                and hasattr(child_module, "weight_packed")
            ):
                continue

            state_dict = {"weight_packed": child_module.weight_packed}
            if hasattr(child_module, "weight_scale"):
                state_dict["weight_scale"] = child_module.weight_scale
            if hasattr(child_module, "weight_shape"):
                state_dict["weight_shape"] = child_module.weight_shape
            if hasattr(child_module, "weight_g_idx"):
                state_dict["weight_g_idx"] = child_module.weight_g_idx
            if hasattr(child_module, "weight_zero_point"):
                state_dict["weight_zero_point"] = child_module.weight_zero_point

            decompressed = PackedQuantizationCompressor.decompress(
                state_dict, child_module.quantization_scheme
            )
            weight_fp = decompressed["weight"].to(dtype).contiguous()

            new_linear = nn.Linear(
                child_module.in_features,
                child_module.out_features,
                bias=child_module.bias is not None,
            )
            new_linear.weight = nn.Parameter(weight_fp)
            if child_module.bias is not None:
                new_linear.bias = nn.Parameter(child_module.bias.to(dtype))
            setattr(parent_module, child_name, new_linear)


def _patch_get_image_features(model):
    """Patch Mistral3Model.get_image_features to compute split_sizes on CPU.

    On TT, int64 arithmetic promotes to bfloat16 internally; prod() of patch counts
    rounds large values (e.g. 2310 → 2320) causing split_with_sizes to fail.
    """
    import torch
    from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

    mistral_model = model.model  # Mistral3Model instance

    original_get_image_features = Mistral3Model.get_image_features

    def patched_get_image_features(
        self,
        pixel_values,
        image_sizes,
        vision_feature_layer=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # return_dict is absorbed here so it doesn't collide with the explicit
        # return_dict=True passed to vision_tower below.
        # Fill in config defaults that @merge_with_config_defaults would normally provide.
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
        # Force split_sizes computation on CPU to avoid bfloat16 int64 rounding on TT
        split_sizes = (
            (torch.as_tensor(image_sizes, dtype=torch.int64, device="cpu")
             // downsample_ratio)
            .prod(dim=-1)
            .tolist()
        )
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_outputs.pooler_output = image_features
        return image_outputs

    mistral_model.get_image_features = patched_get_image_features.__get__(
        mistral_model, Mistral3Model
    )


def _patch_generate_block_attention_mask():
    """Replace generate_block_attention_mask with a functional version.

    The stock version uses in-place XLA tensor assignment inside a Python loop,
    causing a Dynamo graph break and INTERNAL Error code 13 on TT.
    """
    import torch
    import transformers.models.pixtral.modeling_pixtral as pixtral_module

    def functional_generate_block_attention_mask(patch_embeds_list, tensor):
        dtype = tensor.dtype
        seq_len = tensor.shape[1]
        d_min = torch.finfo(dtype).min

        if len(patch_embeds_list) == 1:
            # Single image: all patches can attend to each other → all-zero mask
            causal_mask = torch.zeros(seq_len, seq_len, dtype=dtype)
        else:
            # Multi-image: build mask on CPU to avoid in-place XLA mutation
            causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype)
            block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
            block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
            for start, end in zip(block_start_idx.tolist(), block_end_idx.tolist()):
                causal_mask[start:end, start:end] = 0

        causal_mask = causal_mask.to(device=tensor.device)
        return causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)

    pixtral_module.generate_block_attention_mask = functional_generate_block_attention_mask
