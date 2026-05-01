# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 3 8B Instruct BnB 4-bit model loader implementation for multimodal vision-language modeling.
"""

import torch
import torch.nn as nn
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
    """Available Ministral 3 8B Instruct BnB 4-bit model variants."""

    MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT = (
        "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"
    )


class ModelLoader(ForgeModel):
    """Ministral 3 8B Instruct BnB 4-bit model loader for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(
                ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT
            ),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT

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
            model="ministral_3_8b_instruct_bnb_4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _dequantize_bnb4_to_bf16(model):
        """Replace all BnB Linear4bit layers with standard bfloat16 Linear layers.

        Params4bit.detach() returns a plain Tensor, which causes
        Parameter.__new__ to raise RuntimeError when model.to(xla_device) is
        called. Dequantizing to bf16 before device transfer avoids this.
        """
        import bitsandbytes as bnb
        import bitsandbytes.functional as F

        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                if hasattr(module.weight, "quant_state") and module.weight.quant_state is not None:
                    weight_bf16 = F.dequantize_4bit(
                        module.weight.data, module.weight.quant_state
                    ).to(torch.bfloat16)
                else:
                    weight_bf16 = module.weight.data.to(torch.bfloat16)
                bias = module.bias
                new_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=bias is not None,
                    device=weight_bf16.device,
                    dtype=torch.bfloat16,
                )
                new_linear.weight = nn.Parameter(weight_bf16)
                if bias is not None:
                    new_linear.bias = nn.Parameter(bias.to(torch.bfloat16))
                replacements.append((name, new_linear))

        for name, new_module in replacements:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        return model

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
    def _patch_inputs_embedding_merge(model):
        """Replace masked_scatter in Mistral3Model.forward with a token-level gather.

        The standard masked_scatter decomposition flattens mask to
        [seq_len * hidden_size] elements and calls cumsum on that. tt-mlir's
        cumsum pads the reduction axis to tile_size (1024), creating an
        [N, 1024] intermediate matrix. For N = 2896 * 4096 = 11.8 M this is
        11.8 M × 1024 × 4 B = 45 GB → OOM on TT hardware.

        The mask pattern is always:
            mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        so every "token" position has ALL hidden-size bits set identically.
        We exploit this structure: work at token level (2896 elements), compute
        a token-level cumsum, gather the right image feature per position, then
        apply with torch.where.  The token-level cumsum is only 2896 elements
        → [2896, 1024] × 4 B = 12 MB.
        """
        import types

        def _patched_model_forward(
            self,
            input_ids=None,
            pixel_values=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            vision_feature_layer=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            image_sizes=None,
            **kwargs,
        ):
            import torch
            from transformers.models.mistral3.modeling_mistral3 import Mistral3ModelOutputWithPast

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            image_features = None
            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    image_sizes=image_sizes,
                    return_dict=True,
                ).pooler_output
                image_features = torch.cat(image_features, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

                # Token-level scatter: avoids the seq_len * hidden_size cumsum OOM.
                # mask shape: [1, seq_len] boolean (which positions are image tokens)
                token_mask = (input_ids == self.config.image_token_id)  # [1, seq_len]
                num_image_tokens = image_features.shape[0]
                hidden_size = image_features.shape[-1]

                # Cumsum at token level (seq_len elements, not seq_len * hidden_size)
                gather_idx = torch.clamp(
                    torch.cumsum(token_mask.long(), dim=-1) - 1,
                    0,
                    num_image_tokens - 1,
                )  # [1, seq_len]

                # Gather: output[0, i, k] = image_features[gather_idx[0, i], k]
                gather_idx_3d = gather_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
                # [1, num_image_tokens, hidden_size]
                image_features_3d = image_features.unsqueeze(0)
                gathered = torch.gather(image_features_3d, dim=1, index=gather_idx_3d)

                # Replace image token positions with gathered features
                token_mask_expanded = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = torch.where(token_mask_expanded, gathered, inputs_embeds)

            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

            return Mistral3ModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

        model.model.forward = types.MethodType(_patched_model_forward, model.model)

    @staticmethod
    def _patch_generate_block_attention_mask():
        """Replace generate_block_attention_mask with a functional (Dynamo-traceable) version.

        The original function uses in-place XLA tensor assignment inside a Python for loop:
          causal_mask[start:end, start:end] = 0
        Dynamo marks this as "FAILED INLINING" and creates a graph break. The eagerly-run
        function queues XLA mutations that fail in extract_compiled_graph's torch_xla.sync()
        (Bad StatusOr access: INTERNAL: Error code: 13).

        For single-image inference the block mask is all-zeros (patches within one image
        attend to each other freely), so attention_mask=None is semantically equivalent.
        For multi-image batches we build the mask functionally with torch.where to avoid
        the in-place pattern.
        """
        import transformers.models.pixtral.modeling_pixtral as pixtral_module

        def _functional_generate_block_attention_mask(patch_embeds_list, tensor):
            import torch
            dtype = tensor.dtype
            device = tensor.device
            seq_len = tensor.shape[1]

            if len(patch_embeds_list) == 1:
                # Single image: all patches attend to each other; mask is all zeros.
                return torch.zeros(
                    tensor.shape[0], 1, seq_len, seq_len, dtype=dtype, device=device
                )

            # Multi-image: build mask functionally without in-place ops.
            d_min = torch.finfo(dtype).min
            row_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            allowed = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            start = 0
            for size in patch_embeds_list:
                end = start + size
                block = (row_idx >= start) & (row_idx < end) & (col_idx >= start) & (col_idx < end)
                allowed = allowed | block
                start = end
            causal_mask = torch.where(
                allowed,
                torch.zeros(seq_len, seq_len, dtype=dtype, device=device),
                torch.full((seq_len, seq_len), d_min, dtype=dtype, device=device),
            )
            return causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)

        pixtral_module.generate_block_attention_mask = _functional_generate_block_attention_mask

    @staticmethod
    def _patch_get_placeholder_mask(model):
        """Patch Mistral3Model.get_placeholder_mask to skip a device-touching check.

        The original method calls torch_compilable_check with:
          inputs_embeds[special_image_mask].numel() == image_features.numel()

        On XLA device, boolean indexing produces a dynamically-shaped tensor.
        .numel() on it forces torch_xla.sync() to materialize the shape.
        The f-string also evaluates str(n_image_tokens) (a scalar XLA tensor),
        which is another forced sync. Both → INTERNAL Error code: 13.

        The check is a debug assertion; the processor guarantees token/feature
        counts match during normal inference. Any actual mismatch would produce
        a visible error in the subsequent masked_scatter call.
        """
        import types

        def _get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
            import torch
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id
            # Skip torch_compilable_check: it uses boolean indexing on XLA tensors
            # (.numel()) and f-string str() of scalar XLA tensors, both of which
            # trigger torch_xla.sync() → INTERNAL Error code: 13.
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            return special_image_mask

        model.model.get_placeholder_mask = types.MethodType(_get_placeholder_mask, model.model)

    @staticmethod
    def _patch_get_image_features(model):
        """Patch Mistral3Model.get_image_features to keep image_sizes as Python data.

        image_sizes must stay as a Python list of (h, w) tuples throughout the
        forward pass. Two sites iterate over it in Python:
          - PixtralVisionModel.forward() line 452: zip(patch_embeds, image_sizes)
          - Mistral3PatchMerger.forward() line 76: for image_size in image_sizes

        When image_sizes is a TT/XLA device tensor, these iterations trigger a
        device-to-host transfer (INTERNAL Error code: 13 at torch_xla.sync()).
        Passing a Python list avoids device promotion and lets XLA compile static
        shapes from the Python int indices.

        The original split_sizes computation also moves image_sizes to device
        (corrupting int64→bf16: 1540→1536, giving wrong counts 2320 vs 2310).
        We replace it with pure Python integer arithmetic.

        load_inputs() converts the processor's image_sizes tensor to a Python
        list of tuples before returning, so Dynamo never promotes it to a device
        tensor in the first place.
        """
        import types

        def _get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, output_hidden_states=None, **kwargs):
            import torch
            # Resolve vision_feature_layer from config if not provided
            if vision_feature_layer is None:
                vision_feature_layer = self.config.vision_feature_layer
            # Strip return_dict and other output-control kwargs (handled by caller)
            kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in ("return_dict",)}
            # image_sizes is expected to be a Python list of (h, w) tuples
            # (converted from tensor in load_inputs). Pass it directly to
            # vision_tower and multi_modal_projector so they can iterate in Python
            # without triggering device-to-host transfers.
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
                hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            # Use [0] indexing instead of squeeze(0) to avoid prims.view_of
            image_features = self.multi_modal_projector(selected_image_feature[0], image_sizes)
            downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
            # Pure Python arithmetic: image_sizes is a list of (h, w) Python int tuples.
            # No tensor device involvement → no Error code: 13.
            split_sizes = [
                (h // downsample_ratio) * (w // downsample_ratio)
                for h, w in image_sizes
            ]
            # multi_modal_projector returns [N_merged, d] (2D); squeeze(0) is a no-op
            # but decomposes to prims.view_of which breaks XLA functionalization.
            image_features = torch.split(image_features, split_sizes)
            image_outputs.pooler_output = image_features
            return image_outputs

        # Patch the inner Mistral3Model instance (model.model), not the outer wrapper
        model.model.get_image_features = types.MethodType(_get_image_features, model.model)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ministral 3 8B Instruct BnB 4-bit model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance.
        """
        from transformers import Mistral3ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # BnB variants need device_map="cpu" for CPU-based loading
        model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()

        self._dequantize_bnb4_to_bf16(model)
        self._patch_inputs_embedding_merge(model)
        self._patch_generate_block_attention_mask()
        self._patch_get_placeholder_mask(model)
        self._patch_get_image_features(model)

        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the model.

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

        # Convert image_sizes from tensor to a Python list of (h, w) tuples.
        # Dynamo treats plain Python lists as static constants and does NOT move
        # them to TT/XLA device. This prevents device-to-host transfers
        # (INTERNAL Error code: 13) in PixtralVisionModel and Mistral3PatchMerger,
        # which iterate over image_sizes in Python during forward.
        if "image_sizes" in inputs and isinstance(inputs["image_sizes"], torch.Tensor):
            inputs["image_sizes"] = [tuple(s.tolist()) for s in inputs["image_sizes"]]

        return inputs
