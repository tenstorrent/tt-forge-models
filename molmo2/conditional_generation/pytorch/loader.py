# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for multimodal conditional generation.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
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
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Molmo2 model variants for multimodal conditional generation."""

    MOLMO2_O_7B = "molmo2_o_7b"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.MOLMO2_O_7B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-O-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_O_7B

    sample_text = "Describe this image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers.processing_utils import ProcessorMixin

        # transformers 5.x ProcessorMixin.__init__ rejects kwargs that aren't
        # modality sub-processors; Molmo2Processor passes optional_attributes
        # (image_use_col_tokens etc.) to super().__init__(), triggering TypeError.
        # Pop those attrs before they hit the strict validator and set them after.
        _orig_pm_init = ProcessorMixin.__init__

        def _patched_pm_init(self_pm, *args, **kwargs):
            opt_attrs = getattr(type(self_pm), "optional_attributes", [])
            extras = {
                k: kwargs.pop(k)
                for k in list(kwargs)
                if k in opt_attrs and k not in ("chat_template", "audio_tokenizer")
            }
            _orig_pm_init(self_pm, *args, **kwargs)
            for k, v in extras.items():
                setattr(self_pm, k, v)

        ProcessorMixin.__init__ = _patched_pm_init
        try:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                use_fast=False,
            )
        finally:
            ProcessorMixin.__init__ = _orig_pm_init

        return self.processor

    @staticmethod
    def _default_rope_init(config, device=None):
        # transformers 5.x removed "default" from ROPE_INIT_FUNCTIONS; inject it
        # so Molmo2RotaryEmbedding(rope_type="default") works unchanged.
        rope_theta = getattr(config, "rope_theta", 10000.0)
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
        )
        return inv_freq, 1.0

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        ROPE_INIT_FUNCTIONS.setdefault("default", self._default_rope_init)
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        # transformers 5.x no longer sets use_cache on the top-level config by default
        model.config.use_cache = False

        # TT device lacks int64 ops; image_grids[:,:2].prod() / image_grids[:,2:].prod()
        # fall back to BF16, so e.g. 23*33=759 rounds to 760, breaking the pooled-patch
        # sanity check.  Patch build_batched_images to compute the products on CPU.
        import types

        def _patched_build_batched_images(
            self_inner,
            input_ids,
            pixel_values,
            image_token_pooling,
            image_grids,
            image_num_crops,
        ):
            raw_counts = (
                input_ids == self_inner.config.image_end_token_id
            ).sum(1)
            counts = raw_counts // 2
            N = counts.size(0)
            device = input_ids.device
            num_images = int(counts.sum().item())
            assert image_grids.size(0) == num_images
            assert image_num_crops.size(0) == num_images

            # Compute pooled-patch counts on CPU (int64) to avoid BF16 rounding
            ig_cpu = image_grids.cpu().long()
            with torch.no_grad():
                first_prod = ig_cpu[:, :2].prod(dim=1)
                second_prod = ig_cpu[:, 2:].prod(dim=1)
                num_pooled_patches_per_image = (
                    (first_prod + second_prod)
                    .to(dtype=image_num_crops.cpu().dtype)
                    .to(device)
                )

            n_crops, n_patches, pixels_per_patch = pixel_values.shape

            example_ids_for_image = (
                torch.arange(N, device=device).repeat_interleave(counts)
            )
            assert example_ids_for_image.numel() == num_images

            crops_per_example = torch.zeros(
                N, dtype=image_num_crops.dtype, device=image_num_crops.device
            )
            crops_per_example.index_add_(
                0, example_ids_for_image, image_num_crops
            )

            patches_per_image = image_num_crops * n_patches

            counts_list = counts.tolist()
            index_offset_per_example_list = []
            offset_img = 0
            for c in counts_list:
                per_img_patches = patches_per_image[offset_img : offset_img + c]
                index_offset = [0] + per_img_patches.cumsum(0).tolist()[:-1]
                index_offset_per_example_list.append(index_offset)
                offset_img += c

            num_pooled_patches_per_example = torch.zeros(
                N,
                dtype=num_pooled_patches_per_image.dtype,
                device=device,
            )
            num_pooled_patches_per_example.index_add_(
                0, example_ids_for_image, num_pooled_patches_per_image
            )

            total_crops = int(crops_per_example.sum().item())
            assert total_crops == n_crops, (
                f"Expected {total_crops} crops, but got {n_crops}"
            )

            total_num_pooled_patches = int(
                num_pooled_patches_per_example.sum().item()
            )
            assert total_num_pooled_patches == image_token_pooling.size(0), (
                f"Expected {total_num_pooled_patches} pooled patches, "
                f"but got {image_token_pooling.size(0)}"
            )

            M = int(crops_per_example.max().item())
            images = torch.full(
                (N, M, n_patches, pixels_per_patch),
                fill_value=-1,
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            )

            offset_crop = 0
            for i in range(N):
                num = int(crops_per_example[i].item())
                cur = pixel_values[offset_crop : offset_crop + num]
                images[i, :num] = cur
                offset_crop += num
            assert offset_crop == n_crops

            P = int(num_pooled_patches_per_example.max().item())
            _, dim = image_token_pooling.shape
            new_token_pooling = torch.full(
                (N, P, dim),
                fill_value=-1,
                dtype=image_token_pooling.dtype,
                device=image_token_pooling.device,
            )

            patch_offset = 0
            img_offset = 0
            for i, c in enumerate(counts_list):
                num_patches = int(num_pooled_patches_per_example[i].item())
                cur = image_token_pooling[
                    patch_offset : patch_offset + num_patches
                ].clone()
                index_offset_per_example = index_offset_per_example_list[i]
                per_img_pooled = num_pooled_patches_per_image[
                    img_offset : img_offset + c
                ]
                assert (
                    len(index_offset_per_example) == per_img_pooled.numel()
                )
                offset = 0
                for j in range(c):
                    idx_off = int(index_offset_per_example[j])
                    n = int(per_img_pooled[j].item())
                    cur_slice = cur[offset : offset + n]
                    cur[offset : offset + n] = torch.where(
                        cur_slice >= 0,
                        cur_slice + idx_off,
                        cur_slice,
                    )
                    offset += n
                new_token_pooling[i, :num_patches] = cur
                patch_offset += num_patches
                img_offset += c

            assert patch_offset == total_num_pooled_patches
            assert img_offset == num_images

            return images, new_token_pooling

        model.model.build_batched_images = types.MethodType(
            _patched_build_batched_images, model.model
        )

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
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

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
