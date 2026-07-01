# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenGVLab InternVL3 model loader implementation.
"""

import io
import requests
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class ModelVariant(StrEnum):
    """Available OpenGVLab InternVL3 model variants."""

    OPENGVLAB_INTERNVL3_38B = "OpenGVLab-InternVL3-38B"
    OPENGVLAB_INTERNVL3_78B = "OpenGVLab-InternVL3-78B"


class ModelLoader(ForgeModel):
    """OpenGVLab InternVL3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENGVLAB_INTERNVL3_38B: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3-38B",
            max_length=256,
        ),
        ModelVariant.OPENGVLAB_INTERNVL3_78B: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3-78B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENGVLAB_INTERNVL3_38B

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

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
            model="OpenGVLab/InternVL3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenGVLab InternVL3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OpenGVLab InternVL3 model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # InternVLChatModel (trust_remote_code) targets transformers <4.49 and never calls
        # self.post_init(), so the `all_tied_weights_keys` attribute that
        # _move_missing_keys_from_meta_to_device expects is absent. That method runs *inside*
        # from_pretrained (while moving weights off the meta device), so it can't be fixed on the
        # returned instance. Temporarily wrap it for the duration of this load only, then restore
        # the original so no other model loaded in this process is affected by the patch.
        _orig_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device

        def _safe_move_missing_keys(
            model_self, missing_keys, device_map, device_mesh, hf_quantizer
        ):
            if not hasattr(model_self, "all_tied_weights_keys"):
                model_self.all_tied_weights_keys = {}
            return _orig_move_missing(
                model_self, missing_keys, device_map, device_mesh, hf_quantizer
            )

        PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move_missing_keys
        try:
            model = AutoModel.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                **model_kwargs,
            )
        finally:
            PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move_missing
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OpenGVLab InternVL3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Build pixel_values first so we know the tile count before building the prompt.
        image_size = self.config.vision_config.image_size
        transform = _build_transform(image_size)
        response = requests.get(self.sample_image_url, timeout=30)
        response.raise_for_status()
        sample_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        tiles = _dynamic_preprocess(
            sample_image, image_size=image_size, use_thumbnail=False, max_num=1
        )
        pixel_values = torch.stack([transform(t) for t in tiles])
        img_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        pixel_values = pixel_values.to(img_dtype).repeat(batch_size, 1, 1, 1)

        # Set the image context token ID on the model so forward() can locate image slots.
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id

        # Prepend the image token sequence matching the exact tile count.
        num_image_tokens = self.model.num_image_token * len(tiles)
        image_token_str = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
        question = image_token_str + "\n" + self.sample_text

        # max_length must cover all image tokens plus a budget for text tokens,
        # rounded up to a multiple of 32 for TT hardware tile alignment.
        max_length = num_image_tokens + self._variant_config.max_length
        max_length = ((max_length + 31) // 32) * 32
        if self.tokenizer.chat_template is not None:
            conversation = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = question
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        inputs["pixel_values"] = pixel_values
        inputs["image_flags"] = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""

        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.vision_model.encoder.layers:
            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc1.bias] = ("model",)
            shard_specs[layer.mlp.fc2.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.bias] = ("model",)

        shard_specs[model.mlp1[1].weight] = ("model", "batch")
        shard_specs[model.mlp1[1].bias] = ("model",)
        shard_specs[model.mlp1[3].weight] = ("batch", "model")

        for layer in model.language_model.model.layers:
            # Attention: column-parallel q/k/v (with bias), row-parallel o_proj
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            # MLP: gate/up column-parallel, down row-parallel (all bias=False)
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        return shard_specs
