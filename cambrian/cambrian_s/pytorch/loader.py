# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cambrian-S model loader implementation for multimodal visual question answering.
"""

import os
import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file, cast_input_to_type

# Add bundled cambrian-s model code to sys.path so AutoConfig/AutoModel can register
# the cambrian_qwen architecture. The bundled copy patches one transformers-5.x
# incompatibility: `config.rope_scaling = None` in CambrianQwenForCausalLM.__init__
# was clearing rope_parameters (a property alias in 5.x), causing Qwen2RotaryEmbedding
# to fail. That line has been removed from the bundled copy.
_CAMBRIAN_PKG_DIR = os.path.dirname(__file__)
if _CAMBRIAN_PKG_DIR not in sys.path:
    sys.path.insert(0, _CAMBRIAN_PKG_DIR)

from cambrian.model.language_model.cambrian_qwen2 import (  # noqa: E402
    CambrianQwenConfig,
    CambrianQwenForCausalLM,
)
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402

AutoConfig.register("cambrian_qwen", CambrianQwenConfig, exist_ok=True)
AutoModelForCausalLM.register(CambrianQwenConfig, CambrianQwenForCausalLM, exist_ok=True)


def _tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """Tokenize a prompt that contains <image> placeholders.

    Splits on <image>, tokenizes each chunk, then inserts image_token_index
    between them (matching the cambrian mm_utils.tokenizer_image_token logic).
    """
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    input_ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])

    for i, chunk in enumerate(chunks):
        input_ids.extend(chunk[offset:])
        offset = 0
        if i < len(chunks) - 1:
            input_ids.append(image_token_index)

    return torch.tensor(input_ids, dtype=torch.long)


class ModelVariant(StrEnum):
    """Available Cambrian-S model variants."""

    CAMBRIAN_S_7B_S3 = "S_7B_S3"
    CAMBRIAN_S_3B = "S_3B"


class ModelLoader(ForgeModel):
    """Cambrian-S model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.CAMBRIAN_S_7B_S3: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-7B-S3",
        ),
        ModelVariant.CAMBRIAN_S_3B: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CAMBRIAN_S_7B_S3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cambrian-S",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cambrian-S model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # SigLipVisionEmbeddings.register_buffer("position_ids", ..., persistent=False)
        # is not in the checkpoint, so from_pretrained materializes it with
        # torch.empty (uninitialized garbage) via the meta-device code path.
        # Always reinit so the values are [0, 1, ..., num_positions-1].
        for vt in model.model.vision_tower_aux_list:
            if not hasattr(vt, "vision_tower"):
                continue
            emb = vt.vision_tower.vision_model.embeddings
            if hasattr(emb, "position_ids"):
                emb.register_buffer(
                    "position_ids",
                    torch.arange(emb.num_positions).expand((1, -1)),
                    persistent=False,
                )

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for Cambrian-S.

        Cambrian uses a non-standard input format:
        - input_ids: tokenized text with IMAGE_TOKEN_INDEX (-200) at image positions
        - attention_mask: standard attention mask
        - images: raw image tensor processed by the vision tower's image processor

        This differs from standard VLMs that pass pixel_values through the processor.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Build Qwen2 chat-formatted text with <image> placeholder as a string.
        # apply_chat_template with list-type content fails (Qwen2 template is text-only).
        text_content = f"{DEFAULT_IMAGE_TOKEN}\nWhat is shown in this image?"
        conversation = [{"role": "user", "content": text_content}]
        text_prompt = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Tokenize with <image> replaced by IMAGE_TOKEN_INDEX (-200)
        input_ids = _tokenizer_image_token(text_prompt, self.tokenizer).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # Process image using the model's vision tower image processor.
        # The Cambrian model expects a [B, C, H, W] tensor where H=W=384 (SigLIP2).
        # load_inputs is called after load_model, so self.model is available.
        vision_tower = self.model.model.vision_tower_aux_list[0]
        image_processor = vision_tower.image_processor
        processed = image_processor.preprocess(image, return_tensors="pt")
        image_tensor = processed["pixel_values"][0]  # [C, H, W]

        if dtype_override is not None:
            input_ids = input_ids  # int tensor, no dtype cast
            image_tensor = cast_input_to_type(image_tensor, dtype_override)

        # image_tensor is currently [C, H, W].
        # The Cambrian forward expects `images` as a Python list of tensors, each
        # [batch, nimgs_per_sample, C, H, W].  For a single image: [[1, 1, C, H, W]].
        # prepare_inputs_labels_for_multimodal_for_generation does:
        #   images[0].flatten(0,1).unsqueeze(0) -> [1, C, H, W]  (the actual vision input)
        # image_sizes must also be provided: [(W, H)] for unpad_image (square → no-op).
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]

        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
            # Batch > 1 not supported by Cambrian inference code (assert bs == 1 inside).
            # Best-effort: stack images the same way as batch_size=1 for now.
            image_tensor_4d = image_tensor.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 1, C, H, W]
            images = [image_tensor_4d]
        else:
            images = [image_tensor.unsqueeze(0).unsqueeze(0)]  # list of [1, 1, C, H, W]

        image_sizes = [(w, h)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "image_sizes": image_sizes,
        }

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
