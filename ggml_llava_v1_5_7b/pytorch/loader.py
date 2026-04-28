# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GGML LLaVA v1.5 7B GGUF model loader implementation for multimodal conditional generation.
"""

import re
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


def _load_mmproj_weights(model, repo_id, mmproj_filename, dtype_override=None):
    """Load vision encoder and projector weights from a LLaVA mmproj GGUF file.

    The main GGUF for mys/ggml_llava-v1.5-7b contains only the LLaMA text
    backbone.  The CLIP vision encoder and multimodal projector live in a
    separate mmproj GGUF file.  This function reads that file and populates
    the corresponding sub-modules of the already-constructed HF model.

    GGUFReader.Tensor.data is already reshaped as shape[::-1] (dimensions
    reversed relative to the GGUF file header), so no explicit transpose or
    permute is required — the numpy arrays are already in PyTorch [out, in]
    convention.  The only naming difference to watch out for is that the
    llama.cpp clip GGUF uses 'ffn_down' for the first (expanding) MLP layer
    (HF fc1) and 'ffn_up' for the second (contracting) layer (HF fc2).
    """
    from gguf import GGUFReader
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=mmproj_filename)
    reader = GGUFReader(path)

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        # tensor.data is a numpy array already in reversed-shape (PyTorch) order.
        data = torch.from_numpy(tensor.data.copy())
        if dtype_override is not None:
            data = data.to(dtype_override)

        # ── Multimodal projector ──────────────────────────────────────────────
        if name == "mm.0.weight":
            state_dict["model.multi_modal_projector.linear_1.weight"] = data
        elif name == "mm.0.bias":
            state_dict["model.multi_modal_projector.linear_1.bias"] = data
        elif name == "mm.2.weight":
            state_dict["model.multi_modal_projector.linear_2.weight"] = data
        elif name == "mm.2.bias":
            state_dict["model.multi_modal_projector.linear_2.bias"] = data

        # ── Vision tower embeddings ───────────────────────────────────────────
        elif name == "v.class_embd":
            state_dict["model.vision_tower.vision_model.embeddings.class_embedding"] = data
        elif name == "v.patch_embd.weight":
            state_dict["model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = data
        elif name == "v.position_embd.weight":
            state_dict["model.vision_tower.vision_model.embeddings.position_embedding.weight"] = data
        elif name == "v.pre_ln.weight":
            state_dict["model.vision_tower.vision_model.pre_layrnorm.weight"] = data
        elif name == "v.pre_ln.bias":
            state_dict["model.vision_tower.vision_model.pre_layrnorm.bias"] = data
        elif name == "v.post_ln.weight":
            state_dict["model.vision_tower.vision_model.post_layernorm.weight"] = data
        elif name == "v.post_ln.bias":
            state_dict["model.vision_tower.vision_model.post_layernorm.bias"] = data

        # ── Vision encoder transformer blocks ─────────────────────────────────
        elif m := re.match(r"v\.blk\.(\d+)\.(.+)", name):
            layer_idx = int(m.group(1))
            sub = m.group(2)
            pfx = f"model.vision_tower.vision_model.encoder.layers.{layer_idx}"

            if sub == "attn_q.weight":
                state_dict[f"{pfx}.self_attn.q_proj.weight"] = data
            elif sub == "attn_q.bias":
                state_dict[f"{pfx}.self_attn.q_proj.bias"] = data
            elif sub == "attn_k.weight":
                state_dict[f"{pfx}.self_attn.k_proj.weight"] = data
            elif sub == "attn_k.bias":
                state_dict[f"{pfx}.self_attn.k_proj.bias"] = data
            elif sub == "attn_v.weight":
                state_dict[f"{pfx}.self_attn.v_proj.weight"] = data
            elif sub == "attn_v.bias":
                state_dict[f"{pfx}.self_attn.v_proj.bias"] = data
            elif sub == "attn_out.weight":
                state_dict[f"{pfx}.self_attn.out_proj.weight"] = data
            elif sub == "attn_out.bias":
                state_dict[f"{pfx}.self_attn.out_proj.bias"] = data
            elif sub == "ln1.weight":
                state_dict[f"{pfx}.layer_norm1.weight"] = data
            elif sub == "ln1.bias":
                state_dict[f"{pfx}.layer_norm1.bias"] = data
            elif sub == "ln2.weight":
                state_dict[f"{pfx}.layer_norm2.weight"] = data
            elif sub == "ln2.bias":
                state_dict[f"{pfx}.layer_norm2.bias"] = data
            # llama.cpp clip: ffn_down = expanding fc1, ffn_up = contracting fc2
            elif sub == "ffn_down.weight":
                state_dict[f"{pfx}.mlp.fc1.weight"] = data
            elif sub == "ffn_down.bias":
                state_dict[f"{pfx}.mlp.fc1.bias"] = data
            elif sub == "ffn_up.weight":
                state_dict[f"{pfx}.mlp.fc2.weight"] = data
            elif sub == "ffn_up.bias":
                state_dict[f"{pfx}.mlp.fc2.bias"] = data

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only unexpected keys are a concern; missing keys are the text-backbone
    # weights already loaded from the main GGUF.
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading mmproj: {unexpected}")


class ModelVariant(StrEnum):
    """Available GGML LLaVA v1.5 7B GGUF model variants."""

    GGML_LLAVA_V1_5_7B_Q4_K = "v1.5_7B_Q4_K"
    SECOND_STATE_LLAVA_V1_5_7B_Q4_K_M = "second_state_v1.5_7B_Q4_K_M"


class ModelLoader(ForgeModel):
    """GGML LLaVA v1.5 7B GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K: ModelConfig(
            pretrained_model_name="mys/ggml_llava-v1.5-7b",
        ),
        ModelVariant.SECOND_STATE_LLAVA_V1_5_7B_Q4_K_M: ModelConfig(
            pretrained_model_name="second-state/Llava-v1.5-7B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K

    _GGUF_FILES = {
        ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K: "ggml-model-q4_k.gguf",
        ModelVariant.SECOND_STATE_LLAVA_V1_5_7B_Q4_K_M: "llava-v1.5-7b-Q4_K_M.gguf",
    }

    # Variants whose main GGUF contains only the text backbone; vision weights
    # live in a separate mmproj GGUF file that must be loaded explicitly.
    _MMPROJ_FILES = {
        ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K: "mmproj-model-f16.gguf",
    }

    PROCESSOR_MODEL = "llava-hf/llava-1.5-7b-hf"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GGML LLaVA v1.5 7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.PROCESSOR_MODEL, use_fast=False
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGML LLaVA v1.5 7B GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        mmproj_file = self._MMPROJ_FILES.get(self._variant)
        if mmproj_file is not None:
            # The main GGUF only has the LLaMA text backbone.  Load it with
            # size-mismatch tolerance so the vision-encoder slots get
            # re-initialised; then overwrite them from the mmproj GGUF.
            model_kwargs["ignore_mismatched_sizes"] = True
            model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
            _load_mmproj_weights(
                model, pretrained_model_name, mmproj_file, dtype_override
            )
            model = model.eval()
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for GGML LLaVA v1.5 7B GGUF."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
