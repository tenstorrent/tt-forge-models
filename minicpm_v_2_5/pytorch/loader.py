# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2_5 model loader implementation for multimodal visual question answering
"""

import importlib
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import get_file


def _patch_remote_code():
    """Fix transformers-5.x incompatibilities in MiniCPM-Llama3-V-2.5's cached remote files."""
    cache_base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
    )
    glob_prefix = "openbmb/MiniCPM_hyphen_Llama3_hyphen_V_hyphen_2_5/*"

    # Fix 1: resampler.py uses List[Tensor] but only imports Optional, Tuple (Python 3.12 NameError)
    for path in cache_base.glob(f"{glob_prefix}/resampler.py"):
        text = path.read_text()
        old = "from typing import Optional, Tuple"
        new = "from typing import List, Optional, Tuple"
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 2: MiniCPMV.__init__ never calls self.post_init(), so all_tied_weights_keys
    # (added in transformers 5.x) is never initialized, causing AttributeError in
    # _adjust_tied_keys_with_tied_pointers during from_pretrained.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old = "        self.transform = self.init_transform()\n"
        new = "        self.transform = self.init_transform()\n        self.post_init()\n"
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 3: MiniCPMVBatchFeature.to() uses cast_tensor() which calls
    # torch.is_floating_point(v) unconditionally. When tgt_sizes contains Python
    # int leaves (we convert them to avoid XLA dynamic-shape alignment padding),
    # cast_tensor receives bare ints and torch.is_floating_point(int) raises TypeError.
    for path in cache_base.glob(f"{glob_prefix}/image_processing_minicpmv.py"):
        text = path.read_text()
        old = "        def cast_tensor(v):\n            # check if v is a floating point\n            if torch.is_floating_point(v):"
        new = "        def cast_tensor(v):\n            if not isinstance(v, torch.Tensor):\n                return v\n            # check if v is a floating point\n            if torch.is_floating_point(v):"
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 4: torch.vstack() in PyTorch 2.7 requires Tensor elements; it rejects Python
    # lists. tgt_sizes is now a flat Python list of [h, w] pairs so torch.vstack fails.
    # Use torch.tensor() for Python list input, which creates a CPU int32 tensor whose
    # concrete values keep max_patch_len static so XLA never sees a dynamic bool dimension.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old = "                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)\n"
        new = (
            "                if isinstance(tgt_sizes[0], torch.Tensor):\n"
            "                    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)\n"
            "                else:\n"
            "                    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int32)\n"
        )
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 5: torch.max() on XLA int32/float16 tensors is computed in BF16 internally,
    # which rounds 1036→1040 (bf16 rounds 1036 to nearest representable = 1040 for this
    # value). This corrupts max_patch_len and max_h/max_w. Element-access [0] reads raw
    # bits and returns the correct value (1036). For the non-batch path
    # (batch_vision_input=False), bs=1 always, so len==1 is safe.
    for path in cache_base.glob(f"{glob_prefix}/resampler.py"):
        text = path.read_text()
        old5a = (
            "        max_h = torch.max(tgt_sizes[:, 0])\n"
            "        max_w = torch.max(tgt_sizes[:, 1])\n"
        )
        new5a = (
            "        max_h = int(tgt_sizes[:, 0][0]) if len(tgt_sizes) == 1"
            " else int(torch.max(tgt_sizes[:, 0]))\n"
            "        max_w = int(tgt_sizes[:, 1][0]) if len(tgt_sizes) == 1"
            " else int(torch.max(tgt_sizes[:, 1]))\n"
        )
        if old5a in text and new5a not in text:
            text = text.replace(old5a, new5a, 1)
        new5b = (
            "        max_patch_len = int(patch_len[0]) if len(patch_len) == 1"
            " else int(torch.max(patch_len))\n"
        )
        for old5b in [
            "        max_patch_len = torch.max(patch_len)\n",
            "        max_patch_len = int(torch.max(patch_len))\n",
        ]:
            if old5b in text and new5b not in text:
                text = text.replace(old5b, new5b, 1)
                break
        path.write_text(text)

    # Invalidate the module cache so patched files are re-imported
    for key in list(sys.modules):
        if "MiniCPM_hyphen_Llama3" in key or "minicpm" in key.lower():
            del sys.modules[key]
    importlib.invalidate_caches()


class ModelVariant(StrEnum):
    """Available MiniCPM-Llama3-V-2_5 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MiniCPM-Llama3-V-2_5 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-Llama3-V-2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-Llama3-V-2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2_5 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        _patch_remote_code()

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        # Disable batch vision input: in the batch path, patch_attn_mask (shape [B,1,N])
        # and pixel_values get padded inconsistently by XLA, causing a shape mismatch in
        # idefics2's position embedding. The per-image path derives both tensors from the
        # same pixel_values shape, avoiding the divergence.
        self.model.config.batch_vision_input = False

        if dtype_override:
            self.model = self.model.to(dtype_override)

        self.model.eval()
        return self.model

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for the MiniCPM-Llama3-V-2_5 forward pass.

        Returns:
            dict: {"data": data_dict} where data_dict contains input_ids, position_ids,
                  pixel_values, tgt_sizes, and image_bound as expected by forward().
        """
        processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "What is in the image?"
        msgs = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt, [image], return_tensors="pt", max_length=2048)

        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Flatten tgt_sizes to a list of [h, w] int pairs (Python lists, not tensors).
        # The processor returns [tensor_of_shape(N,2)] — a list with one 2D tensor.
        # Using a flat Python list makes torch.vstack create a CPU tensor, so
        # max_patch_len is a concrete Python int and avoids XLA BF16-rounding of max().
        flat_tgt = []
        for ts in inputs["tgt_sizes"]:
            if isinstance(ts, torch.Tensor):
                flat_tgt.extend(ts.tolist())
            else:
                flat_tgt.extend(ts)
        inputs["tgt_sizes"] = flat_tgt

        return {"data": inputs}

    def decode_output(self, outputs, **kwargs):
        """Decode model outputs into human-readable text."""
        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
