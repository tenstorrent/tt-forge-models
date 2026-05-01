# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2.5 model loader implementation for multimodal visual question answering.
"""

import importlib
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
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
from ...tools.utils import get_file


def _patch_cached_remote_files():
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

    # Invalidate the module cache so patched files are re-imported
    for key in list(sys.modules):
        if "MiniCPM_hyphen_Llama3" in key or "minicpm" in key.lower():
            del sys.modules[key]
    importlib.invalidate_caches()


class ModelVariant(StrEnum):
    """Available MiniCPM-Llama3-V-2.5 model variants."""

    MINICPM_LLAMA3_V_2_5 = "MiniCPM-Llama3-V-2.5"


class ModelLoader(ForgeModel):
    """MiniCPM-Llama3-V-2.5 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.MINICPM_LLAMA3_V_2_5: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-Llama3-V-2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_LLAMA3_V_2_5

    sample_text = "What is in the image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize MiniCPM-Llama3-V-2.5 model loader."""
        super().__init__(variant)
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

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2.5 model instance."""
        _patch_cached_remote_files()
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        model.eval()

        # Disable batch vision input: in the batch path, patch_attn_mask (shape [B,1,N])
        # and pixel_values (last dim N*14) get padded inconsistently by XLA, causing a
        # shape mismatch in idefics2's position embedding. The per-image path derives
        # both tensors from the same pixel_values shape, avoiding the divergence.
        model.config.batch_vision_input = False

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MiniCPM-Llama3-V-2.5."""
        if self.tokenizer is None:
            self._load_tokenizer()

        processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        # Build prompt with image placeholder the same way chat() does
        msgs = [{"role": "user", "content": f"(<image>./</image>)\n{self.sample_text}"}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt, [image], return_tensors="pt", max_length=2048)

        # forward() requires position_ids; generate them from the sequence length
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Convert tgt_sizes from tensors to nested Python lists.
        # In the model, tgt_sizes is used to compute max_patch_len = torch.max(patches).
        # When tgt_sizes is an XLA tensor, max_patch_len becomes a dynamic XLA scalar,
        # causing torch.zeros((bs, max_patch_len), dtype=bool) to produce a bool tensor
        # whose dimension XLA pads from 1036→1040 (next multiple of 8) while the paired
        # float key tensor stays at 1036, causing a shape mismatch in multi-head attention.
        # Keeping tgt_sizes as Python int lists prevents XLA from treating max_patch_len
        # as a dynamic shape.
        inputs["tgt_sizes"] = [
            ts.tolist() if isinstance(ts, torch.Tensor) else ts
            for ts in inputs["tgt_sizes"]
        ]

        return {"data": inputs}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, str):
            return outputs

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
