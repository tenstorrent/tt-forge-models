# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2_5 model loader implementation for multimodal visual question answering
"""

import glob
import os
import sys
from copy import deepcopy
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


def _clear_module_cache(cache_dir, filename):
    for pyc in glob.glob(os.path.join(cache_dir, "__pycache__", f"{filename}*.pyc")):
        os.remove(pyc)


def _patch_remote_code():
    # resampler.py uses List[Tensor] but only imports Optional, Tuple from typing.
    # modeling_minicpmv.py MiniCPMV.__init__ does not call self.post_init(), so
    # transformers 5.x all_tied_weights_keys is never initialized.
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    for module_dir in glob.glob(os.path.join(cache_dir, "openbmb", "MiniCPM*", "*")):
        resampler = os.path.join(module_dir, "resampler.py")
        if os.path.exists(resampler):
            with open(resampler) as f:
                content = f.read()
            if "from typing import Optional, Tuple" in content and "List" not in content.split("from typing import")[1].split("\n")[0]:
                content = content.replace(
                    "from typing import Optional, Tuple",
                    "from typing import List, Optional, Tuple",
                )
                with open(resampler, "w") as f:
                    f.write(content)
                _clear_module_cache(module_dir, "resampler")

        modeling = os.path.join(module_dir, "modeling_minicpmv.py")
        if os.path.exists(modeling):
            with open(modeling) as f:
                content = f.read()
            _patched = False
            # Add post_init() call at the end of MiniCPMV.__init__ so that
            # transformers 5.x all_tied_weights_keys is initialised before
            # _finalize_model_loading accesses it.
            old = "        self.transform = self.init_transform()\n\n    def init_vision_module"
            new = "        self.transform = self.init_transform()\n        self.post_init()\n\n    def init_vision_module"
            if old in content and new not in content:
                content = content.replace(old, new)
                _patched = True
            # max_patches is computed from tgt_sizes (torch.max of dynamic ints).
            # Under XLA, torch.zeros with that dynamic size gets padded to the
            # next alignment boundary, while position_ids uses pixel_values dims
            # directly.  Recompute max_patches from the padded pixel_values shape
            # after pad_sequence so both are consistent.
            old_mp = (
                "                if self.config.batch_vision_input:\n"
                "                    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])\n"
                "\n"
                "                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,\n"
                "                                                                       padding_value=0.0)\n"
                "                    B, L, _ = all_pixel_values.shape\n"
                "                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)\n"
                "\n"
                "                    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)\n"
            )
            new_mp = (
                "                if self.config.batch_vision_input:\n"
                "                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,\n"
                "                                                                       padding_value=0.0)\n"
                "                    B, L, _ = all_pixel_values.shape\n"
                "                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)\n"
                "\n"
                "                    _, _, h_pv, w_pv = all_pixel_values.shape\n"
                "                    max_patches = (h_pv // self.vpm.patch_size) * (w_pv // self.vpm.patch_size)\n"
                "\n"
                "                    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)\n"
            )
            if old_mp in content and new_mp not in content:
                content = content.replace(old_mp, new_mp)
                _patched = True
            if _patched:
                with open(modeling, "w") as f:
                    f.write(content)
                _clear_module_cache(module_dir, "modeling_minicpmv")

    for key in list(sys.modules):
        mod = key.lower()
        if ("resampler" in mod or "modeling_minicpmv" in mod) and "minicpm" in mod:
            del sys.modules[key]


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
        self.processor = None

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

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2_5 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        _patch_remote_code()

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.model.eval()

        return self.model

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for the MiniCPM-Llama3-V-2_5 forward pass.

        Returns:
            dict: {"data": data_dict} where data_dict contains input_ids, position_ids,
                  pixel_values, tgt_sizes, and image_bound as expected by forward().
        """
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "What is in the image?"
        msgs = deepcopy([{"role": "user", "content": question}])
        msgs[0]["content"] = [image, msgs[0]["content"]]

        # Preprocess messages the same way chat() does: extract images and
        # replace them with (<image>./</image>) tags in the text.
        images = []
        for msg in msgs:
            content = msg["content"]
            if isinstance(content, str):
                content = [content]
            cur_parts = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_parts.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_parts.append(c)
            msg["content"] = "\n".join(cur_parts)

        prompt = self.processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.processor(prompt, images, return_tensors="pt")

        seq_len = model_inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

        data = {
            "input_ids": model_inputs["input_ids"],
            "position_ids": position_ids,
            "pixel_values": model_inputs["pixel_values"],
            "tgt_sizes": model_inputs["tgt_sizes"],
            "image_bound": model_inputs["image_bound"],
        }

        return {"data": data}

    def decode_output(self, outputs, **kwargs):
        """Decode model outputs into human-readable text."""
        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
