# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2.5 model loader implementation for multimodal visual question answering.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
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
        self._processor = None

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

    @staticmethod
    def _patch_resampler(model_name: str) -> None:
        """Ensure resampler.py has List imported from typing (missing in all published revisions)."""
        try:
            from transformers.dynamic_module_utils import get_cached_module_file

            resampler_path = Path(get_cached_module_file(model_name, "resampler.py"))
            content = resampler_path.read_text()
            if (
                "from typing import Optional, Tuple" in content
                and "from typing import List" not in content
            ):
                resampler_path.write_text(
                    content.replace(
                        "from typing import Optional, Tuple",
                        "from typing import List, Optional, Tuple",
                    )
                )
        except Exception:
            pass

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2.5 model instance."""
        model_name = self._variant_config.pretrained_model_name
        self._patch_resampler(model_name)
        model = AutoModel.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self._processor

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MiniCPM-Llama3-V-2.5."""
        if self._processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        # Build messages in the format expected by the processor (mirrors chat() logic)
        msgs = [{"role": "user", "content": [image, self.sample_text]}]
        images = []
        for msg in msgs:
            content = msg["content"]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self._processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(prompt, images, return_tensors="pt")

        # Ensure int64 for embedding layer compatibility
        inputs["input_ids"] = inputs["input_ids"].long()

        # forward() requires position_ids (not returned by processor)
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Return as kwargs dict so forward(data=inputs) is called correctly
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
