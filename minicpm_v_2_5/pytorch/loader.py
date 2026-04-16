# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2_5 model loader implementation for multimodal visual question answering
"""

from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import constants as hf_constants
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

    @staticmethod
    def _patch_resampler():
        """Patch upstream resampler.py to add missing 'List' typing import.

        The HuggingFace model code for MiniCPM-V-2.5 uses ``List`` in type
        annotations without importing it from ``typing``, which fails on
        Python 3.12+.  This patches the cached dynamic module copy used by
        ``transformers``.
        """
        modules_dir = Path(hf_constants.HF_HOME) / "modules" / "transformers_modules"
        for resampler in modules_dir.glob("**/MiniCPM*/**/resampler.py"):
            text = resampler.read_text()
            if (
                "from typing import Optional, Tuple" in text
                and ", List," not in text
                and "List," not in text.split("from typing import")[1].split("\n")[0]
            ):
                text = text.replace(
                    "from typing import Optional, Tuple",
                    "from typing import List, Optional, Tuple",
                )
                resampler.write_text(text)

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2_5 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        # The upstream model code uses `List` without importing it from typing.
        # Patch the cached dynamic module before loading.  On a fresh download
        # the file won't exist yet, so we attempt a load, patch on failure, and
        # retry once.
        self._patch_resampler()
        try:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                **kwargs,
            )
        except NameError:
            self._patch_resampler()
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

        # Return the LLM backbone directly.  The full multimodal forward
        # method contains vision processing with data-dependent dynamic shapes
        # that torch.compile / dynamo cannot trace.  We pre-compute vision
        # embeddings in load_inputs and feed them to the LLM.
        return self.model.llm

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the MiniCPM-Llama3-V-2_5 model.

        Pre-computes vision embeddings and returns LLM-ready kwargs so that
        the compilable LLM backbone is the only thing torch.compile traces.
        """
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "What is in the image?"
        msgs = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
        prompt = self.processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, [image], return_tensors="pt", max_length=2048)
        inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)

        # Run the vision encoder on CPU to produce merged embeddings, then
        # feed only the resulting static tensors to the LLM backbone.
        data = dict(inputs)
        with torch.no_grad():
            vllm_embedding, _ = self.model.get_vllm_embedding(data)

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return {
            "inputs_embeds": vllm_embedding,
            "position_ids": position_ids,
        }

    def decode_output(self, outputs, **kwargs):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Raw model output (string from .chat() method)

        Returns:
            str: Decoded output text
        """
        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
