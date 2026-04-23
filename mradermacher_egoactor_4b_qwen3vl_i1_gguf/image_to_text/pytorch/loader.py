# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher EgoActor 4B Qwen3VL i1 GGUF model loader implementation for image to text.
"""

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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


class ModelVariant(StrEnum):
    """Available mradermacher EgoActor 4B Qwen3VL i1 GGUF model variants for image to text."""

    EGOACTOR_4B_QWEN3VL_I1_GGUF = "4b_qwen3vl_i1_gguf"


class ModelLoader(ForgeModel):
    """mradermacher EgoActor 4B Qwen3VL i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.EGOACTOR_4B_QWEN3VL_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/EgoActor-4b-Qwen3VL-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EGOACTOR_4B_QWEN3VL_I1_GGUF

    GGUF_REPO = "mradermacher/EgoActor-4b-Qwen3VL-i1-GGUF"
    GGUF_FILE = "EgoActor-4b-Qwen3VL.i1-Q4_K_M.gguf"
    BASE_MODEL = "BAAI-Agents/EgoActor-4b-Qwen3VL"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher EgoActor 4B Qwen3VL i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # GGUF repos do not ship config.json; download weights separately and
        # load config from the base model snapshot. Passing an absolute gguf_file
        # path with a local pretrained_model_name_or_path causes os.path.join to
        # resolve to the absolute path, bypassing the HF hub filename lookup.
        gguf_path = hf_hub_download(repo_id=self.GGUF_REPO, filename=self.GGUF_FILE)
        base_dir = snapshot_download(
            repo_id=self.BASE_MODEL,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
        )
        model_kwargs["gguf_file"] = gguf_path

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_dir, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
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
        return inputs
