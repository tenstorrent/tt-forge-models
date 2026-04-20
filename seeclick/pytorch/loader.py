# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeeClick model loader implementation for GUI grounding tasks.
"""

import sys
import types
from typing import Optional

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

_STUB_NAMES = [
    "DisjunctiveConstraint",
    "BeamSearchScorer",
    "PhrasalConstraint",
    "ConstrainedBeamSearchScorer",
]
for _name in _STUB_NAMES:
    if not hasattr(transformers, _name):
        setattr(transformers, _name, type(_name, (), {}))

if "transformers.generation.utils" in sys.modules:
    _gen_utils = sys.modules["transformers.generation.utils"]
    if not hasattr(_gen_utils, "SampleOutput"):
        _gen_utils.SampleOutput = type("SampleOutput", (), {})

if "transformers_stream_generator" not in sys.modules:
    _mock = types.ModuleType("transformers_stream_generator")
    _mock.init_stream_support = lambda: None
    sys.modules["transformers_stream_generator"] = _mock

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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available SeeClick model variants."""

    SEECLICK = "SeeClick"


class ModelLoader(ForgeModel):
    """SeeClick model loader for GUI grounding tasks."""

    _VARIANTS = {
        ModelVariant.SEECLICK: ModelConfig(
            pretrained_model_name="cckevinn/SeeClick",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEECLICK

    sample_text = 'In this UI screenshot, what is the position of the element corresponding to the command "search"?'

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize SeeClick model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SeeClick",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    TOKENIZER_NAME = "Qwen/Qwen-VL-Chat"

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_NAME, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeeClick model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for SeeClick."""
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")

        query = self.tokenizer.from_list_format(
            [
                {"image": str(image_file)},
                {"text": self.sample_text},
            ]
        )
        inputs = self.tokenizer(query, return_tensors="pt")

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
