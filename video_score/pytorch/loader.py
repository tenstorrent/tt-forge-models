# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoScore model loader for AI-generated video quality regression.

VideoScore is an Idefics2-based sequence classification model that scores
generated videos across five aspects (visual quality, temporal consistency,
dynamic degree, text-to-video alignment, factual consistency) on a 1.0-4.0
scale. The Idefics2ForSequenceClassification head is provided by the
mantis-vl package (https://github.com/TIGER-AI-Lab/Mantis).
"""
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type


REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


class ModelVariant(StrEnum):
    """Available VideoScore model variants."""

    VIDEOSCORE = "videoscore"


class ModelLoader(ForgeModel):
    """VideoScore model loader implementation for AI-generated video quality regression."""

    _VARIANTS = {
        ModelVariant.VIDEOSCORE: LLMModelConfig(
            pretrained_model_name="TIGER-Lab/VideoScore",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIDEOSCORE

    sample_text_prompt = (
        "A dog is running in the park chasing a red frisbee on a sunny afternoon."
    )
    num_frames = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoScore",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the VideoScore Idefics2ForSequenceClassification model.

        Requires the mantis-vl package:
            pip install mantis-vl
        """
        import importlib.util
        import os
        import sys

        # The local mantis/ directory in this repo (Mantis-8M time series model)
        # shadows the mantis-vl pip package. Load modeling_idefics2 directly from
        # site-packages to avoid the namespace collision.
        Idefics2ForSequenceClassification = None
        for sp in sys.path:
            modeling = os.path.join(
                sp, "mantis", "models", "idefics2", "modeling_idefics2.py"
            )
            if os.path.isfile(modeling):
                spec = importlib.util.spec_from_file_location(
                    "_mantis_vl_idefics2_modeling", modeling
                )
                _mod = importlib.util.module_from_spec(spec)
                sys.modules["_mantis_vl_idefics2_modeling"] = _mod
                spec.loader.exec_module(_mod)
                Idefics2ForSequenceClassification = (
                    _mod.Idefics2ForSequenceClassification
                )
                break
        if Idefics2ForSequenceClassification is None:
            from mantis.models.idefics2 import Idefics2ForSequenceClassification

        # mantis-vl's tie_weights() doesn't accept recompute_mapping added in transformers 5.x
        _orig_tie_weights = Idefics2ForSequenceClassification.tie_weights

        def _tie_weights_compat(self, **kwargs):
            _orig_tie_weights(self)

        Idefics2ForSequenceClassification.tie_weights = _tie_weights_compat

        # DynamicCache.get_usable_length was renamed to get_seq_length in transformers 5.x;
        # mantis-vl's modeling code still uses the old name.
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = DynamicCache.get_seq_length

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Idefics2ForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # transformers 5.x no longer provides pad_token_id via PretrainedConfig
        # __getattribute__; set it explicitly to avoid AttributeError in mantis-vl forward.
        if not hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build sample inputs: a regression prompt plus synthetic video frames."""
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        rng = np.random.default_rng(42)
        frames = [
            Image.fromarray(
                rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ).convert("RGB")
            for _ in range(self.num_frames)
        ]

        eval_prompt = REGRESSION_QUERY_PROMPT.format(
            text_prompt=self.sample_text_prompt
        )
        # Idefics2Processor in transformers 5.x no longer has a chat_template for
        # this checkpoint; build the prompt string directly using the VideoScore format.
        image_token = str(self.processor.image_token)
        eou_token = str(self.processor.end_of_utterance_token)
        text = (
            "User: "
            + image_token * len(frames)
            + "\n"
            + eval_prompt
            + eou_token
            + "\nAssistant: "
        )

        inputs = self.processor(
            images=frames,
            text=text,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
