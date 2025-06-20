# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration classes for ForgeModel implementations
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class StrEnum(Enum):
    """Enum with string representation matching its value"""

    def __str__(self) -> str:
        return self.value


class ModelGroup(StrEnum):
    """Model groups for categorization and reporting"""

    GENERALITY = "generality"
    RED = "red"
    PRIORITY = "priority"


class ModelTask(StrEnum):
    """
    Classification of tasks models can perform.

    Based on HuggingFace task classifications.
    """

    NLP_TEXT_CLS = "nlp_text_cls"  # Text classification
    NLP_TOKEN_CLS = "nlp_token_cls"  # Token classification
    NLP_QA = "nlp_qa"  # Question answering
    NLP_CAUSAL_LM = "nlp_causal_lm"  # Causal language modeling
    NLP_MASKED_LM = "nlp_masked_lm"  # Masked language modeling
    NLP_TRANSLATION = "nlp_translation"  # Translation
    NLP_SUMMARIZATION = "nlp_summarization"  # Summarization
    NLP_MULTI_CHOICE = "nlp_multi_choice"  # Multiple choice

    # Audio tasks
    AUDIO_CLS = "audio_cls"  # Audio classification
    AUDIO_ASR = "audio_asr"  # Automatic speech recognition

    # Computer Vision tasks
    CV_IMAGE_CLS = "cv_image_cls"  # Image classification
    CV_IMAGE_SEG = "cv_image_seg"  # Image segmentation
    CV_VIDEO_CLS = "cv_video_cls"  # Video classification
    CV_OBJECT_DET = "cv_object_det"  # Object detection
    CV_IMAGE_FE = "cv_image_fe"  # Image feature extraction

    # Multimodal tasks
    MM_IMAGE_CAPT = "mm_image_capt"  # Image captioning
    MM_VISUAL_QA = "mm_visual_qa"  # Visual question answering


class ModelSource(StrEnum):
    """Where the model was sourced from"""

    HUGGING_FACE = "huggingface"
    TORCH_HUB = "torch_hub"
    CUSTOM = "custom"


class Framework(StrEnum):
    """Framework the model is implemented in"""

    JAX = "jax"
    TORCH = "pytorch"
    NUMPY = "numpy"


class Parallelism(StrEnum):
    """Multi-device parallelism strategy the model is using."""

    SINGLE_DEVICE = "single_device"
    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"


@dataclass(frozen=True)
class ModelInfo:
    """
    Dashboard/reporting metadata about a model.
    Used for categorization and metrics tracking.
    """

    model: str
    variant: str
    group: ModelGroup
    task: ModelTask
    source: ModelSource
    framework: Framework

    @property
    def name(self) -> str:
        """Generate a standardized model identifier"""
        return f"{self.framework}_{self.model}_{self.variant}_{self.task}_{self.source}"

    def to_report_dict(self) -> dict:
        """Represents self as dict suitable for pytest reporting pipeline."""
        return {
            "task": str(self.task),
            "source": str(self.source),
            "framework": str(self.framework),
            "model_arch": self.model,
            "variant_name": self.variant,
        }


@dataclass
class ModelConfig:
    """
    Base configuration for model variants.
    Contains common configuration parameters that apply across model types.
    """

    pretrained_model_name: str

    # Common configuration fields shared across models

    def __post_init__(self):
        """Validate required fields after initialization"""
        pass


@dataclass
class LLMModelConfig(ModelConfig):
    """Configuration specific to language models"""

    max_length: int
    attention_mechanism: Optional[str] = None
    sliding_window: Optional[int] = None

    # Additional LLM-specific configuration
