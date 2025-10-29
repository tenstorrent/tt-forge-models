from __future__ import annotations
from enum import Enum

class StrEnum(str, Enum):
    ...

class ModelTask(StrEnum):
    NLP_MASKED_LM: ModelTask

class ModelGroup(StrEnum):
    GENERALITY: ModelGroup

class ModelSource(StrEnum):
    HUGGING_FACE: ModelSource

class Framework(StrEnum):
    TORCH: Framework

class LLMModelConfig:
    pretrained_model_name: str
    max_length: int
    def __init__(self, *, pretrained_model_name: str, max_length: int) -> None: ...

class ModelInfo:
    model: str
    variant: str
    group: ModelGroup
    task: ModelTask
    source: ModelSource
    framework: Framework
    def __init__(self, *, model: str, variant: str, group: ModelGroup, task: ModelTask, source: ModelSource, framework: Framework) -> None: ...
