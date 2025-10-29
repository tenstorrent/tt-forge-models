from __future__ import annotations
from .config import LLMModelConfig

class ForgeModel:
    _variant_config: LLMModelConfig
    def __init__(self, variant: str | None = ...) -> None: ...
