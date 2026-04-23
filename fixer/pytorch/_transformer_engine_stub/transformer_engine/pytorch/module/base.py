"""Stub for transformer_engine.pytorch.module.base."""
import torch.nn as nn


class TransformerEngineBaseModule(nn.Module):
    """Stub TransformerEngineBaseModule."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_tensor_parallel_group(self, *args, **kwargs):
        pass
