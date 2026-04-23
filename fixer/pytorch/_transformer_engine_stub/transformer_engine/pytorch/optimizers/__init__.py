"""Stub for transformer_engine.pytorch.optimizers."""
import torch.optim as optim


class FusedAdam(optim.Adam):
    def __init__(self, *args, **kwargs):
        kwargs.pop('set_grad_none', None)
        super().__init__(*args, **kwargs)


class FusedSGD(optim.SGD):
    pass


def multi_tensor_applier(*args, **kwargs):
    pass


def multi_tensor_scale(*args, **kwargs):
    pass
