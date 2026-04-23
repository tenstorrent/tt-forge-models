"""Stub for transformer_engine.pytorch.distributed."""
import torch


class CudaRNGStatesTracker:
    def __init__(self):
        self._states = {}

    def add(self, name, seed):
        pass

    def get_states(self):
        return self._states


def get_all_rng_states():
    return {}


def graph_safe_rng_available():
    return False


def checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)


def gather_along_first_dim(tensor, *args, **kwargs):
    return tensor
