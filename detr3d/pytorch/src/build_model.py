# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from third_party.tt_forge_models.detr3d.pytorch.src.utils import MODELS as MMCV_MODELS
from third_party.tt_forge_models.detr3d.pytorch.src.utils import (
    Registry,
    build_from_cfg,
)

# from tests.models.detr3d.src.utils import build_from_cfg

MODELS = Registry("models", parent=MMCV_MODELS)

DETECTORS = MODELS
BACKBONES = MODELS
LOSSES = MODELS
NECKS = MODELS
HEADS = MODELS

TRANSFORMER = Registry("Transformer")
TRANSFORMER_LAYER_SEQUENCE = Registry("transformer-layers sequence")
TRANSFORMER_LAYER = Registry("transformerLayer")
ATTENTION = Registry("attention")
FEEDFORWARD_NETWORK = Registry("feed-forward Network")
CONV_LAYERS = Registry("conv layer")
POSITIONAL_ENCODING = Registry("position encoding")
BBOX_CODERS = Registry("bbox_coder")


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
