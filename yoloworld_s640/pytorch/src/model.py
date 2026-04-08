# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Clean PyTorch rewrite of YOLO-World architecture — Small-640 variant only.

All class/attribute names are kept identical to the original yoloworld/pytorch/src
implementation so that checkpoint state-dict keys load without remapping.

Key naming contracts (must match checkpoint):
  ConvModule    : self.conv (Conv2d), self.bn (BatchNorm2d), self.activate (SiLU)
  MultiModal    : self.image_model, self.text_model
  YOLOv8PAFPN  : self.reduce_layers, self.upsample_layers, self.top_down_layers,
                  self.downsample_layers, self.bottom_up_layers, self.out_layers
  HeadModule    : self.cls_preds, self.reg_preds, self.cls_contrasts
"""

import copy
import itertools
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    return max(round(x * deepen_factor), 1) if x > 1 else x


# ─────────────────────────────────────────────────────────────────────────────
# ConvModule  (Conv2d + BatchNorm2d + SiLU)
# Attribute names must match original: self.conv / self.batch_norm2d / self.activate
# ─────────────────────────────────────────────────────────────────────────────


class ConvModule(nn.Module):
    """Conv2d followed by optional BN and optional SiLU.

    Attribute naming mirrors the original checkpoint keys:
      self.conv   → ``…conv.weight``
      self.bn     → ``…bn.weight``  (checkpoint uses 'bn', not 'batch_norm2d')
      self.activate → activation (not stored in checkpoint)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple] = 1,
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = dict(type="SiLU", inplace=True),
        conv_cfg: Optional[dict] = None,  # unused, kept for API compat
    ):
        super().__init__()
        has_norm = norm_cfg is not None
        if bias == "auto":
            bias = not has_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if has_norm:
            eps = norm_cfg.get("eps", 1e-5)
            momentum = norm_cfg.get("momentum", 0.1)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self._has_norm = has_norm

        if act_cfg is not None:
            act_type = act_cfg.get("type", "SiLU")
            inplace = act_cfg.get("inplace", True)
            if act_type == "SiLU":
                self.activate = nn.SiLU(inplace=inplace)
            elif act_type == "ReLU":
                self.activate = nn.ReLU(inplace=inplace)
            elif act_type == "Hardsigmoid":
                self.activate = nn.Hardsigmoid(inplace=inplace)
            else:
                self.activate = nn.SiLU(inplace=True)
        else:
            self.activate = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self._has_norm:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

    @property
    def norm(self):
        return self.bn if self._has_norm else None

    def forward(self, x: Tensor, activate: bool = True, norm: bool = True) -> Tensor:
        x = self.conv(x)
        if self._has_norm and norm:
            x = self.bn(x)
        if self.activate is not None and activate:
            x = self.activate(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DepthwiseSeparableConvModule
# ─────────────────────────────────────────────────────────────────────────────

_BN_CFG = dict(type="BN", momentum=0.03, eps=0.001)
_SILU_CFG = dict(type="SiLU", inplace=True)


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: int = 1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = _SILU_CFG,
        **kwargs,
    ):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DarknetBottleneck
# ─────────────────────────────────────────────────────────────────────────────


class DarknetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        kernel_size: Sequence[int] = (1, 3),
        padding: Sequence[int] = (0, 1),
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return out + identity
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CSPLayerWithTwoConv
# ─────────────────────────────────────────────────────────────────────────────


class CSPLayerWithTwoConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(block(x_main[-1]) for block in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SPPFBottleneck
# ─────────────────────────────────────────────────────────────────────────────


class SPPFBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Union[int, Sequence[int]] = 5,
        use_conv_first: bool = True,
        mid_channels_scale: float = 0.5,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
            )
            conv2_in = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_in, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        if self.conv1 is not None:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8CSPDarknet backbone
# ─────────────────────────────────────────────────────────────────────────────


class YOLOv8CSPDarknet(nn.Module):
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, None, 3, True, True],  # None replaced by last_stage_out_channels
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        last_stage_out_channels: int = 1024,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Tuple = (2, 3, 4),
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        arch_cfg = copy.deepcopy(self.arch_settings[arch])
        arch_cfg[-1][1] = last_stage_out_channels
        self.arch_setting = arch_cfg
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.stem = ConvModule(
            input_channels,
            make_divisible(arch_cfg[0][0], widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.layers = ["stem"]

        for idx, setting in enumerate(arch_cfg):
            stage = self._build_stage(idx, setting)
            self.add_module(f"stage{idx + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{idx + 1}")

    def _build_stage(self, stage_idx: int, setting: list) -> list:
        in_ch, out_ch, num_blocks, add_id, use_spp = setting
        in_ch = make_divisible(in_ch, self.widen_factor)
        out_ch = make_divisible(out_ch, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = [
            ConvModule(in_ch, out_ch, 3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            CSPLayerWithTwoConv(out_ch, out_ch, num_blocks=num_blocks, add_identity=add_id,
                                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        ]
        if use_spp:
            stage.append(SPPFBottleneck(out_ch, out_ch, kernel_sizes=5, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return stage

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers[1:], start=1):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingCLIPLanguageBackbone  (used only during reparameterize)
# ─────────────────────────────────────────────────────────────────────────────


class HuggingCLIPLanguageBackbone(nn.Module):
    def __init__(self, model_name: str, frozen_modules: Sequence[str] = (), dropout: float = 0.0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_cfg = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_cfg)
        self._freeze(frozen_modules)

    def _freeze(self, frozen_modules):
        if "all" in frozen_modules:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, texts: List[List[str]]) -> Tuple[Tensor, Optional[Tensor]]:
        num_per_batch = [len(t) for t in texts]
        assert max(num_per_batch) == min(num_per_batch), "batch text counts must be equal"
        flat = list(itertools.chain(*texts))
        enc = self.tokenizer(text=flat, return_tensors=None, padding=True)
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).to(next(self.model.parameters()).device)
        attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long).to(input_ids.device)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        txt_feats = out.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        txt_feats = txt_feats.reshape(len(texts), num_per_batch[0], -1)
        return txt_feats, None


# ─────────────────────────────────────────────────────────────────────────────
# MultiModalYOLOBackbone
# ─────────────────────────────────────────────────────────────────────────────


class MultiModalYOLOBackbone(nn.Module):
    def __init__(self, image_model: nn.Module, text_model: Optional[nn.Module] = None):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model

    def forward(self, image: Tensor, text: Optional[List[List[str]]]) -> Tuple:
        img_feats = self.image_model(image)
        if text is not None and self.text_model is not None:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tuple[Tensor, None]:
        assert self.text_model is not None
        return self.text_model(text)

    def forward_image(self, image: Tensor) -> Tuple[Tensor, ...]:
        return self.image_model(image)


# ─────────────────────────────────────────────────────────────────────────────
# CSPLayer  (used in YOLOv5PAFPN top/bottom-up layers)
# ─────────────────────────────────────────────────────────────────────────────


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.final_conv = ConvModule(2 * mid, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.blocks = nn.Sequential(
            *[DarknetBottleneck(mid, mid, 1.0, add_identity=add_identity, norm_cfg=norm_cfg, act_cfg=act_cfg)
              for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        return self.final_conv(torch.cat([x_main, x_short], 1))


# ─────────────────────────────────────────────────────────────────────────────
# MaxSigmoidAttnBlock  +  MaxSigmoidCSPLayerWithTwoConv
# ─────────────────────────────────────────────────────────────────────────────


class MaxSigmoidAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_heads: int = 1,
        use_depthwise: bool = False,
        with_scale: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = _BN_CFG,
        use_einsum: bool = True,
    ):
        super().__init__()
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert out_channels % num_heads == 0 and embed_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum
        self.embed_conv = (
            ConvModule(in_channels, embed_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
            if embed_channels != in_channels
            else None
        )
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if with_scale else 1.0
        self.project_conv = conv(
            in_channels, out_channels, kernel_size, stride=1, padding=padding,
            norm_cfg=norm_cfg, act_cfg=None,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape
        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)
        if self.use_einsum:
            attn_weight = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2).reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide).reshape(batch, m, height, width, n)
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale
        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        return x.reshape(B, -1, H, W)


class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        num_heads: int = 1,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        with_scale: bool = False,
        add_identity: bool = True,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
        use_einsum: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # override final_conv to include the extra attn token
        self.final_conv = ConvModule(
            (3 + num_blocks) * self.mid_channels,
            out_channels, 1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels, self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(block(x_main[-1]) for block in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


# ─────────────────────────────────────────────────────────────────────────────
# BaseYOLONeck → YOLOv5PAFPN → YOLOv8PAFPN → YOLOWorldPAFPN
# ─────────────────────────────────────────────────────────────────────────────


class BaseYOLONeck(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[int, List[int]],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        upsample_feats_cat_first: bool = True,
        freeze_all: bool = False,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList([self.build_reduce_layer(i) for i in range(len(in_channels))])
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))
        self.out_layers = nn.ModuleList([self.build_out_layer(i) for i in range(len(in_channels))])

    @abstractmethod
    def build_reduce_layer(self, idx: int) -> nn.Module: ...
    @abstractmethod
    def build_upsample_layer(self, idx: int) -> nn.Module: ...
    @abstractmethod
    def build_top_down_layer(self, idx: int) -> nn.Module: ...
    @abstractmethod
    def build_downsample_layer(self, idx: int) -> nn.Module: ...
    @abstractmethod
    def build_bottom_up_layer(self, idx: int) -> nn.Module: ...
    @abstractmethod
    def build_out_layer(self, idx: int) -> nn.Module: ...

    def forward(self, inputs: List[Tensor]) -> Tuple:
        reduce_outs = [self.reduce_layers[i](inputs[i]) for i in range(len(self.in_channels))]
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            up = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            cat = torch.cat([up, feat_low], 1) if self.upsample_feats_cat_first else torch.cat([feat_low, up], 1)
            inner_outs.insert(0, self.top_down_layers[len(self.in_channels) - 1 - idx](cat))
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            down = self.downsample_layers[idx](outs[-1])
            outs.append(self.bottom_up_layers[idx](torch.cat([down, inner_outs[idx + 1]], 1)))
        return tuple(self.out_layers[i](outs[i]) for i in range(len(self.in_channels)))


class YOLOv5PAFPN(BaseYOLONeck):
    def __init__(self, in_channels, out_channels, deepen_factor=1.0, widen_factor=1.0,
                 num_csp_blocks=1, norm_cfg=_BN_CFG, act_cfg=_SILU_CFG, **kwargs):
        self.num_csp_blocks = num_csp_blocks
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         deepen_factor=deepen_factor, widen_factor=widen_factor,
                         norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)

    def build_reduce_layer(self, idx):
        if idx == len(self.in_channels) - 1:
            return ConvModule(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
            )
        return nn.Identity()

    def build_upsample_layer(self, *args, **kwargs):
        return nn.Upsample(scale_factor=2, mode="nearest")

    def build_top_down_layer(self, idx):
        base = CSPLayer(
            make_divisible(self.in_channels[idx - 1] * 2, self.widen_factor),
            make_divisible(self.in_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )
        if idx == 1:
            return base
        return nn.Sequential(
            base,
            ConvModule(
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                make_divisible(self.in_channels[idx - 2], self.widen_factor),
                1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
            ),
        )

    def build_downsample_layer(self, idx):
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx):
        return CSPLayer(
            make_divisible(self.in_channels[idx] * 2, self.widen_factor),
            make_divisible(self.in_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )

    def build_out_layer(self, *args, **kwargs):
        return nn.Identity()


class YOLOv8PAFPN(YOLOv5PAFPN):
    def __init__(self, in_channels, out_channels, deepen_factor=1.0, widen_factor=1.0,
                 num_csp_blocks=3, norm_cfg=_BN_CFG, act_cfg=_SILU_CFG, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         deepen_factor=deepen_factor, widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)

    def build_reduce_layer(self, idx):
        return nn.Identity()

    def build_top_down_layer(self, idx):
        return CSPLayerWithTwoConv(
            make_divisible(self.in_channels[idx - 1] + self.in_channels[idx], self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx):
        return CSPLayerWithTwoConv(
            make_divisible(self.out_channels[idx] + self.out_channels[idx + 1], self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )


class YOLOWorldPAFPN(YOLOv8PAFPN):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], int],
        guide_channels: int,
        embed_channels: List[int],
        num_heads: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
        **kwargs,
    ):
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            deepen_factor=deepen_factor, widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs,
        )

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return MaxSigmoidCSPLayerWithTwoConv(
            in_channels=make_divisible(self.in_channels[idx - 1] + self.in_channels[idx], self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx - 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx - 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return MaxSigmoidCSPLayerWithTwoConv(
            in_channels=make_divisible(self.out_channels[idx] + self.out_channels[idx + 1], self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx + 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx + 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, img_feats: List[Tensor], txt_feats: Optional[Tensor] = None) -> Tuple:
        reduce_outs = [self.reduce_layers[i](img_feats[i]) for i in range(len(self.in_channels))]
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            up = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            cat = torch.cat([up, feat_low], 1) if self.upsample_feats_cat_first else torch.cat([feat_low, up], 1)
            inner_outs.insert(0, self.top_down_layers[len(self.in_channels) - 1 - idx](cat, txt_feats))
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            down = self.downsample_layers[idx](outs[-1])
            outs.append(self.bottom_up_layers[idx](torch.cat([down, inner_outs[idx + 1]], 1), txt_feats))
        return tuple(self.out_layers[i](outs[i]) for i in range(len(self.in_channels)))


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive heads
# ─────────────────────────────────────────────────────────────────────────────


class BNContrastiveHead(nn.Module):
    def __init__(self, embed_dims: int, norm_cfg: dict, use_einsum: bool = True):
        super().__init__()
        eps = norm_cfg.get("eps", 1e-5)
        momentum = norm_cfg.get("momentum", 0.1)
        # Must be BatchNorm2d — matches build_norm_layer(BN, embed_dims)[1]
        # stored as self.norm (key: 'cls_contrasts.N.norm.*')
        self.norm = nn.BatchNorm2d(embed_dims, eps=eps, momentum=momentum)
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        # x: [B, embed_dims, H, W]; w: [B, num_classes, embed_dims]
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        if self.use_einsum:
            x = torch.einsum("bchw,bkc->bkhw", x, w)
        else:
            B, C, H, W = x.shape
            _, k, c = w.shape
            x_p = x.permute(0, 2, 3, 1).reshape(B, -1, c)
            x = torch.matmul(x_p, w.permute(0, 2, 1)).reshape(B, H, W, k).permute(0, 3, 1, 2)
        return x * self.logit_scale.exp() + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# YOLOWorldHeadModule  (inference forward only)
# ─────────────────────────────────────────────────────────────────────────────


class YOLOWorldHeadModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: Union[int, Sequence],
        embed_dims: int,
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        featmap_strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        use_bn_head: bool = False,
        use_einsum: bool = True,
        norm_cfg: Optional[dict] = _BN_CFG,
        act_cfg: Optional[dict] = _SILU_CFG,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.reg_max = reg_max
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        in_ch_list = [make_divisible(c, widen_factor) for c in in_channels]
        self.in_channels = in_ch_list

        reg_out = max(16, in_ch_list[0] // 4, reg_max * 4)
        cls_out = max(in_ch_list[0], num_classes)

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        for i in range(self.num_levels):
            self.reg_preds.append(nn.Sequential(
                ConvModule(in_ch_list[i], reg_out, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(reg_out, reg_out, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                nn.Conv2d(reg_out, 4 * reg_max, 1),
            ))
            self.cls_preds.append(nn.Sequential(
                ConvModule(in_ch_list[i], cls_out, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(cls_out, cls_out, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                nn.Conv2d(cls_out, embed_dims, 1),
            ))
            if use_bn_head:
                self.cls_contrasts.append(BNContrastiveHead(embed_dims, norm_cfg, use_einsum=use_einsum))
            else:
                raise NotImplementedError("Only use_bn_head=True is supported in this loader")

        proj = torch.arange(reg_max, dtype=torch.float)
        self.register_buffer("proj", proj, persistent=False)

    def forward(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        cls_logits, bbox_preds = [], []
        txt_feats_list = [txt_feats] * self.num_levels
        for i in range(self.num_levels):
            feat = img_feats[i]
            b, _, h, w = feat.shape
            cls_embed = self.cls_preds[i](feat)
            cls_logit = self.cls_contrasts[i](cls_embed, txt_feats_list[i])
            bbox_dist = self.reg_preds[i](feat)
            if self.reg_max > 1:
                bbox_dist = bbox_dist.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
                bbox_pred = bbox_dist.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
                bbox_pred = bbox_pred.transpose(1, 2).reshape(b, -1, h, w)
            else:
                bbox_pred = bbox_dist
            cls_logits.append(cls_logit)
            bbox_preds.append(bbox_pred)
        return cls_logits, bbox_preds


# ─────────────────────────────────────────────────────────────────────────────
# YOLOWorldDetector  (top-level, inference after reparameterize)
# ─────────────────────────────────────────────────────────────────────────────


class YOLOWDetDataPreprocessor(nn.Module):
    """Normalises a uint8 BGR image batch into float RGB in [0, 1]."""

    def forward(self, batch: dict, training: bool = False) -> dict:
        imgs = batch["inputs"]
        # imgs arrive as float (already stacked by load_inputs), scale to [0,1]
        imgs = imgs.float() / 255.0
        batch["inputs"] = imgs
        return batch


class YOLOWorldDetector(nn.Module):
    """YOLO-World detector — inference path only (after reparameterize).

    After calling ``reparameterize(texts)``, ``self.text_feats`` is a frozen
    tensor and ``forward(img)`` requires only the image.
    """

    def __init__(
        self,
        backbone: MultiModalYOLOBackbone,
        neck: YOLOWorldPAFPN,
        head_module: YOLOWorldHeadModule,
        data_preprocessor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = _YOLOWorldHead(head_module)
        self.data_preprocessor = data_preprocessor or YOLOWDetDataPreprocessor()
        self.text_feats: Optional[Tensor] = None
        self.texts: Optional[List[List[str]]] = None

    def reparameterize(self, texts: List[List[str]]) -> None:
        self.texts = texts
        # Collect all labels into a single-batch call so text_feats is [1, K, 512]
        # (forward_text with [['l1', 'l2', ...]] → reshape(-1, K, 512) → [1, K, 512])
        all_labels = [t[0] if len(t) == 1 else t[0] for t in texts]
        text_feats, _ = self.backbone.forward_text([all_labels])
        # Cast to model dtype so neck/head linear layers don't see dtype mismatch
        model_dtype = next(self.neck.parameters()).dtype
        self.text_feats = text_feats.to(dtype=model_dtype)

    def forward(self, img: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Inference forward — must call reparameterize first."""
        assert self.text_feats is not None, "Call reparameterize(texts) before forward()"
        img_feats = self.backbone.forward_image(img)
        text_feats = self.text_feats.to(device=img.device)
        neck_feats = self.neck(img_feats, text_feats)
        return self.bbox_head(neck_feats, text_feats)


class _YOLOWorldHead(nn.Module):
    """Thin wrapper so checkpoint key 'bbox_head.head_module.*' matches."""

    def __init__(self, head_module: YOLOWorldHeadModule):
        super().__init__()
        self.head_module = head_module

    def forward(self, img_feats, txt_feats):
        return self.head_module(img_feats, txt_feats, None)


# ─────────────────────────────────────────────────────────────────────────────
# Factory for Small-640
# ─────────────────────────────────────────────────────────────────────────────

#  Small-640 hardcoded hyperparameters
_S640 = dict(
    deepen_factor=0.33,
    widen_factor=0.5,
    last_stage_out_channels=1024,
    num_training_classes=80,
    num_test_classes=1203,
    text_channels=512,
    strides=[8, 16, 32],
    # neck raw channels (widen applied internally)
    neck_in_channels=[256, 512, 1024],
    neck_out_channels=[256, 512, 1024],
    neck_embed_channels=[128, 256, 512],
    neck_num_heads=[4, 8, 16],
    norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
    act_cfg=dict(type="SiLU", inplace=True),
)


def build_yoloworld_s640(clip_model_name: str = "openai/clip-vit-base-patch32") -> YOLOWorldDetector:
    """Instantiate an untrained YOLO-World Small-640 detector."""
    p = _S640
    image_backbone = YOLOv8CSPDarknet(
        arch="P5",
        last_stage_out_channels=p["last_stage_out_channels"],
        deepen_factor=p["deepen_factor"],
        widen_factor=p["widen_factor"],
        norm_cfg=p["norm_cfg"],
        act_cfg=p["act_cfg"],
    )
    text_backbone = HuggingCLIPLanguageBackbone(
        model_name=clip_model_name,
        frozen_modules=["all"],
    )
    backbone = MultiModalYOLOBackbone(image_model=image_backbone, text_model=text_backbone)
    neck = YOLOWorldPAFPN(
        in_channels=p["neck_in_channels"],
        out_channels=p["neck_out_channels"],
        guide_channels=p["text_channels"],
        embed_channels=p["neck_embed_channels"],
        num_heads=p["neck_num_heads"],
        deepen_factor=p["deepen_factor"],
        widen_factor=p["widen_factor"],
        num_csp_blocks=3,
        norm_cfg=p["norm_cfg"],
        act_cfg=p["act_cfg"],
    )
    head_module = YOLOWorldHeadModule(
        num_classes=p["num_training_classes"],
        in_channels=p["neck_in_channels"],
        embed_dims=p["text_channels"],
        widen_factor=p["widen_factor"],
        featmap_strides=p["strides"],
        reg_max=16,
        use_bn_head=True,
        use_einsum=True,
        norm_cfg=p["norm_cfg"],
        act_cfg=p["act_cfg"],
    )
    return YOLOWorldDetector(backbone=backbone, neck=neck, head_module=head_module)
