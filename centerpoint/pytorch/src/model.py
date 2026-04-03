# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterPoint model — standalone reimplementation matching mmdet3d checkpoint.

Reference: https://github.com/tianweiy/CenterPoint (CVPR 2021)

RPN + CenterHead for TT compilation:
  BEV pseudo-image (B, 64, 512, 512) → RPN neck → CenterHead → raw heatmaps

Layer structure matches the mmdetection3d checkpoint exactly for direct
weight loading. The det3d RPN uses ZeroPad2d at position 0 in each block,
so checkpoint indices are shifted +1 when loading.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class RPN(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: Tuple[int, ...] = (3, 5, 5),
        ds_strides: Tuple[int, ...] = (2, 2, 2),
        ds_filters: Tuple[int, ...] = (64, 128, 256),
        us_strides: Tuple[float, ...] = (0.5, 1, 2),
        us_filters: Tuple[int, ...] = (128, 128, 128),
    ):
        super().__init__()
        in_ch_list = [in_channels, *ds_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for i, (n_layers, ds_stride, ds_filter) in enumerate(
            zip(layer_nums, ds_strides, ds_filters)
        ):
            layers: List[nn.Module] = [
                nn.ZeroPad2d(1),
                nn.Conv2d(in_ch_list[i], ds_filter, 3, stride=ds_stride, bias=False),
                nn.BatchNorm2d(ds_filter, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(n_layers):
                layers += [
                    nn.Conv2d(ds_filter, ds_filter, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ds_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
            self.blocks.append(nn.Sequential(*layers))

            us_stride_val = us_strides[i]
            us_filter = us_filters[i]
            if us_stride_val > 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(ds_filter, us_filter, int(us_stride_val), stride=int(us_stride_val), bias=False),
                    nn.BatchNorm2d(us_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))
            elif us_stride_val == 1:
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(ds_filter, us_filter, 1, stride=1, bias=False),
                    nn.BatchNorm2d(us_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))
            else:
                k = round(1 / us_stride_val)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(ds_filter, us_filter, k, stride=k, bias=False),
                    nn.BatchNorm2d(us_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU — matches mmdet3d checkpoint layout."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SepHead(nn.Module):
    """Separate head per prediction branch — matches mmdet3d's SeparateHead."""

    def __init__(self, in_channels: int, heads: Dict[str, Tuple[int, int]],
                 head_conv: int = 64, final_kernel: int = 3):
        super().__init__()
        self.head_names = list(heads.keys())
        pad = final_kernel // 2
        for name, (out_ch, num_conv) in heads.items():
            layers: List[nn.Module] = []
            cur_in = in_channels
            for _ in range(num_conv - 1):
                layers.append(ConvBNReLU(cur_in, head_conv, final_kernel, padding=pad))
                cur_in = head_conv
            layers.append(nn.Conv2d(head_conv, out_ch, final_kernel, padding=pad, bias=True))
            self.__setattr__(name, nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: self._modules[name](x) for name in self.head_names}


class CenterHead(nn.Module):

    def __init__(self, in_channels: int = 384, share_conv_channel: int = 64,
                 head_conv: int = 64, common_heads: Dict[str, Tuple[int, int]] = None,
                 tasks_num_classes: List[int] = None):
        super().__init__()
        if common_heads is None:
            common_heads = {"reg": (2, 2), "height": (1, 2), "dim": (3, 2),
                            "rot": (2, 2), "vel": (2, 2)}
        if tasks_num_classes is None:
            tasks_num_classes = [1, 2, 2, 1, 2, 2]

        self.shared_conv = ConvBNReLU(in_channels, share_conv_channel, 3, padding=1)

        self.task_heads = nn.ModuleList()
        for num_cls in tasks_num_classes:
            heads = dict(**common_heads, heatmap=(num_cls, 2))
            self.task_heads.append(SepHead(share_conv_channel, heads, head_conv=head_conv))

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x = self.shared_conv(x)
        results = []
        for task in self.task_heads:
            out = task(x)
            if "heatmap" in out:
                out["hm"] = out.pop("heatmap")
            results.append(out)
        return results


class CenterPointRPNHead(nn.Module):
    """RPN + CenterHead — the neural network portion that runs on TT."""

    def __init__(self):
        super().__init__()
        self.rpn = RPN()
        self.head = CenterHead()

    def forward(self, bev: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        return self.head(self.rpn(bev))


class PillarFeatureNet(nn.Module):
    def __init__(self, in_features: int = 5, out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.pfn = nn.Sequential(
            nn.Linear(in_features, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, pillars: torch.Tensor) -> torch.Tensor:
        B, P, N, F = pillars.shape
        feats = self.pfn(pillars.reshape(B * P * N, F))
        feats = feats.reshape(B, P, N, self.out_channels)
        return feats.max(dim=2).values


class PointPillarsScatter(nn.Module):
    def __init__(self, bev_h: int = 512, bev_w: int = 512, channels: int = 64):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, pillar_feats: torch.Tensor, pillar_coords: torch.Tensor) -> torch.Tensor:
        B, P, C = pillar_feats.shape
        bev = pillar_feats.new_zeros((B, C, self.bev_h, self.bev_w))
        y = pillar_coords[..., 0].long().clamp(0, self.bev_h - 1)
        x = pillar_coords[..., 1].long().clamp(0, self.bev_w - 1)
        flat = y * self.bev_w + x
        bev_flat = bev.view(B, C, self.bev_h * self.bev_w)
        for b in range(B):
            idx = flat[b].unsqueeze(0).expand(C, -1)
            bev_flat[b].scatter_add_(1, idx, pillar_feats[b].transpose(0, 1))
        return bev


class CenterPointFullPipeline(nn.Module):
    def __init__(self, pillar_in_features: int = 5, pillar_out_channels: int = 64,
                 bev_h: int = 512, bev_w: int = 512):
        super().__init__()
        self.pfn = PillarFeatureNet(pillar_in_features, pillar_out_channels)
        self.scatter = PointPillarsScatter(bev_h, bev_w, pillar_out_channels)
        self.rpn_head = CenterPointRPNHead()

    def forward(self, pillars: torch.Tensor, pillar_coords: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        pillar_feats = self.pfn(pillars)
        bev = self.scatter(pillar_feats, pillar_coords)
        return self.rpn_head(bev)
