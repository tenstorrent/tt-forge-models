# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM


class SamOPTConfig(OPTConfig):
    model_type = "sam_opt"

    def __init__(self, **kwargs):
        self.sam_image_size = kwargs.pop("sam_image_size", 1024)
        self.sam_mm_projector_in = kwargs.pop("sam_mm_projector_in", 1024)
        self.sam_mm_projector_out = kwargs.pop("sam_mm_projector_out", 768)
        self.vision_tower = kwargs.pop("vision_tower", None)
        self.vision_select_layer = kwargs.pop("vision_select_layer", -1)
        self.freeze_vision_tower = kwargs.pop("freeze_vision_tower", False)
        self.use_im_start_end = kwargs.pop("use_im_start_end", True)
        self.image_token_len = kwargs.pop("image_token_len", 256)
        self.im_start_token = kwargs.pop("im_start_token", 50825)
        self.im_end_token = kwargs.pop("im_end_token", 50826)
        self.im_patch_token = kwargs.pop("im_patch_token", 50265)
        super().__init__(**kwargs)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, kernel_size=16, stride=16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
        q_size / k_size, 1.0
    )
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = input_size is not None
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim)
            )

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(x)


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SamVisionTower(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    window_size=0 if i in global_attn_indexes else window_size,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
            )
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(
            out_chans, out_chans * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.net_3 = nn.Conv2d(
            out_chans * 2, out_chans * 4, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        x = self.net_2(x)
        x = self.net_3(x)
        return x


class SamOPTForCausalLM(OPTForCausalLM):
    config_class = SamOPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.model.vision_tower = SamVisionTower(img_size=config.sam_image_size)
        self.model.mm_projector = nn.Linear(
            config.sam_mm_projector_in, config.sam_mm_projector_out
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.decoder.embed_tokens(input_ids)

        if pixel_values is not None and input_ids is not None:
            vision_out = self.model.vision_tower(pixel_values)
            B = vision_out.shape[0]
            vision_features = vision_out.permute(0, 2, 3, 1).reshape(
                B, -1, vision_out.shape[1]
            )
            vision_features = self.model.mm_projector(vision_features)

            new_embeds = inputs_embeds.clone()
            for i in range(B):
                mask = input_ids[i] == self.config.im_patch_token
                new_embeds[i, mask] = vision_features[i].to(inputs_embeds.dtype)
            inputs_embeds = new_embeds

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )


def register_sam_opt():
    AutoConfig.register("sam_opt", SamOPTConfig)
    AutoModelForCausalLM.register(SamOPTConfig, SamOPTForCausalLM)
