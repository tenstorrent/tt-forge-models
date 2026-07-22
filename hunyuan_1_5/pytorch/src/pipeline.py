# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 (480p t2v distilled) pipeline: DiT tensor-parallel on TT,
text encoders/scheduler/VAE on CPU.

guidance_scale=1.0 for this checkpoint -> CFG disabled -> single transformer
forward per step (no guider object needed). use_meanflow=False -> no
timestep_r. image_embeds (t2v) is zeros(batch, 729, image_embed_dim) — 729 is
HunyuanVideo15Pipeline.vision_num_semantic_tokens.
"""

import os
import re
import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import ByT5Tokenizer, Qwen2Tokenizer

from .model_utils import (
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    load_text_encoder,
    load_text_encoder_2,
    load_transformer,
    load_vae,
    shard_transformer_specs,
)

PROMPT = "a cat sitting on a boat"
SEED = 42
HEIGHT = 480
WIDTH = 848
NUM_FRAMES = 25  # using 25 instead of 121, reason: https://github.com/tenstorrent/tt-xla/issues/5761
NUM_INFERENCE_STEPS = 10  # using 10 instead of 50, reason: https://github.com/tenstorrent/tt-xla/issues/5761
FPS = 15
DTYPE = torch.bfloat16

VISION_NUM_SEMANTIC_TOKENS = 729  # HunyuanVideo15Pipeline.vision_num_semantic_tokens
TOKENIZER_2_MAX_LENGTH = 256
PROMPT_TEMPLATE_ENCODE_START_IDX = 108
TOKENIZER_MAX_LENGTH = 1000
HIDDEN_STATE_SKIP_LAYER = 2  # hidden_states[-3]

SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:"
    "         1. The main content and theme of the video."
    "         2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "         3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "         4. background environment, light, style and atmosphere."
    "         5. camera angles, movements, and transitions used in the video."
)


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _decompose_layernorm(model):
    """Replace every nn.LayerNorm's forward with explicit mean/var/rsqrt ops in
    the model dtype (bf16).

    Empirical workaround for a TT PCC drop / NaN in the early transformer blocks
    of the first DiT inference step, traced to LayerNorm but not reproducible in
    a standalone sanity check (still under investigation). Details:
    https://github.com/tenstorrent/tt-xla/issues/5762

    Covers norm1/norm2's inner LayerNorm; NOT the QK-norm (rms_norm) in attention.
    """
    for mod in model.modules():
        if isinstance(mod, torch.nn.LayerNorm):

            def _fwd(x, m=mod):
                mu = x.mean(-1, keepdim=True)
                var = (x - mu).pow(2).mean(-1, keepdim=True)
                y = (x - mu) * torch.rsqrt(var + m.eps)
                if m.weight is not None:
                    y = y * m.weight
                if m.bias is not None:
                    y = y + m.bias
                return y

            mod.forward = _fwd


def format_text_input(prompt: list, system_message: str) -> list:
    return [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p if p else " "},
        ]
        for p in prompt
    ]


def extract_glyph_texts(prompt: str):
    """Matches double/Chinese-full-width quotes only (not single quotes)."""
    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [m[0] or m[1] for m in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if result:
        return ". ".join([f'Text "{t}"' for t in result]) + ". "
    return None


def save_video(frames, filepath: str = "output.mp4", fps: int = FPS):
    """Save generate()'s frames (PIL images) as an MP4 — used by the demo."""
    export_to_video(frames, filepath, fps=fps)


class HunyuanVideo15Config:
    def __init__(
        self,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_frames: int = NUM_FRAMES,
        shard: bool = True,
        transformer_on_tt: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt


class HunyuanVideo15Pipeline:
    """DiT sharded on TT; text encoders, scheduler, VAE stay on CPU."""

    def __init__(self, config: HunyuanVideo15Config):
        self.config = config
        self.mesh_shape = None  # set when sharded; read by the benchmark harness
        self._perf = None  # per-stage/per-step timings from the last generate()

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio
        self.vae_scale_factor_spatial = self.vae.config.spatial_compression_ratio
        self.num_channels_latents = self.vae.config.latent_channels
        self.scaling_factor = self.vae.config.scaling_factor
        self.image_embed_dim = self.transformer.config.image_embed_dim
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Decompose nn.LayerNorm into explicit ops (workaround for a TT PCC
        # drop; see _decompose_layernorm). Applied before the transformer moves
        # to device / shards.
        _decompose_layernorm(self.transformer)

        if self.config.transformer_on_tt:
            if self.config.shard:
                self.shard_to_tt()
            else:
                self.transformer = self.transformer.to(xm.xla_device())

    def load_models(self):
        logger.info("[load_models] text_encoder (Qwen2.5-VL, ~7.07B) ...")
        self.text_encoder = load_text_encoder(DTYPE)
        logger.info("[load_models] text_encoder_2 (ByT5, ~0.22B) ...")
        self.text_encoder_2 = load_text_encoder_2(DTYPE)
        logger.info("[load_models] transformer (~8.33B) ...")
        self.transformer = load_transformer(DTYPE)
        logger.info("[load_models] vae (~1.26B) ...")
        self.vae = load_vae(DTYPE, enable_tiling=True)

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            REPO_ID, subfolder="tokenizer_2"
        )

    def shard_to_tt(self):
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. Expected one of {sorted(MESH_SHAPES)}."
            )
        self.mesh = Mesh(
            np.array(range(num_devices)), MESH_SHAPES[num_devices], MESH_NAMES
        )
        self.mesh_shape = tuple(self.mesh.mesh_shape)
        self.transformer = self.transformer.to(xm.xla_device())
        for tensor, spec in shard_transformer_specs(self.transformer).items():
            xs.mark_sharding(tensor, self.mesh, spec)

    def _get_mllm_prompt_embeds(self, prompt: list):
        text_inputs = self.tokenizer.apply_chat_template(
            format_text_input(prompt, SYSTEM_MESSAGE),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_ENCODE_START_IDX,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
        prompt_attention_mask = text_inputs.attention_mask
        prompt_embeds = prompt_embeds[:, PROMPT_TEMPLATE_ENCODE_START_IDX:]
        prompt_attention_mask = prompt_attention_mask[
            :, PROMPT_TEMPLATE_ENCODE_START_IDX:
        ]
        return prompt_embeds, prompt_attention_mask

    def _get_byt5_prompt_embeds(self, prompt: list):
        embeds_list, mask_list = [], []
        for glyph_text in [extract_glyph_texts(p) for p in prompt]:
            if glyph_text is None:
                embeds = torch.zeros(
                    (1, TOKENIZER_2_MAX_LENGTH, self.text_encoder_2.config.d_model),
                    dtype=self.text_encoder_2.dtype,
                )
                mask = torch.zeros((1, TOKENIZER_2_MAX_LENGTH), dtype=torch.int64)
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=TOKENIZER_2_MAX_LENGTH,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                mask = txt_tokens.attention_mask
            embeds_list.append(embeds)
            mask_list.append(mask)
        return torch.cat(embeds_list, dim=0), torch.cat(mask_list, dim=0)

    def _encode_prompt(self, prompt: str):
        prompt = [prompt]
        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(prompt)
        dtype = self.transformer.dtype
        return (
            prompt_embeds.to(dtype=dtype),
            prompt_embeds_mask.to(dtype=dtype),
            prompt_embeds_2.to(dtype=dtype),
            prompt_embeds_mask_2.to(dtype=dtype),
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        seed: Optional[int] = SEED,
        num_inference_steps: Optional[int] = None,
        output_type: str = "pil",
    ):
        cfg = self.config
        steps = num_inference_steps or cfg.num_inference_steps
        cpu = torch.device("cpu")
        on_tt = cfg.transformer_on_tt

        # Per-stage/per-step timings for the benchmark harness (components =
        # CPU stages, steps = per-DiT-forward device latency, total = wall time).
        perf = {"components": {}, "steps": [], "step_metric_name": "transformer_step"}
        gen_start = time.perf_counter()

        def _to_tt(x):
            return x.to(xm.xla_device()) if on_tt else x

        def _to_cpu(x):
            return x.to(cpu) if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("[generate] encoding prompt ...")
        t0 = time.perf_counter()
        (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
        ) = self._encode_prompt(prompt)
        perf["components"]["text_encode"] = time.perf_counter() - t0

        latent_shape = (
            1,
            self.num_channels_latents,
            (cfg.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            cfg.height // self.vae_scale_factor_spatial,
            cfg.width // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(
            latent_shape, generator=generator, device=cpu, dtype=self.transformer.dtype
        )

        b, c, f, h, w = latents.shape
        cond_latents_concat = torch.zeros(
            b, c, f, h, w, dtype=self.transformer.dtype, device=cpu
        )
        mask_concat = torch.zeros(
            b, 1, f, h, w, dtype=self.transformer.dtype, device=cpu
        )
        image_embeds = torch.zeros(
            1,
            VISION_NUM_SEMANTIC_TOKENS,
            self.image_embed_dim,
            dtype=self.transformer.dtype,
            device=cpu,
        )

        sigmas = np.linspace(1.0, 0.0, steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=cpu)
        timesteps = self.scheduler.timesteps

        # Loop-invariant DiT inputs: move to TT once, not per step.
        eh_tt = _to_tt(prompt_embeds)
        mask_tt = _to_tt(prompt_embeds_mask)
        eh2_tt = _to_tt(prompt_embeds_2)
        mask2_tt = _to_tt(prompt_embeds_mask_2)
        img_tt = _to_tt(image_embeds)

        logger.info("[generate] DiT denoising loop: {} steps", len(timesteps))
        for i, t in enumerate(timesteps):
            logger.info("[generate] step {}/{}", i + 1, len(timesteps))
            latent_model_input = torch.cat(
                [latents, cond_latents_concat, mask_concat], dim=1
            )
            timestep = t.expand(latent_model_input.shape[0]).to(
                latent_model_input.dtype
            )

            step_start = time.perf_counter()
            noise_pred = self.transformer(
                hidden_states=_to_tt(latent_model_input),
                timestep=_to_tt(timestep),
                encoder_hidden_states=eh_tt,
                encoder_attention_mask=mask_tt,
                encoder_hidden_states_2=eh2_tt,
                encoder_attention_mask_2=mask2_tt,
                image_embeds=img_tt,
                return_dict=False,
            )[0]
            noise_pred = _to_cpu(
                noise_pred
            )  # forces the device sync -> real per-step latency
            perf["steps"].append(time.perf_counter() - step_start)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        logger.info("[generate] VAE decode ...")
        t0 = time.perf_counter()
        latents = latents.to(self.vae.dtype) / self.scaling_factor
        video = self.vae.decode(latents, return_dict=False)[0]
        frames = self.video_processor.postprocess_video(video, output_type=output_type)[
            0
        ]
        perf["components"]["vae"] = time.perf_counter() - t0

        perf["total"] = time.perf_counter() - gen_start
        self._perf = perf
        return frames
