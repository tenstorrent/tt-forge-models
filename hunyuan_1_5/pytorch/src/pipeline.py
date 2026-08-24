# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 (480p t2v distilled) pipeline: DiT and both text encoders on
TT, scheduler/VAE on CPU.

The Qwen2.5-VL encoder and the DiT are tensor-parallel sharded over one shared
mesh; ByT5 (0.22B) carries no shard spec, so SPMD replicates it. All three are
loaded, compiled and uploaded once in `setup()` and stay resident, so repeat
`generate()` calls reuse both weights and compiled graphs.

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
    HunyuanVideo15TransformerWrapper,
    QwenPromptEmbedsWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

# The double-quoted span is what routes text through text_encoder_2 (the ByT5
# glyph encoder); without it the pipeline feeds the DiT zero glyph embeds.
PROMPT = 'A girl holding a paper with words "Hello, world!"'
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
        text_encoders_on_tt: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt
        self.text_encoders_on_tt = text_encoders_on_tt


class HunyuanVideo15Pipeline:
    """DiT and both text encoders on TT; scheduler and VAE stay on CPU."""

    def __init__(self, config: HunyuanVideo15Config):
        self.config = config
        self.mesh = None  # set when sharded; shared by every TT component
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
        self.image_embed_dim = self.transformer.transformer.config.image_embed_dim
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # One mesh, shared by every TT component. SPMD has to be enabled before
        # the first device op, so this runs ahead of any .to(xla_device()).
        if self.config.shard and (
            self.config.transformer_on_tt or self.config.text_encoders_on_tt
        ):
            self._init_mesh()

        if self.config.text_encoders_on_tt:
            self.load_text_encoders_to_tt()

        if self.config.transformer_on_tt:
            self.transformer = self._place_on_tt(
                self.transformer, lambda m: shard_transformer_specs(m.transformer)
            )
            # forward, not the module, so self.transformer stays an nn.Module and
            # callers can still wrap forward (e.g. the nightly PCC check).
            self.transformer.forward = torch.compile(
                self.transformer.forward, backend="tt"
            )

    def load_models(self):
        logger.info("[load_models] text_encoder (Qwen2.5-VL, ~7.07B) ...")
        # Wrapped even on CPU so both paths call it the same way; the wrapper
        # holds the hidden-state pick and the template-prefix drop.
        self.text_encoder = QwenPromptEmbedsWrapper(
            load_text_encoder(DTYPE),
            HIDDEN_STATE_SKIP_LAYER,
            PROMPT_TEMPLATE_ENCODE_START_IDX,
        ).eval()
        logger.info("[load_models] text_encoder_2 (ByT5, ~0.22B) ...")
        self.text_encoder_2 = load_text_encoder_2(DTYPE)
        logger.info("[load_models] transformer (~8.33B) ...")
        # Same wrapper the loader hands out, so every TT component takes
        # positional tensors and returns a bare tensor.
        self.transformer = HunyuanVideo15TransformerWrapper(
            load_transformer(DTYPE)
        ).eval()
        logger.info("[load_models] vae (~1.26B) ...")
        self.vae = load_vae(DTYPE, enable_tiling=True)

    def load_text_encoders_to_tt(self):
        """Qwen sharded, ByT5 (0.22B) replicated — then compile both forwards."""
        self.text_encoder = self._place_on_tt(
            self.text_encoder, lambda m: shard_text_encoder_specs(m.encoder)
        )
        self.text_encoder_2 = self._place_on_tt(self.text_encoder_2)
        for module in (self.text_encoder, self.text_encoder_2):
            module.forward = torch.compile(module.forward, backend="tt")

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            REPO_ID, subfolder="tokenizer_2"
        )

    def _init_mesh(self):
        """Enable SPMD and build the ("batch", "model") mesh all components share."""
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
        logger.info("[setup] mesh {} over {} device(s)", self.mesh_shape, num_devices)

    def _place_on_tt(self, module, shard_spec_fn=None):
        """Move a component to the device and apply its shard spec.

        `shard_spec_fn` maps the moved module to its tensor -> partition_spec
        dict; omit it to run replicated across the mesh.
        """
        module = module.to(xm.xla_device())
        if self.mesh is not None and shard_spec_fn is not None:
            specs = shard_spec_fn(module)
            assert (
                specs
            ), f"{type(module).__name__} shard spec is empty — would run replicated"
            for tensor, spec in specs.items():
                xs.mark_sharding(tensor, self.mesh, spec)
        return module

    def _to_encoder_device(self, x):
        """Move a text-encoder input to TT when the encoders run there."""
        return x.to(xm.xla_device()) if self.config.text_encoders_on_tt else x

    def _from_encoder_device(self, x):
        """Bring a text-encoder output back to host, forcing the device sync."""
        return x.to("cpu") if self.config.text_encoders_on_tt else x

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
        # The encoder gets the full mask — attention runs over the template
        # prefix; only the copy going downstream to the DiT drops it. The
        # embeds' hidden-state pick and prefix drop live in the wrapper's
        # forward, so they stay inside the graph.
        prompt_attention_mask = text_inputs.attention_mask

        logger.info("[STAGE] text_encoder (Qwen, sharded, bf16)")
        t0 = time.perf_counter()
        prompt_embeds = self._from_encoder_device(
            self.text_encoder(
                self._to_encoder_device(text_inputs.input_ids),
                self._to_encoder_device(prompt_attention_mask),
            )
        )
        self._perf["components"]["text_encoder"] = time.perf_counter() - t0

        return (
            prompt_embeds,
            prompt_attention_mask[:, PROMPT_TEMPLATE_ENCODE_START_IDX:],
        )

    def _get_byt5_prompt_embeds(self, prompt: list):
        embeds_list, mask_list = [], []
        elapsed = 0.0
        for glyph_text in [extract_glyph_texts(p) for p in prompt]:
            if glyph_text is None:
                # No quoted text: the embeds are fabricated zeros, so stay on
                # host instead of compiling a graph to produce them.
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
                logger.info("[STAGE] text_encoder_2 (ByT5, replicated, bf16)")
                # The encoder takes a float mask; DTYPE keeps it in the dtype the
                # model itself runs in. The int mask flows downstream.
                t0 = time.perf_counter()
                embeds = self._from_encoder_device(
                    self.text_encoder_2(
                        self._to_encoder_device(txt_tokens.input_ids),
                        self._to_encoder_device(txt_tokens.attention_mask.to(DTYPE)),
                    )[0]
                )
                elapsed += time.perf_counter() - t0
                mask = txt_tokens.attention_mask
            embeds_list.append(embeds)
            mask_list.append(mask)
        self._perf["components"]["text_encoder_2"] = elapsed
        return torch.cat(embeds_list, dim=0), torch.cat(mask_list, dim=0)

    def _encode_prompt(self, prompt: str):
        prompt = [prompt]
        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(prompt)
        return (
            prompt_embeds.to(dtype=DTYPE),
            prompt_embeds_mask.to(dtype=DTYPE),
            prompt_embeds_2.to(dtype=DTYPE),
            prompt_embeds_mask_2.to(dtype=DTYPE),
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
        # Bound to self up front so the encode helpers can record into it.
        self._perf = perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
        }
        gen_start = time.perf_counter()

        def _to_tt(x):
            return x.to(xm.xla_device()) if on_tt else x

        def _to_cpu(x):
            return x.to(cpu) if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("[generate] encoding prompt ...")
        # Each encoder records its own component time (device-synced) inside
        # _encode_prompt.
        (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
        ) = self._encode_prompt(prompt)

        latent_shape = (
            1,
            self.num_channels_latents,
            (cfg.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            cfg.height // self.vae_scale_factor_spatial,
            cfg.width // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(
            latent_shape, generator=generator, device=cpu, dtype=DTYPE
        )

        b, c, f, h, w = latents.shape
        cond_latents_concat = torch.zeros(b, c, f, h, w, dtype=DTYPE, device=cpu)
        mask_concat = torch.zeros(b, 1, f, h, w, dtype=DTYPE, device=cpu)
        image_embeds = torch.zeros(
            1,
            VISION_NUM_SEMANTIC_TOKENS,
            self.image_embed_dim,
            dtype=DTYPE,
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

            # Positional, in HunyuanVideo15TransformerWrapper.forward's order.
            step_start = time.perf_counter()
            noise_pred = self.transformer(
                _to_tt(latent_model_input),
                _to_tt(timestep),
                eh_tt,
                mask_tt,
                eh2_tt,
                mask2_tt,
                img_tt,
            )
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
        return frames
