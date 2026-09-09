# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage 2.1 (Distilled) text-to-image pipeline running on Tenstorrent.

The Qwen2.5-VL encoder, the ByT5 glyph encoder and the MMDiT transformer run on
TT in bf16; the scheduler and the VAE run on CPU in fp32. Qwen and the
transformer are tensor-parallel sharded; ByT5 (0.22B) carries no shard spec, so
SPMD replicates it across the mesh.

Every TT model is loaded, compiled and uploaded once in ``setup()`` and stays
resident: in bf16 the three together are ~13 GB of a chip's 32 GB. Repeat
``generate()`` calls reuse both the weights and the compiled graphs.

The math mirrors ``HunyuanImagePipeline.__call__`` (distilled: guider disabled →
single conditional forward per step, distilled-guidance embedding, meanflow
``timestep_r``, ByT5 glyph stream).

Consumed by the runnable example (``examples/pytorch/hunyuan_image_2_1.py``), the
image-gen benchmark and the nightly PCC test. ``self._perf`` holds per-component
times after each ``generate()``.
"""

import re
import time
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from transformers import ByT5Tokenizer, Qwen2Tokenizer

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    NUM_CHANNELS_LATENTS,
    VAE_SCALE_FACTOR,
    QwenPromptEmbedsWrapper,
)

REPO_ID = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"
PROMPT = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, "
    "standing in a painting studio, wearing a red knitted scarf and a red beret "
    "with the word 'Tencent' on it, holding a paintbrush with a focused "
    "expression as it paints an oil painting of the Mona Lisa, rendered in a "
    "photorealistic photographic style."
)
# The three TT components run bf16; the VAE stays fp32 on CPU.
TT_DTYPE = torch.bfloat16
SEED = 649151
NUM_INFERENCE_STEPS = 8
DISTILLED_GUIDANCE_SCALE = 3.5
HEIGHT = 2048
WIDTH = 2048

# Verbatim from HunyuanImagePipeline.__init__.
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1000  # Qwen2.5-VL, before the template-prefix drop
TOKENIZER_2_MAX_LENGTH = 128  # ByT5
HIDDEN_STATE_SKIP_LAYER = 2  # hidden_states[-(SKIP + 1)] == hidden_states[-3]


# Copied verbatim from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage
def extract_glyph_text(prompt: str):
    """Extract quoted text for glyph (ByT5) rendering, or None if none found."""
    text_prompt_texts = []
    text_prompt_texts.extend(re.findall(r"\'(.*?)\'", prompt))
    text_prompt_texts.extend(re.findall(r"\"(.*?)\"", prompt))
    text_prompt_texts.extend(re.findall(r"‘(.*?)’", prompt))
    text_prompt_texts.extend(re.findall(r"“(.*?)”", prompt))
    if text_prompt_texts:
        return ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
    return None


class HunyuanImage21Config:
    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        self.tokenizer_max_length = TOKENIZER_MAX_LENGTH
        self.tokenizer_2_max_length = TOKENIZER_2_MAX_LENGTH


class HunyuanImage21Pipeline:
    """Qwen + ByT5 + transformer on TT (bf16); scheduler + VAE on CPU (fp32).

    Built once with ``setup()``; ``generate()`` can be called repeatedly against
    the already-resident models.
    """

    def __init__(self, config: HunyuanImage21Config):
        self.config = config
        self._perf = {}

    def setup(self):
        # One mesh, shared by every TT component; the transformer's loader
        # supplies the shape and Qwen's reports the same for a device count.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape, mesh_names = ModelLoader(
            ModelVariant.TRANSFORMER
        ).get_mesh_config(self.num_devices)
        self.mesh = get_mesh(self.mesh_shape, mesh_names)
        logger.info(
            "[setup] mesh {} over {} device(s)", self.mesh_shape, self.num_devices
        )

        self.load_models()
        self.text_encoder = self._to_tt(
            self.text_encoder, ModelVariant.TEXT_ENCODER, lambda m: m.encoder
        )
        self.text_encoder_2 = self._to_tt(
            self.text_encoder_2, ModelVariant.TEXT_ENCODER_2
        )
        self.transformer = self._to_tt(
            self.transformer, ModelVariant.TRANSFORMER, lambda m: m
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.repo_id, subfolder="scheduler"
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self.config.repo_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            self.config.repo_id, subfolder="tokenizer_2"
        )

    def load_models(self):
        # TT-bound models: load on CPU and compile here, then setup() uploads
        # them. We wrap forward, not the module, so each attribute stays an
        # nn.Module and callers can still wrap forward (e.g. the PCC check).
        self.text_encoder = QwenPromptEmbedsWrapper(
            ModelLoader(ModelVariant.TEXT_ENCODER).load_model(dtype_override=TT_DTYPE),
            HIDDEN_STATE_SKIP_LAYER,
            PROMPT_TEMPLATE_ENCODE_START_IDX,
        )
        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=TT_DTYPE
        )
        self.transformer = ModelLoader(ModelVariant.TRANSFORMER).load_model(
            dtype_override=TT_DTYPE
        )
        for module in (self.text_encoder, self.text_encoder_2, self.transformer):
            module.forward = torch.compile(module.forward, backend="tt")

        # VAE stays on CPU (0.41B).
        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )

    def _to_tt(self, module, variant: ModelVariant, shard_from=None):
        """Move a component to the device and apply its shard spec.

        ``shard_from`` maps the moved module to the object the loader's spec is
        written against; omit it to run replicated.
        """
        module = module.to(torch_xla.device())
        if shard_from is not None:
            specs = ModelLoader(variant).load_shard_spec(shard_from(module))
            assert specs, f"{variant} shard spec is empty — would run replicated/OOM"
            for tensor, spec in specs.items():
                xs.mark_sharding(tensor, self.mesh, spec)
        return module

    def _encode_qwen(self, prompt):
        """Qwen2.5-VL text encoder — TT (bf16, sharded)."""
        logger.info("[STAGE] text_encoder (Qwen, sharded, bf16): TT")
        dev = torch_xla.device()
        drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        tokens = self.tokenizer(
            [PROMPT_TEMPLATE_ENCODE.format(prompt)],
            max_length=self.config.tokenizer_max_length + drop_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        t0 = time.perf_counter()
        # .cpu() is the sync point: it forces the graph to run and only returns
        # once the result is on host, so the timer ends there.
        prompt_embeds = (
            self.text_encoder(tokens.input_ids.to(dev), tokens.attention_mask.to(dev))
            .cpu()
            .to(torch.float32)
        )
        self._perf["components"]["text_encoder"] = time.perf_counter() - t0
        return prompt_embeds, tokens.attention_mask[:, drop_idx:]

    def _encode_byt5(self, prompt):
        """ByT5 glyph encoder — TT (bf16, replicated)."""
        logger.info("[STAGE] text_encoder_2 (ByT5, bf16): TT")
        dev = torch_xla.device()
        glyph_text = extract_glyph_text(prompt)
        if glyph_text is None:
            dim = self.text_encoder_2.config.d_model
            embeds = torch.zeros((1, self.config.tokenizer_2_max_length, dim))
            mask = torch.zeros(
                (1, self.config.tokenizer_2_max_length), dtype=torch.int64
            )
            return embeds, mask

        tokens = self.tokenizer_2(
            glyph_text,
            padding="max_length",
            max_length=self.config.tokenizer_2_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Encoder takes the float mask; the int mask flows downstream.
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask.to(TT_DTYPE)

        # out[0] is last_hidden_state, the only output, so no wrapper is needed.
        t0 = time.perf_counter()
        prompt_embeds_2 = (
            self.text_encoder_2(input_ids.to(dev), attention_mask.to(dev))[0]
            .cpu()
            .to(torch.float32)
        )
        self._perf["components"]["text_encoder_2"] = time.perf_counter() - t0
        return prompt_embeds_2, tokens.attention_mask

    def generate(
        self,
        prompt: str = PROMPT,
        distilled_guidance_scale: float = DISTILLED_GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        batch_size = 1
        dev = torch_xla.device()
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # ──────────────────── Text encoders (TT) ──────────────────────
            prompt_embeds, prompt_embeds_mask = self._encode_qwen(prompt)
            prompt_embeds_2, prompt_embeds_mask_2 = self._encode_byt5(prompt)

            # ──────────── Latents / timesteps / guidance (CPU) ────────────
            latents_h = int(self.config.height) // VAE_SCALE_FACTOR
            latents_w = int(self.config.width) // VAE_SCALE_FACTOR
            latents = randn_tensor(
                (batch_size, NUM_CHANNELS_LATENTS, latents_h, latents_w),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            self.scheduler.set_timesteps(sigmas=sigmas, device="cpu")
            timesteps = self.scheduler.timesteps
            self.scheduler.set_begin_index(0)
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * batch_size, dtype=torch.float32
                )
                * 1000.0
            )

            # ─────── Transformer denoising loop (TT, bf16, sharded) ───────
            logger.info(
                "[STAGE] transformer (sharded, bf16): start ({} steps)",
                num_inference_steps,
            )
            # The trajectory stays fp32 on host; floats cast to the TT dtype at
            # the device boundary, ints (the masks) go as-is.
            to_dev = lambda x: (
                x.to(TT_DTYPE).to(dev) if x.is_floating_point() else x.to(dev)
            )

            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)
                timestep = t.expand(batch_size).to(latents.dtype)
                # meanflow: refiner timestep = next timestep (0 on the last step).
                if i == len(timesteps) - 1:
                    timestep_r = torch.tensor([0.0])
                else:
                    timestep_r = timesteps[i + 1]
                timestep_r = timestep_r.expand(batch_size).to(latents.dtype)

                tt_inputs = [
                    to_dev(latents),
                    to_dev(timestep),
                    to_dev(timestep_r),
                    to_dev(guidance),
                    to_dev(prompt_embeds),
                    to_dev(prompt_embeds_mask),
                    to_dev(prompt_embeds_2),
                    to_dev(prompt_embeds_mask_2),
                ]
                t0 = time.perf_counter()
                # .cpu() is the sync point: it forces the graph to run and only
                # returns once the result is on host, so the timer ends there.
                noise_pred = self.transformer(*tt_inputs).cpu().to(torch.float32)
                self._perf["steps"].append(time.perf_counter() - t0)

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            logger.info("[STAGE] transformer: done")

            # ────────────────────── VAE decode (CPU) ──────────────────────
            logger.info("[STAGE] vae: CPU")
            t0 = time.perf_counter()
            latents = latents.to(torch.float32) / self.vae.vae.config.scaling_factor
            image = self.vae(latents)
            self._perf["components"]["vae"] = time.perf_counter() - t0
            logger.info("[STAGE] vae: done")

            self._perf["total"] = time.perf_counter() - t_total_start
            return image


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale ([-1,1]→[0,255]), reshape and save the pipeline output as PNG."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
