# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HiDream-I1-Full text-to-image pipeline running on Tenstorrent.

The CLIP-L and CLIP-G text encoders, the Sparse-MoE MM-DiT transformer (~17 B)
and the VAE decoder run on TT in bf16, the dtype the model ships in. Only the
transformer is tensor-parallel sharded across the device mesh; the CLIPs and the
VAE carry no shard spec, so SPMD replicates them. The T5-XXL and Llama-3.1-8B
encoders stay on CPU for now (see ``load_models``), as do the UniPC scheduler and
the host-side trajectory.

Every TT component is loaded, sharded, uploaded and compiled once in ``setup()``
and stays resident — the sharded DiT plus the replicated CLIPs and VAE are ~10 GB
of each chip's 32 GB — so ``generate()`` is pure compute and repeat calls reuse
both the weights and the compiled graphs.

The math mirrors ``HiDreamImagePipeline.__call__`` at batch_size=1, one image per
prompt: CFG doubles the DiT batch, HiDream predicts the negated flow so the sign
flip precedes guidance, and the UniPC branch drives the trajectory.

Consumed by the runnable example (``examples/pytorch/hidream_i1.py``), the
image-gen benchmark and the nightly PCC test. ``self._perf`` holds per-component
times after each ``generate()``.
"""

import math
import time
from types import SimpleNamespace
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from transformers import CLIPTokenizer, PreTrainedTokenizerFast, T5Tokenizer
from tt_torch.sparse_mlp import enable_sparse_mlp

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    HIDREAM_REPO_ID,
    LATENT_CHANNELS,
    LLAMA_REPO_ID,
    MAX_SEQ_LEN,
    VAE_SCALE,
)

REPO_ID = HIDREAM_REPO_ID
LLAMA_ID = LLAMA_REPO_ID
PROMPT = 'A cat holding a sign that says "HiDream.ai".'
# encode_prompt turns an unset negative prompt into "" and encodes it — there is
# no force_zeros_for_empty_prompt shortcut.
NEGATIVE_PROMPT = None
SEED = 0
# 10 for now, will be bumped to 50 once the rest of the components are enabled on TT.
NUM_INFERENCE_STEPS = 10
# Per the model card: Full -> guidance 5.0. > 1 enables CFG, doubling the DiT
# batch.
GUIDANCE_SCALE = 5.0
HEIGHT = 1024
WIDTH = 1024
DEFAULT_SAMPLE_SIZE = 128  # HiDreamImagePipeline.default_sample_size
# Every TT component runs bf16, the dtype the model ships in.
TT_DTYPE = torch.bfloat16

# Stand-in config for enable_sparse_mlp: create_a2a_from_deepseek_v3_moe looks up
# DeepSeek's attribute names, and HiDream's config spells it
# num_activated_experts -- without num_experts_per_tok it falls back to its
# default of 6.
MOE_CONFIG = SimpleNamespace(n_routed_experts=4, num_experts_per_tok=2)


def _strip_cpu_golden(ff) -> None:
    """Drop the pre-stack expert Linears ``enable_sparse_mlp`` keeps for its own
    golden-eval fallback. They are unsharded, so ``.to(dev)`` would replicate
    ~20 GB of unused weights onto every device; the PCC golden is a separate CPU
    twin built by the caller.
    """
    mlp = getattr(ff, "mlp", None)
    if mlp is None:
        return
    if hasattr(mlp, "_original_mlp"):
        object.__setattr__(mlp, "_original_mlp", None)
    experts = getattr(mlp, "experts", None)
    if experts is not None and "original_experts" in getattr(experts, "_modules", {}):
        del experts._modules["original_experts"]


class HiDreamI1Config:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
    ):
        self.repo_id = REPO_ID
        self.llama_id = LLAMA_ID
        self.height = height
        self.width = width
        self.max_sequence_length = MAX_SEQ_LEN
        self.default_sample_size = DEFAULT_SAMPLE_SIZE
        self.vae_scale_factor = VAE_SCALE
        # Harness-set compile options; used to preserve them around the
        # VAE-only opt_level switch in generate().
        self.compile_options = compile_options or {}


class HiDreamI1Pipeline:
    """CLIP encoders + transformer + VAE decoder on TT (bf16); T5, Llama and the
    UniPC scheduler on CPU.

    Built once with ``setup()``; ``generate()`` can be called repeatedly against
    the already-resident models.
    """

    def __init__(self, config: HiDreamI1Config):
        self.config = config
        self.repo_id = config.repo_id
        self._perf = {}

    def setup(self):
        # One mesh, shared by every TT component; the transformer's loader
        # supplies the shape, and it is the only sharded component.
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
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        """Load every component on CPU, then upload and compile the TT ones.

        T5 and Llama stay on the host — see the note below — so they are neither
        compiled nor uploaded.

        We wrap ``forward``, not the module, so each TT component stays an
        ``nn.Module`` and callers can wrap ``forward`` again (e.g. the nightly
        PCC check).

        Only the transformer (17 B) carries a shard spec; without one SPMD would
        replicate it onto every chip and OOM. CLIP-L (0.12 B), CLIP-G (0.70 B)
        and the VAE (0.08 B) are small enough to replicate.
        """
        self.text_encoder = self._to_tt(ModelVariant.TEXT_ENCODER)
        self.text_encoder_2 = self._to_tt(ModelVariant.TEXT_ENCODER_2)
        # T5-XXL and Llama stay on CPU. In bf16 on TT both fall well short of the
        # PCC gate against their own bf16 CPU twins — T5 to 0.761719 on the
        # prompt, Llama to 0.828125 on the negative one — and fp32 lifts neither
        # (0.761571 / 0.833107), so this is not a precision drop.
        # T5:    https://github.com/tenstorrent/tt-xla/issues/6018
        # Llama: https://github.com/tenstorrent/tt-xla/issues/6019
        self.text_encoder_3 = self._on_cpu(ModelVariant.TEXT_ENCODER_3)
        self.text_encoder_4 = self._on_cpu(ModelVariant.TEXT_ENCODER_4)
        self.transformer = self._to_tt(
            ModelVariant.TRANSFORMER, sharded=True, prepare=self._swap_moe
        )
        self.vae = self._to_tt(ModelVariant.VAE)

    def _swap_moe(self, transformer):
        """Swap the DiT's MoE blocks, before the device upload.

        The swap stacks the per-expert Linears into new parameters, so it has to
        run on CPU. cluster_axis=1 — the mesh is (1, N), so axis 0 would dispatch
        nowhere.
        """
        transformer = enable_sparse_mlp(
            transformer,
            mesh=self.mesh_shape,
            cluster_axis=1,
            config=MOE_CONFIG,
        )
        for module in transformer.modules():
            _strip_cpu_golden(module)
        return transformer

    def _on_cpu(self, variant: ModelVariant):
        """Load a component that stays on the host — no compile, no upload."""
        logger.info("[load] {} ({}, CPU)", variant, TT_DTYPE)
        return ModelLoader(variant).load_model(dtype_override=TT_DTYPE)

    def _to_tt(self, variant: ModelVariant, sharded: bool = False, prepare=None):
        """Load a component, compile its forward and upload it to the mesh.

        ``prepare`` runs on the CPU module before the upload. The loader's shard
        spec is written against the wrapper ``load_model`` returns, which is what
        we hold here; without ``sharded`` the module runs replicated.
        """
        logger.info(
            "[load] {} ({}, {})",
            variant,
            TT_DTYPE,
            "sharded" if sharded else "replicated",
        )
        module = ModelLoader(variant).load_model(dtype_override=TT_DTYPE)
        if prepare is not None:
            module = prepare(module)

        module.forward = torch.compile(module.forward, backend="tt")
        module = module.to(torch_xla.device())
        if sharded:
            specs = ModelLoader(variant).load_shard_spec(module)
            assert specs, f"{variant} shard spec is empty — would run replicated/OOM"
            for tensor, spec in specs.items():
                xs.mark_sharding(tensor, self.mesh, spec)
        return module

    def _add_component_time(self, name: str, seconds: float) -> None:
        """Accumulate a component's time — under CFG each encoder runs twice."""
        components = self._perf["components"]
        components[name] = components.get(name, 0.0) + seconds

    def load_scheduler(self):
        # model_index.json pins UniPCMultistepScheduler for HiDream-I1-Full.
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.repo_id, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer_2"
        )
        self.tokenizer_3 = T5Tokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer_3"
        )
        # Not in the HiDream snapshot; the pipeline expects the caller to supply it.
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.config.llama_id)
        # HiDreamImagePipeline.__init__ does this; Llama ships no pad token and
        # padding="max_length" would raise without it.
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_clip_prompt_embeds(self, tokenizer, encoder, name: str, prompt: str):
        """CLIP-L / CLIP-G pooled embedding — TT (bf16, replicated)."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        dev = torch_xla.device()

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(self.config.max_sequence_length, 218),  # 218, not 77
            truncation=True,
            return_tensors="pt",
        )
        # CLIPPooledWrapper == text_encoder(input_ids, output_hidden_states=True)[0].
        # .cpu() is the sync point: it forces the graph to run and only returns
        # once the result is on host, so the timer ends there.
        t0 = time.perf_counter()
        embeds = encoder(text_inputs.input_ids.to(dev)).cpu()
        self._add_component_time(name, time.perf_counter() - t0)
        return embeds

    def _get_t5_prompt_embeds(self, prompt: str):
        """T5-XXL encoder — CPU (bf16); see load_models for why it is not on TT."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(
                self.config.max_sequence_length, self.tokenizer_3.model_max_length
            ),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # T5EncoderWrapper == text_encoder_3(input_ids, attention_mask=...)[0].
        t0 = time.perf_counter()
        embeds = self.text_encoder_3(text_inputs.input_ids, text_inputs.attention_mask)
        self._add_component_time("text_encoder_3", time.perf_counter() - t0)
        return embeds

    def _get_llama3_prompt_embeds(self, prompt: str):
        """Llama-3.1-8B encoder — CPU (bf16); see load_models for why not on TT."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(
                self.config.max_sequence_length, self.tokenizer_4.model_max_length
            ),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # LlamaStackedHiddenWrapper == stack(hidden_states[1:], dim=0) -> (32,1,128,4096).
        t0 = time.perf_counter()
        embeds = self.text_encoder_4(text_inputs.input_ids, text_inputs.attention_mask)
        self._add_component_time("text_encoder_4", time.perf_counter() - t0)
        return embeds

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        do_classifier_free_guidance: bool = True,
    ):
        """Mirror HiDreamImagePipeline.encode_prompt at batch_size=1, 1 image/prompt."""
        logger.info(
            "[STAGE] text encoders (CLIP-L, CLIP-G replicated): TT; (T5, Llama): CPU"
        )
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""

        pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
            self.tokenizer, self.text_encoder, "text_encoder", prompt
        )
        negative_pooled_prompt_embeds_1 = (
            self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, "text_encoder", negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
            self.tokenizer_2, self.text_encoder_2, "text_encoder_2", prompt
        )
        negative_pooled_prompt_embeds_2 = (
            self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, "text_encoder_2", negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        # CLIP-L (768) ++ CLIP-G (1280) -> the DiT's 2048-d pooled conditioning.
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
        )
        negative_pooled_prompt_embeds = (
            torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2],
                dim=-1,
            )
            if do_classifier_free_guidance
            else None
        )

        prompt_embeds_t5 = self._get_t5_prompt_embeds(prompt)
        negative_prompt_embeds_t5 = (
            self._get_t5_prompt_embeds(negative_prompt)
            if do_classifier_free_guidance
            else None
        )

        prompt_embeds_llama3 = self._get_llama3_prompt_embeds(prompt)
        negative_prompt_embeds_llama3 = (
            self._get_llama3_prompt_embeds(negative_prompt)
            if do_classifier_free_guidance
            else None
        )

        return (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1
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

            # Resolution snap from __call__: rescale to the model's pixel budget,
            # then floor to a multiple of vae_scale_factor * 2. Identity at 1024.
            height, width = self.config.height, self.config.width
            division = self.config.vae_scale_factor * 2
            s_max = (
                self.config.default_sample_size * self.config.vae_scale_factor
            ) ** 2
            scale = math.sqrt(s_max / (width * height))
            width = int(width * scale // division * division)
            height = int(height * scale // division * division)

            # ──────────────────── Text encoders (TT) ──────────────────────
            (
                prompt_embeds_t5,
                negative_prompt_embeds_t5,
                prompt_embeds_llama3,
                negative_prompt_embeds_llama3,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

            # CFG concat — this is what makes the DiT batch 2. llama3 concatenates
            # on dim 1, since dim 0 is the 32-layer stack, not the batch.
            if do_classifier_free_guidance:
                prompt_embeds_t5 = torch.cat(
                    [negative_prompt_embeds_t5, prompt_embeds_t5], dim=0
                )
                prompt_embeds_llama3 = torch.cat(
                    [negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1
                )
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                )

            # ─────────────── Latents / timesteps (CPU) ────────────────────
            # bf16 trajectory (dtype follows the pooled embeds, as in the stock
            # pipeline), advanced by the TT transformer outputs.
            latents_h = 2 * (int(height) // (self.config.vae_scale_factor * 2))
            latents_w = 2 * (int(width) // (self.config.vae_scale_factor * 2))
            latents = randn_tensor(
                (batch_size, LATENT_CHANNELS, latents_h, latents_w),
                generator=generator,
                device=torch.device("cpu"),
                dtype=pooled_prompt_embeds.dtype,
            )

            # UniPC branch of __call__; `mu` is only used by the FlowMatchEuler path.
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # ─────── Transformer denoising loop (TT, bf16, sharded) ───────
            logger.info(
                "[STAGE] transformer (sharded, bf16): start ({} steps)",
                num_inference_steps,
            )
            to_dev = lambda x: x.to(dev)  # inputs are already bf16 / int

            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                tt_inputs = [
                    to_dev(latent_model_input),
                    to_dev(timestep),
                    to_dev(prompt_embeds_t5),
                    to_dev(prompt_embeds_llama3),
                    to_dev(pooled_prompt_embeds),
                ]
                t0 = time.perf_counter()
                # .cpu() is the sync point: it forces the graph to run and only
                # returns once the result is on host, so the timer ends there.
                # Kept in bf16: the trajectory below must match the stock pipeline.
                noise_pred = self.transformer(*tt_inputs).cpu()
                self._perf["steps"].append(time.perf_counter() - t0)

                # HiDream predicts the negated flow; the sign flip precedes guidance.
                noise_pred = -noise_pred
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            logger.info("[STAGE] transformer: done")

            # ─────────────── VAE decode (TT, bf16, replicated) ────────────
            logger.info("[STAGE] vae (bf16, replicated): TT")
            vae_config = self.vae.vae.config
            latents = (latents / vae_config.scaling_factor) + vae_config.shift_factor

            # opt_level=1 keeps ttir.group_norm -> ttnn.group_norm; opt_level=0
            # decomposes GroupNorm (reshape+mean+sub) which OOMs the decoder.
            # https://github.com/tenstorrent/tt-xla/issues/4710
            torch_xla.set_custom_compile_options(
                {**self.config.compile_options, "optimization_level": 1}
            )
            t0 = time.perf_counter()
            image = self.vae(to_dev(latents)).cpu()
            self._perf["components"]["vae"] = time.perf_counter() - t0
            # Restore the harness options (un-merge the VAE opt_level bump) so a
            # repeat generate() compiles the other components as before.
            torch_xla.set_custom_compile_options(self.config.compile_options)
            logger.info("[STAGE] vae: done")

            self._perf["total"] = time.perf_counter() - t_total_start
            return image


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale ([-1,1]→[0,255]), reshape and save the pipeline output as PNG."""
    image = (
        (torch.clamp(image.float() / 2 + 0.5, 0.0, 1.0) * 255.0)
        .round()
        .to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
