# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — end-to-end text-to-image pipeline for the imagegen harness.

Lumina-Image-2.0 is a flow-matching text-to-image diffusion model, so a single
generation is:

  1. Gemma-2 text encoding of the (system-prompt-prefixed) prompt, and of the
     negative prompt for classifier-free guidance.
  2. A DiT denoising loop -- ``Lumina2Transformer2DModel`` denoises the latent
     over ``num_inference_steps`` FlowMatchEuler steps, two forwards per step
     (conditional + unconditional) with normalization-based guidance.
  3. A single VAE decode of the final latent to an RGB image.

This reimplements the diffusers ``Lumina2Pipeline.__call__`` with an explicit
CPU/TT device split, and -- like the GLM-Image pipeline in this package --
reuses the diffusers pipeline's *own* helper methods so that only the device
split is bespoke. Concretely, a real ``Lumina2Pipeline`` is constructed around
the already-loaded components (``_build_diffusers_pipeline``) and the pipeline's
``encode_prompt`` / ``prepare_latents`` / ``check_inputs`` / ``image_processor``
are called verbatim; the single seam is ``_get_gemma_prompt_embeds``, which is
overridden to run the Gemma-2 forward on TT. That keeps the parts most easily
got wrong by hand -- the system-prompt prefix (applied to the *positive* prompt
only), the negative-prompt path, the ``num_images_per_prompt`` repeat/view,
latent shape and dtype, and the ``[-1, 1] -> uint8`` post-processing --
identical to upstream.

Device map (default ``TT_COMPONENTS`` -- only the DiT on Tenstorrent, as in the
GLM-Image pipeline in this package):

  - Gemma-2 text encoder            -> CPU
  - Lumina2Transformer2DModel (DiT) -> TT (sharded), twice per step for CFG
  - AutoencoderKL decoder           -> CPU
  - tokenizer, scheduler, latent sampling, CFG combine, post-processing -> CPU

Pass ``tt_components`` to put more components on TT; see ``TT_COMPONENTS`` for why
the other two are on CPU today. TT-resident components are 2-D-mesh sharded
("batch", "model"), single-axis Megatron tensor parallelism on the "model" axis
(see ``model_utils``'s ``shard_*_specs``). Only per-stage activations cross the
CPU<->TT boundary.

Notes:
  - The Gemma-2 forward pins ``use_cache=False``, and is wrapped in
    ``_force_hf_mask_tracing`` when it runs on TT; both are graph-capture
    workarounds, not model changes (see that context manager and
    ``_encode_on_tt``).
  - The DiT's RoPE is rewritten to real-valued arithmetic host-side by
    ``load_transformer`` (bit-exact; see ``model_utils``), because tt-mlir has no
    lowering for the complex ops diffusers' Lumina RoPE emits.
  - The DiT is run through ``torch.compile(backend="tt")``
    (``_compile_for_tt``). That is a *numerics* requirement, not a perf one --
    the StableHLO composite pass only runs in that backend. See
    ``LuminaImageConfig.compile_on_tt``.
  - The negative caption mask is widened to the positive's token count so both
    CFG forwards share one DiT executable (``_pad_caption_mask``). This one is a
    workaround for an open TT bug and *does* deviate from upstream semantics;
    see ``LuminaImageConfig.pad_negative_caption``.
"""

import os
from contextlib import contextmanager, nullcontext
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers.masking_utils as hf_masking_utils
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .model_utils import DTYPE

# A long, detailed caption in the style Lumina-Image-2.0 is documented on (see
# ``Lumina2Pipeline``'s diffusers example) -- stacked subject, material, lighting
# and camera terms are what its system prompt ("generate superior images with the
# superior degree of image-text alignment") is tuned for, and it exercises a
# realistic caption length rather than a near-minimal one.
PROMPT = (
    "Fresh handmade ramen served in a traditional ceramic bowl, rich broth, "
    "perfectly sliced pork, soft-boiled egg, fresh herbs, chopsticks, dramatic "
    "restaurant lighting, professional food photography, shallow depth of field."
)
# Lumina-T2I's negative prompt is the empty string (see Lumina2Pipeline docs).
NEGATIVE_PROMPT = ""
SEED = 42
# Native Lumina-Image-2.0 resolution (sample_size 128 * vae_scale_factor 8).
# Both dims must be divisible by 16 (vae_scale_factor * 2).
HEIGHT = 1024
WIDTH = 1024
# Lumina2Pipeline.__call__ defaults.
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0
MAX_SEQUENCE_LENGTH = 256
CFG_TRUNC_RATIO = 1.0
CFG_NORMALIZATION = True
# Weight dtype for every component on TT (bf16 fits DRAM).
MODEL_DTYPE = DTYPE
# Components routed through the ``tt`` torch.compile backend by
# ``_compile_for_tt``. Attribute names on the pipeline; each must be invoked as
# ``self.<name>(...)`` so the returned OptimizedModule is what actually runs.
#
# Only the DiT. The other two are deliberately absent:
#   - text encoder: ``_encode_on_tt`` calls the *inner*
#     ``self.text_encoder.encoder`` (not the wrapper) from inside the
#     ``_force_hf_mask_tracing`` mock patch, so compiling the wrapper would have
#     no effect and compiling the inner model would put a Python-level
#     monkeypatch of ``masking_utils.is_tracing`` inside a dynamo trace. It
#     measures PCC 0.9997 as is.
#   - VAE: on CPU by default (see TT_COMPONENTS), so there is nothing to
#     compile. If it is moved back onto TT it must be added here too -- running it
#     on TT *without* the composite pass is what produced a pure-noise image.
TT_BACKEND_COMPONENTS = ("transformer",)
# Which learned components are resident on Tenstorrent; everything else stays on
# CPU. Matches the GLM-Image pipeline in this package, where only the DiT runs on
# TT.
#
# The DiT is the component under test and the one that dominates runtime (2
# forwards per denoising step). The other two are on CPU because neither is
# validated on TT today:
#   - VAE: ``test_vae_decoder_sharded`` is skipped ("Timed out in CI"), and on TT
#     it emitted 30 ``stablehlo.batch_norm_training`` ops and zero composites --
#     the same signature that cost the DiT 0.918 PCC -- which is what produced a
#     pure-noise image while the DiT itself measured 0.99997 at every step.
#   - text encoder: measures 0.9997 on TT, but it cannot get the composite pass
#     through ``_compile_for_tt`` (see TT_BACKEND_COMPONENTS), so CPU keeps it a
#     clean reference rather than a second uncomposited graph.
TT_COMPONENTS = ("transformer",)
# Default for ``LuminaImageConfig.pad_negative_caption`` -- see there.
PAD_NEGATIVE_CAPTION = True
# Pipeline attribute names of the three learned components, in pipeline order.
COMPONENT_NAMES = ("text_encoder", "transformer", "vae")


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) -- required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


@contextmanager
def _force_hf_mask_tracing():
    """Make transformers' attention-mask helpers take their ``is_tracing`` path.

    Gemma-2 builds its causal mask via ``transformers.masking_utils``, whose
    helpers (``_ignore_causal_mask_sdpa``, ``find_packed_sequence_indices``) run
    data-dependent ``.all()`` checks inside plain Python ``if`` statements --
    gated on ``not is_tracing(tensor)``. ``is_tracing`` only detects dynamo /
    jit / fx / fake-tensor, NOT torch_xla's lazy tensors, so on TT those ``.all()``
    reductions execute on device and lower to mesh-distributed ``prod`` ops the
    multichip runtime cannot collapse to a single buffer (error code 13 /
    "Can't get a single buffer ... distributed over mesh shape [1, 8]").

    Forcing ``is_tracing`` True makes these helpers skip the data-dependent
    branches and emit a fully static causal mask (single-sequence packed mask of
    zeros, broadcast-built 4D mask) -- the correct behavior for a graph-captured
    backend like torch_xla, and numerically the same mask the skipped branch
    would have produced. Scoped to the encoder forward only.
    """
    from unittest.mock import patch

    with patch.object(hf_masking_utils, "is_tracing", lambda *a, **k: True):
        yield


class LuminaImageConfig:
    """Generation knobs; every default matches ``Lumina2Pipeline.__call__``."""

    def __init__(
        self,
        on_tt: bool = True,
        shard: bool = True,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        cfg_trunc_ratio: float = CFG_TRUNC_RATIO,
        cfg_normalization: bool = CFG_NORMALIZATION,
        compile_on_tt: bool = True,
        tt_components=TT_COMPONENTS,
        pad_negative_caption: bool = PAD_NEGATIVE_CAPTION,
    ):
        self.on_tt = on_tt
        # 2-D-mesh single-axis Megatron tensor-parallel sharding (see
        # model_utils.shard_*_specs). Needed so the large transformer/VAE
        # activations do not OOM on a single device. Set False to run each
        # component unsharded on one TT device (only viable at tiny sizes).
        self.shard = shard
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        # Ratio of the timestep interval over which normalization-based guidance
        # is applied; 1.0 (the upstream default) means CFG on every step.
        self.cfg_trunc_ratio = cfg_trunc_ratio
        self.cfg_normalization = cfg_normalization
        # Route TT_BACKEND_COMPONENTS through torch.compile(backend="tt").
        #
        # Not a perf knob -- it is required for numerics. The composite pass
        # (tt_torch.backend.passes.handle_composite_ops) only runs inside the
        # "tt" torch.compile backend, and it is what emits the
        # ``tenstorrent.scaled_dot_product_attention`` / ``layer_norm`` /
        # ``gather`` StableHLO composites that tt-mlir maps to fused kernels.
        # Executing the modules as plain LazyTensor graphs instead (the old
        # behaviour) emits *zero* composites: all 30 DiT attentions fall back to
        # a generic decomposed SDPA, and ``norm_out``'s LayerNorm lowers to
        # ``stablehlo.batch_norm_training`` in bf16 -- accumulating mean/variance
        # over 2304 channels in bf16 in the last op before the output
        # projection. That measured DiT PCC 0.9179 e2e while the component test
        # (which goes through ``run_graph_test`` -> backend="tt") passed 0.99 on
        # byte-identical shapes and masks.
        self.compile_on_tt = compile_on_tt
        # Which components are resident on TT; the rest run on CPU. Gated by
        # ``on_tt`` -- with ``on_tt=False`` nothing goes to device regardless.
        self.tt_components = frozenset(tt_components)
        # Widen the negative caption mask so both CFG forwards run one DiT
        # executable (see ``LuminaImagePipeline._pad_caption_mask``).
        #
        # WORKAROUND for an open TT bug, not a model change. The positive caption
        # (system prompt + PROMPT) tokenizes to many more real tokens than
        # NEGATIVE_PROMPT ("", 1 token), so the two CFG forwards compile two
        # shape-specialized executables. With both resident and alternating, the
        # second one is correct on its *first* execution and then decays
        # monotonically -- measured 0.999965 -> 0.9754 -> 0.9150 -> 0.8438 over
        # steps 1-4 -- while the first holds 0.99997 on the same latents. That was
        # characterized at joint sequences 4134 / 4097 (a 38-token caption); the
        # exact lengths move with PROMPT but the alternation does not. Every
        # single-executable configuration is clean: 30 steps with CFG off, and 60
        # forwards with the two forwards sharing one executable. So this collapses
        # the run to one executable and sidesteps it.
        #
        # Set False to reproduce the bug. Remove once the underlying
        # cross-executable issue is fixed.
        self.pad_negative_caption = pad_negative_caption

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1


class LuminaImagePipeline:
    """Lumina-Image-2.0 pipeline; ``TT_COMPONENTS`` selects what runs on TT."""

    def __init__(self, config: Optional[LuminaImageConfig] = None):
        self.config = config if config is not None else LuminaImageConfig()

    # ── setup ──────────────────────────────────────────────────────────
    def setup(self):
        self.load_models()
        if self.config.on_tt:
            if self.config.shard:
                self.shard_to_tt()
            else:
                # Unsharded: move the TT-resident components to a single device.
                dev = xm.xla_device()
                for name in self._tt_resident():
                    setattr(self, name, getattr(self, name).to(dev))
            if self.config.compile_on_tt:
                self._compile_for_tt()
        logger.info(
            f"Device map: TT={sorted(self._tt_resident())}, "
            f"CPU={sorted(set(COMPONENT_NAMES) - self._tt_resident())}"
        )

    def _tt_resident(self) -> frozenset:
        """Component names actually resident on TT for this run."""
        if not self.config.on_tt:
            return frozenset()
        return frozenset(self.config.tt_components)

    def _on_tt(self, name: str) -> bool:
        return name in self._tt_resident()

    def _compile_for_tt(self):
        """Route ``TT_BACKEND_COMPONENTS`` through the ``tt`` compile backend.

        Must run *after* the components are on device and sharded: dynamo traces
        on first call, so the marked shardings are already attached to the
        weights by then and ``load_shard_spec``'s attribute walk (which needs the
        plain module) has already happened.

        ``torch.compile`` returns an ``OptimizedModule`` wrapper, which stays
        transparent for the two things done to these components afterwards:
        attribute reads forward to ``_orig_mod`` via ``__getattr__``, and
        assigning ``.forward`` (how the nightly test's PCC gate wraps each stage)
        lands on the wrapper itself -- ``"forward"`` is in dynamo's
        ``_opt_mod_attributes`` -- so the gate shadows the dynamo entry point and
        still calls it, rather than being silently bypassed or pushed inside the
        traced region.
        """
        for name in TT_BACKEND_COMPONENTS:
            # Only meaningful for components actually resident on TT -- a CPU
            # module must not be handed to the tt backend.
            if not self._on_tt(name):
                continue
            module = getattr(self, name)
            setattr(self, name, torch.compile(module, backend="tt"))
            logger.info(f"Compiled {name} with the tt backend (composite ops on).")

    def load_models(self):
        # Each Lumina-Image-2.0 component is an independently loadable variant of
        # the same repo; load_model returns a plain-tensor-forward wrapper.
        from ..loader import ModelLoader, ModelVariant

        self.te_loader = ModelLoader(ModelVariant.TEXT_ENCODER)
        self.tf_loader = ModelLoader(ModelVariant.TRANSFORMER)
        self.vae_loader = ModelLoader(ModelVariant.VAE)

        self.model_dtype = MODEL_DTYPE
        self.text_encoder = self.te_loader.load_model(dtype_override=self.model_dtype)
        self.transformer = self.tf_loader.load_model(dtype_override=self.model_dtype)
        self.vae = self.vae_loader.load_model(dtype_override=self.model_dtype)

        self.pipe = self._build_diffusers_pipeline()
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

    def _build_diffusers_pipeline(self):
        """Build a real ``Lumina2Pipeline`` around the loaded components.

        Nothing is downloaded beyond the tokenizer and the scheduler config --
        the transformer / text encoder / VAE are the modules already loaded
        above (unwrapped from their plain-tensor forward wrappers). The pipeline
        object is never ``__call__``-ed; it is here so ``generate`` can reuse
        upstream's ``encode_prompt``, ``prepare_latents``, ``check_inputs``,
        ``system_prompt`` and ``image_processor`` verbatim instead of
        reimplementing them.

        ``_get_gemma_prompt_embeds`` is the one method overridden: it is the
        exact seam where the text encode has to move to TT. Everything
        ``encode_prompt`` layers on top of it -- prefixing the system prompt to
        the *positive* prompt only, the negative-prompt path, the
        ``num_images_per_prompt`` repeat/view -- stays upstream's code.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler, Lumina2Pipeline
        from transformers import AutoTokenizer

        repo_id = self.te_loader._variant_config.pretrained_model_name
        pipe = Lumina2Pipeline(
            transformer=self.transformer.transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                repo_id, subfolder="scheduler"
            ),
            vae=self.vae.vae,
            text_encoder=self.text_encoder.encoder,
            tokenizer=AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer"),
        )
        pipe._get_gemma_prompt_embeds = self._get_gemma_prompt_embeds
        return pipe

    def shard_to_tt(self):
        # Enable SPMD, build the ("batch", "model") mesh once, then move each
        # TT-resident component (``TT_COMPONENTS``) to the XLA device and mark
        # every weight in its shard spec -- mirrors the runtime sharding the graph
        # tester does, and the specs all target the same mesh. Those weights stay
        # resident + sharded for the whole run while only activations cross the
        # CPU<->TT boundary. Components not on TT are left untouched on CPU.
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = self.te_loader.get_mesh_config(num_devices)
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
        logger.info(f"Created device mesh: {mesh_shape} with {num_devices} devices.")

        dev = xm.xla_device()
        loaders = {
            "text_encoder": self.te_loader,
            "transformer": self.tf_loader,
            "vae": self.vae_loader,
        }
        for name in COMPONENT_NAMES:
            if not self._on_tt(name):
                continue
            module = getattr(self, name).to(dev)
            setattr(self, name, module)
            for tensor, spec in loaders[name].load_shard_spec(module).items():
                xs.mark_sharding(tensor, self.mesh, spec)

    @staticmethod
    def _pad_caption_mask(negative_mask, positive_mask):
        """Widen the negative caption mask to the positive's real-token count.

        Keeps both CFG forwards on one DiT executable. ``Lumina2RotaryPosEmbed``
        takes ``l_effective_cap_len`` from ``attention_mask.sum(dim=1)``, and that
        number fixes the joint sequence length and hence which shape-specialized
        executable runs. Since ``NEGATIVE_PROMPT`` ("") is 1 token against the
        positive caption's many, the loop otherwise alternates two executables --
        see ``pad_negative_caption`` for why that is currently broken. Reads the
        target off ``positive_mask``, so it follows ``PROMPT`` whatever its length.

        Semantics: the uncond forward then attends over the extra Gemma-2 padding
        embeddings instead of masking them, so it is not the reference
        unconditional prediction. Measured harmless on PROMPT/SEED (a coherent
        1024x1024 generation, DiT PCC >=0.9998 across all 60 forwards), but it is
        a real deviation, not a no-op.
        """
        padded = negative_mask.clone()
        for i in range(padded.shape[0]):
            before = int(negative_mask[i].sum())
            target = int(positive_mask[i].sum())
            if target <= before:
                continue
            padded[i, :target] = 1
            logger.info(
                f"[CFG] negative caption mask row {i}: {before} -> "
                f"{int(padded[i].sum())} real tokens, matching the positive prompt "
                f"so both CFG forwards share one DiT executable"
            )
        return padded

    # ── per-stage CPU<->TT casts ───────────────────────────────────────
    # Per component: a stage running on CPU must not have its inputs pushed to
    # the XLA device, so these take the component name rather than assuming every
    # stage is on TT.
    def _to_stage(self, name: str, x):
        return x.to(device=xm.xla_device()) if self._on_tt(name) else x.to("cpu")

    def _cpu(self, x):
        return x.to("cpu")

    # ── text encode: the one upstream method that has to change ────────
    @torch.no_grad()
    def _get_gemma_prompt_embeds(
        self,
        prompt,
        device: Optional[torch.device] = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    ):
        """TT override of ``Lumina2Pipeline._get_gemma_prompt_embeds``.

        Line-for-line the upstream method -- tokenize to a fixed length, warn on
        truncation, run Gemma-2, take the second-to-last hidden state, cast to
        the text encoder's dtype -- with the encoder forward routed to TT and the
        results returned on ``device`` (CPU) for the host-side remainder of
        ``encode_prompt``. ``prompt`` arrives already system-prompt-prefixed (or
        as the raw negative prompt) from ``encode_prompt``, exactly as upstream.
        """
        tokenizer = self.pipe.tokenizer
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because Gemma can "
                f"only handle sequences up to {max_sequence_length} tokens: "
                f"{removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask
        prompt_embeds = self._encode_on_tt(text_input_ids, prompt_attention_mask)
        prompt_embeds = prompt_embeds.to(dtype=self.model_dtype, device=device)
        return prompt_embeds, prompt_attention_mask.to(device)

    def _encode_on_tt(self, input_ids, attention_mask):
        """Run the Gemma-2 forward, returning ``hidden_states[-2]`` on CPU.

        Named for its original role; it runs on whichever device the text encoder
        is resident on (CPU under the default ``TT_COMPONENTS``).

        ``use_cache=False`` is required to avoid the Gemma-2 sliding-window cache
        slice that exceeds tt-mlir's slice bound (tenstorrent/tt-xla#4900); a
        single encode pass needs no cache. Kept unconditional so the CPU and TT
        paths encode identically.

        On TT the forward is wrapped in ``_force_hf_mask_tracing`` so Gemma-2's
        mask helpers emit a static causal mask instead of running data-dependent
        ``.all()`` reductions on device (see that context manager). The real 2D
        padding mask is passed (not ``None``): with ``attention_mask=None``,
        ``_preprocess_mask_arguments`` would call ``find_packed_sequence_indices``,
        whose int ``cumsum`` lowers to a tt-metal accumulation kernel that rejects
        the bool->int64 dtype ("Unsupported data format for add_int"). Passing the
        mask skips that packed-sequence branch entirely, and with tracing forced
        the helper builds the padding+causal 4D mask from elementwise ops only.

        That patch is scoped to the TT path: on CPU the data-dependent branches
        are harmless, and skipping the patch keeps this the plain upstream HF
        forward -- i.e. a clean reference rather than a second variant.
        """
        on_tt = self._on_tt("text_encoder")
        mask_ctx = _force_hf_mask_tracing() if on_tt else nullcontext()
        with mask_ctx:
            out = self.text_encoder.encoder(
                input_ids=self._to_stage("text_encoder", input_ids),
                attention_mask=self._to_stage("text_encoder", attention_mask),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return self._cpu(out.hidden_states[-2])

    # ── generation ─────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``Lumina2Pipeline.__call__`` with a CPU/TT split.

          - Gemma-2 text encode            -> TT
          - transformer denoising forwards -> TT (twice per step for CFG)
          - scheduler step + CFG combine   -> CPU
          - AutoencoderKL decode           -> TT
          - post-processing                -> CPU (diffusers image processor)

        The numbered steps below are ``Lumina2Pipeline.__call__``'s own. Returns
        whatever upstream returns for ``output_type``: ``"pil"`` a list of
        ``PIL.Image``, ``"np"`` a ``(B, H, W, 3)`` array in ``[0, 1]``, ``"pt"``
        a ``(B, 3, H, W)`` tensor in ``[0, 1]``, ``"latent"`` the raw latent.
        """
        from diffusers.pipelines.lumina2.pipeline_lumina2 import (
            calculate_shift,
            retrieve_timesteps,
        )

        cfg = self.config
        pipe = self.pipe
        scheduler = self.scheduler
        do_cfg = cfg.do_classifier_free_guidance
        height, width = cfg.height, cfg.width
        num_inference_steps = cfg.num_inference_steps
        # Host device for everything outside the three TT stages.
        cpu = torch.device("cpu")

        # 1. Check inputs. Raise error if not correct.
        pipe.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            max_sequence_length=cfg.max_sequence_length,
        )

        # 2. Define call parameters.
        batch_size = 1

        # 3. Encode input prompt -- upstream encode_prompt, Gemma-2 forward on TT
        #    via the _get_gemma_prompt_embeds override. The system prompt is
        #    prefixed to the positive prompt only (upstream behavior); the
        #    negative prompt is encoded raw.
        logger.info("[STAGE] Gemma-2 text encode (TT): start")
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt,
            do_cfg,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device=cpu,
            max_sequence_length=cfg.max_sequence_length,
        )
        logger.info("[STAGE] Gemma-2 text encode (TT): done")

        if (
            do_cfg
            and cfg.pad_negative_caption
            and negative_prompt_attention_mask is not None
        ):
            negative_prompt_attention_mask = self._pad_caption_mask(
                negative_prompt_attention_mask, prompt_attention_mask
            )

        # 4. Prepare latents. Sampled on CPU as upstream does (randn_tensor with
        #    a CPU generator), so the values are seed-identical to a CPU run.
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        latents = pipe.prepare_latents(
            batch_size,
            pipe.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            cpu,
            generator,
        )

        # 5. Prepare timesteps (flow-match scheduler, CPU).
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # NOTE: upstream passes latents.shape[1] (the latent channel count) here,
        # not the patched image sequence length. Kept verbatim so the shift --
        # and hence the whole timestep schedule -- matches the reference.
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        # Upstream also pins the timestep schedule to CPU when XLA is available.
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu, sigmas=sigmas, mu=mu
        )
        num_train_timesteps = scheduler.config.num_train_timesteps

        # Loop-invariant conditioning: cast to TT once, reused every step.
        prompt_embeds_tt = self._to_stage("transformer", prompt_embeds)
        prompt_mask_tt = self._to_stage("transformer", prompt_attention_mask)
        if do_cfg:
            negative_prompt_embeds_tt = self._to_stage(
                "transformer", negative_prompt_embeds
            )
            negative_prompt_mask_tt = self._to_stage(
                "transformer", negative_prompt_attention_mask
            )

        # 6. Denoising loop (transformer on TT, scheduler on CPU).
        for i, t in enumerate(timesteps):
            logger.info(
                f"[STEP] denoise {i + 1}/{num_inference_steps} (t={float(t):.4f})"
            )
            # compute whether apply classifier-free truncation on this timestep
            do_classifier_free_truncation = (
                i + 1
            ) / num_inference_steps > cfg.cfg_trunc_ratio
            # reverse the timestep since Lumina uses t=0 as the noise and t=1 as
            # the image
            current_timestep = 1 - t / num_train_timesteps
            # broadcast to batch dimension
            current_timestep = current_timestep.expand(latents.shape[0])

            timestep_tt = self._to_stage("transformer", current_timestep)
            latents_tt = self._to_stage("transformer", latents)

            noise_pred_cond = self._cpu(
                self.transformer(
                    latents_tt, timestep_tt, prompt_embeds_tt, prompt_mask_tt
                )
            ).float()

            # perform normalization-based guidance scale on a truncated timestep
            # interval
            if do_cfg and not do_classifier_free_truncation:
                noise_pred_uncond = self._cpu(
                    self.transformer(
                        latents_tt,
                        timestep_tt,
                        negative_prompt_embeds_tt,
                        negative_prompt_mask_tt,
                    )
                ).float()
                noise_pred = noise_pred_uncond + cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # apply normalization after classifier-free guidance
                if cfg.cfg_normalization:
                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_pred = noise_pred * (cond_norm / noise_norm)
            else:
                noise_pred = noise_pred_cond

            # compute the previous noisy sample x_t -> x_t-1
            noise_pred = -noise_pred
            latents = scheduler.step(
                noise_pred.to(latents.dtype), t, latents, return_dict=False
            )[0]

            if cfg.on_tt:
                xm.mark_step()

        # 7. VAE decode (TT) -> RGB image in [-1, 1], then post-process.
        if output_type == "latent":
            return latents
        logger.info("[STAGE] VAE decode (TT): start")
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor
        image = self._cpu(self.vae(self._to_stage("vae", latents)))
        logger.info("[STAGE] VAE decode (TT): done")

        # Post-process ([-1, 1] -> output_type) via the diffusers image
        # processor, matching ``Lumina2Pipeline.__call__``.
        return pipe.image_processor.postprocess(image, output_type=output_type)
