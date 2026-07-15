# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Infinity 2B — end-to-end text-to-image pipeline for the imagegen harness.

Infinity is an autoregressive next-scale-prediction text-to-image model (not a
diffusion model): a generation is a Python loop over a fixed scale schedule -- a
transformer forward + multinomial sampling + BSQ-VAE code accumulation per scale,
then a single VAE decode. This reimplements the model's ``autoregressive_infer_cfg``
with an explicit CPU/TT device split:

  - Transformer on Tenstorrent, tensor-parallel sharded (mesh (1, num_devices),
    Megatron head-parallel attention from ``loader.load_shard_spec``).
  - T5-XL text encoder, multinomial sampling and BSQ-VAE decode stay on CPU.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from ..loader import ModelLoader, ModelVariant
from . import model as _m

PROMPT = "A fantasy landscape with mountains and rivers"
SEED = 42
# Resolution preset: "1M" -> 1024x1024 (the model's native target);
# "0.25M" -> 512x512; "0.06M" -> 256x256. Output size is derived from the preset.
PN = "1M"
H_DIV_W = 1.0
HEIGHT, WIDTH = _m.dynamic_resolution_h_w[H_DIV_W][PN]["pixel"]
# Transformer weight dtype on TT (bf16 fits 1M in DRAM).
DTYPE = torch.bfloat16


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


class InfinityStep(nn.Module):
    """One next-scale-prediction step of the Infinity transformer, as one forward.

    Mirrors the inference-relevant path of ``Infinity.forward`` -- text
    conditioning -> ``word_embed(norm0_ve(x))`` -> SOS prefix -> lvl embedding ->
    block stack -> logits head -- so the whole device side of a scale is a single
    ``nn.Module.forward``, i.e. a single ``torch.compile`` graph.

    Not ``Infinity.forward`` itself, which carries training-only behaviour that is
    wrong or unhelpful here: the ``cond_drop_rate=0.1`` augmentation (one call in
    ten replaces the prompt conditioning with ``cfg_uncond``), activation
    checkpointing, flex-attn, and ``pad_to_multiplier`` padding.

    Holds the transformer as a submodule, so the parameters (and hence the shard
    spec, which is keyed by parameter) are shared, not copied.

    Args:
        transformer: the ``Infinity`` transformer.
        scale_schedule: the FULL schedule. Passed to the blocks for rope, whose
            precomputed grid is keyed by the full tuple and sliced to the packed
            length; the per-call prefix (``sub_sched``) is what the lvl embedding
            and the block-causal mask are built from.
    """

    def __init__(self, transformer, scale_schedule):
        super().__init__()
        self.transformer = transformer
        self.scale_schedule = scale_schedule

    def forward(
        self,
        kv_raw,
        x_wo_prefix,
        cu_seqlens_k,
        attn_bias,
        sub_sched,
        L_si,
        max_seqlen_k,
    ):
        """Logits for the last ``L_si`` positions of scales ``sub_sched``.

        ``x_wo_prefix`` is the RAW (pre-``word_embed``) packed residual for the
        earlier scales, or ``None`` at the first scale, where the sequence is the
        SOS token alone. Batch-1 by construction (sequential CFG).
        """
        m = self.transformer
        kv = m.text_norm(kv_raw)
        sos = cond_BD = m.text_proj_for_sos((kv, cu_seqlens_k, max_seqlen_k))
        ca_kv = (m.text_proj_for_ca(kv), cu_seqlens_k, max_seqlen_k)
        with torch.amp.autocast("cuda", enabled=False):
            # bf16 throughout (no .float()): an f32 input to the bf16
            # shared_ada_lin Linear yields a mismatched-dtype dot that fails
            # HLO->MHLO conversion on TT.
            gss = m.shared_ada_lin(cond_BD).contiguous()

        x_BLC = sos.unsqueeze(1).expand(1, 1, -1) + m.pos_start.expand(1, 1, -1)
        if x_wo_prefix is not None:
            x_BLC = torch.cat((x_BLC, m.word_embed(m.norm0_ve(x_wo_prefix))), dim=1)
        x_BLC = m.add_lvl_embeding_for_x_BLC(x_BLC, sub_sched)
        for chunk in m.block_chunks:
            x_BLC = chunk(
                x=x_BLC,
                cond_BD=gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=attn_bias,
                attn_fn=None,
                scale_schedule=self.scale_schedule,
                rope2d_freqs_grid=m.rope2d_freqs_grid,
            )
        return m.get_logits(x_BLC[:, -L_si:], cond_BD)


class InfinityConfig:
    def __init__(
        self,
        cfg: float = 3.0,
        tau: float = 0.5,
        top_k: int = 900,
        top_p: float = 0.97,
        pn: str = PN,
        h_div_w: float = H_DIV_W,
        max_scales: Optional[int] = None,
        shard: bool = True,
        transformer_on_tt: bool = True,
    ):
        self.cfg = cfg
        self.tau = tau
        self.top_k = top_k
        self.top_p = top_p
        self.pn = pn
        self.h_div_w = h_div_w
        self.width = WIDTH
        self.height = HEIGHT
        # Scales (transformer passes) to run; None runs the full schedule. A
        # smaller value still decodes a full-resolution (coarse) image, since
        # every scale's codes are accumulated at the final resolution.
        self.max_scales = max_scales
        # Megatron tensor-parallel sharding (needed so the large-scale
        # attention does not OOM).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt


class InfinityPipeline:
    """Infinity 2B pipeline: transformer sharded on TT, sampling + VAE on CPU.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. The
    sharded transformer is placed and wrapped in a ``torch.compile``d
    :class:`InfinityStep` in ``setup()``; each scale's graph is compiled lazily on
    its first forward and reused by the second (uncond) CFG branch and by later
    ``generate()`` calls.
    """

    def __init__(self, config: InfinityConfig):
        self.config = config

    def setup(self):
        self.scale_schedule = self._build_scale_schedule()
        self.load_models()
        self.step = InfinityStep(self.model, self.scale_schedule).eval()
        if not self.config.transformer_on_tt:
            # CPU-only reference run: no device placement, no compile.
            return

        if self.config.shard:
            self.shard_to_tt()
        else:
            self.model = self.model.to(xm.xla_device())
        self._move_rope_cache_to_tt()
        # One static graph per scale (the packed sequence grows each scale), so
        # the default recompile limit of 8 would be hit part-way through the
        # schedule -- and Dynamo would then skip the frame and silently fall back
        # to eager (i.e. lazy-tensor) execution for the remaining scales.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, len(self.scale_schedule) + 8
        )
        # forward, not the module, so self.step stays an nn.Module. dynamic=False:
        # the tt backend compiles static shapes and every scale's length is known,
        # so let each scale specialize instead of letting automatic_dynamic_shapes
        # produce a dynamic graph on the second scale.
        self.step.forward = torch.compile(
            self.step.forward, backend="tt", dynamic=False
        )

    def load_models(self):
        # Loading the transformer side-loads the T5-XL tokenizer/encoder and the
        # BSQ-VAE onto the loader; both stay on CPU.
        self.loader = ModelLoader(ModelVariant.INFINITY_2B)
        self.model = self.loader.load_model(dtype_override=DTYPE)
        self.tokenizer = self.loader.tokenizer
        self.text_encoder = self.loader.text_encoder
        self.vae = self.loader.vae
        self.model_dtype = self.model.pos_start.dtype

    def shard_to_tt(self):
        # Enable SPMD, build the (1, num_devices) mesh, move the transformer to the XLA
        # device, then mark every weight in the Megatron shard spec.
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = self.loader.get_mesh_config(num_devices)
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
        self.model = self.model.to(xm.xla_device())
        for tensor, spec in self.loader.load_shard_spec(self.model).items():
            xs.mark_sharding(tensor, self.mesh, spec)

    def _move_rope_cache_to_tt(self):
        """Move this schedule's precomputed rope2d cache onto the device.

        ``model.rope2d_freqs_grid`` is a plain dict (not a buffer), so
        ``model.to(device)`` leaves it on CPU -- and ``apply_rotary_emb`` does
        ``rope2d_freqs_grid[key] = rope2d_freqs_grid[key].to(qk.device)`` *inside*
        the traced region, which would put a host->device copy in every compiled
        graph. Pre-moving makes that line an identity.
        """
        grid = self.model.rope2d_freqs_grid
        key = str(tuple(self.scale_schedule))
        grid[key] = grid[key].to(xm.xla_device())

    def _build_scale_schedule(self):
        sched = _m.dynamic_resolution_h_w[self.config.h_div_w][self.config.pn]["scales"]
        return [(1, h, w) for (_, h, w) in sched]

    def _ensure_cpu_twin(self):
        """Lazily build the fp32 CPU golden step (only when PCC-checking).

        A second transformer instance kept in fp32 on CPU behind its own
        :class:`InfinityStep`, reusing the already loaded BSQ-VAE (so only the
        transformer weights are re-read). ~8GB CPU RAM, so it is built once, on
        first use. Being the same wrapper, it covers exactly what the TT step
        covers: conditioning, word_embed, blocks and the logits head.
        """
        if getattr(self, "_cpu_twin", None) is None:
            run_args = self.loader._build_run_args()
            twin = _m.load_transformer(self.loader.vae, run_args)
            self._cpu_twin = InfinityStep(
                twin.to("cpu").float().eval(), self.scale_schedule
            ).eval()
        return self._cpu_twin

    def _golden_logits(self, *step_args):
        """Run one step on the fp32 CPU twin.

        Takes the same arguments as the TT step, already on CPU in fp32, so the
        returned logits are a pure fp32 reference for this scale's TT forward.
        """
        return self._ensure_cpu_twin()(*step_args).float()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: Optional[int] = SEED,
        pcc_hook=None,
    ) -> torch.Tensor:
        """Reimplements ``Infinity.autoregressive_infer_cfg`` with a CPU/TT split.

        - T5-XL text encode -> CPU
        - one :class:`InfinityStep` per scale per CFG branch -> TT
          (``torch.compile(backend="tt")``, one graph per scale)
        - multinomial sampling -> CPU
        - BSQ-VAE indices->codes, residual accumulation, decode -> CPU

        Args:
            prompt: text prompt.
            seed: multinomial-sampling seed (``None`` -> unseeded).
            pcc_hook: optional ``callable(tag, device_logits, golden_logits)``.
                When given, every per-scale, per-CFG-branch TT step is repeated on
                a lazy-loaded fp32 CPU twin fed the *same* inputs (raw text
                features, raw packed residual, attn bias), and both logits are
                handed to the hook -- so it measures the bf16-TT step against the
                ideal fp32 reference, isolated per scale (the twin is fed the
                TT-carried state, not its own, so errors do not accumulate across
                scales). ``None`` -> no golden, no overhead.
        """
        m = self.model
        vae = self.vae
        on_tt = self.config.transformer_on_tt

        # CPU <-> TT casts (no-ops when the transformer runs on CPU).
        tt_cast = lambda x: x.to(device=xm.xla_device()) if on_tt else x
        cpu_cast = lambda x: x.to("cpu") if on_tt else x

        scale_schedule = self.scale_schedule
        num_stages_minus_1 = len(scale_schedule) - 1
        tau_list = [self.config.tau] * len(scale_schedule)
        cfg_list = [self.config.cfg] * len(scale_schedule)
        B = 1

        # ── T5-XL text encode (CPU) ───────────────────────────────────
        logger.info("[STAGE] T5 text encode (CPU): start")
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = _m.encode_prompt(
            self.tokenizer, self.text_encoder, prompt
        )
        logger.info("[STAGE] T5 text encode (CPU): done")

        # Seed the model's (CPU) multinomial sampling generator.
        if seed is not None:
            m.rng.manual_seed(seed)
            rng = m.rng
        else:
            rng = None

        # ── Classifier-free guidance: sequential cond + uncond passes ──
        # Two batch-1 forwards per scale, combined on the logits. A batch-2
        # (stacked) forward de-shards the attention score matmul -> OOM at 1M;
        # batch-1 keeps it head-sharded. cfg=1 -> single conditional pass.
        use_cfg = self.config.cfg != 1
        kv_branches = [kv_compact]
        if use_cfg:
            cfg_uncond = m.cfg_uncond.detach().to("cpu", dtype=kv_compact.dtype)
            kv_uncond = kv_compact.clone()
            total = 0
            for le in lens:
                kv_uncond[total : total + le] = cfg_uncond[:le]
                total += le
            kv_branches.append(kv_uncond)

        # ── Per-branch raw text features -> TT (batch=1) ───────────────
        # The conditioning projections live inside the compiled step, so only the
        # raw T5 features cross to the device -- once, then reused every scale.
        cu_seqlens_tt = tt_cast(cu_seqlens_k)
        kv_tt = [tt_cast(kv.to(self.model_dtype)) for kv in kv_branches]

        # ── Next-scale prediction loop (packed recompute, stays sharded) ──
        # No KV cache: each scale rebuilds the full token sequence generated so far
        # and runs all blocks over it in ONE batch-1 forward per CFG branch, with a
        # block-causal attn_bias (each scale attends to itself + earlier scales).
        # The carried state is RAW token embeddings re-projected through the sharded
        # q/k/v weights every scale, so the attention score stays head-sharded. (A
        # KV cache instead de-shards: cached K/V cross the CPU sampling boundary
        # replicated and feed SDPA directly -> 16 heads on one device -> OOM.)
        n_run = len(scale_schedule)
        if self.config.max_scales is not None:
            n_run = min(self.config.max_scales, n_run)

        def _build_attn_bias(sched):
            l_end = sum(int(np.prod(s)) for s in sched)
            d = torch.cat(
                [torch.full((int(np.prod(s)),), i) for i, s in enumerate(sched)]
            ).view(1, l_end, 1)
            bias = torch.where(d >= d.transpose(1, 2), 0.0, -torch.inf)
            return bias.reshape(1, 1, l_end, l_end).float()

        # Per-scale RAW (pre-word_embed) VAE residuals, kept on the host and
        # shared by both CFG branches: the SOS prefix and the word_embed
        # projection are branch-specific and happen inside the step.
        raw_inputs = []
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            if si >= n_run:
                break
            is_last_run = si == n_run - 1
            cfg = cfg_list[si]
            logger.info(f"[STEP] scale {si + 1}/{n_run} {tuple(pn)}")

            sub_sched = scale_schedule[: si + 1]
            L_si = int(np.prod(pn))

            x_cpu = torch.cat(raw_inputs, dim=1) if raw_inputs else None
            x_tt = None if x_cpu is None else tt_cast(x_cpu.to(self.model_dtype))
            bias_cpu = _build_attn_bias(sub_sched)
            bias_tt = tt_cast(bias_cpu.to(self.model_dtype))

            # --- one batch-1 forward per CFG branch (TT, sharded) -> logits (CPU) ---
            branch_tags = ("cond", "uncond")
            branch_logits = []
            for bi, kv in enumerate(kv_tt):
                logits = self.step(
                    kv, x_tt, cu_seqlens_tt, bias_tt, sub_sched, L_si, max_seqlen_k
                )

                tt_logits = cpu_cast(logits).float().mul(1 / tau_list[si])
                branch_logits.append(tt_logits)

                # --- fp32 CPU twin on the same inputs -> golden logits ---
                if pcc_hook is not None:
                    golden = self._golden_logits(
                        kv_branches[bi].float(),
                        None if x_cpu is None else x_cpu.float(),
                        cu_seqlens_k,
                        bias_cpu,
                        sub_sched,
                        L_si,
                        max_seqlen_k,
                    ).mul(1 / tau_list[si])
                    pcc_hook(
                        f"scale {si + 1}/{n_run} {branch_tags[bi]}", tt_logits, golden
                    )

            # CFG combine on logits: cfg*cond + (1-cfg)*uncond.
            if use_cfg:
                logits_BlV = cfg * branch_logits[0] + (1 - cfg) * branch_logits[1]
            else:
                logits_BlV = branch_logits[0]

            # Bit-label codebook: every code is a sequence of binary bits.
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = _m.sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV,
                rng=rng,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_samples=1,
            )[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

            # --- BSQ-VAE: indices -> codes, accumulate residual (CPU) ---
            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1).unsqueeze(1)  # (B,1,h,w,d)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
            if si != num_stages_minus_1:
                # Add this scale's contribution (always at the final resolution).
                summed_codes = summed_codes + F.interpolate(
                    codes, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up
                )
                # On the last executed scale there is no next pass to feed.
                if is_last_run:
                    break
                # Build the next scale's shared RAW input and append it; the step
                # projects it through norm0_ve + word_embed on device.
                next_stage = F.interpolate(
                    summed_codes,
                    size=scale_schedule[si + 1],
                    mode=vae.quantizer.z_interplote_up,
                )
                next_stage = next_stage.squeeze(-3)
                next_stage = next_stage.reshape(*next_stage.shape[:2], -1)
                next_stage = torch.permute(next_stage, [0, 2, 1])  # (B, L_next, d_vae)
                raw_inputs.append(next_stage)
            else:
                summed_codes = summed_codes + codes

        # ── BSQ-VAE decode (CPU) -> RGB image in [-1, 1] ───────────────
        logger.info("[STAGE] BSQ-VAE decode (CPU): start")
        summed_codes = summed_codes.to("cpu")
        img = vae.decode(summed_codes.squeeze(-3))
        logger.info("[STAGE] BSQ-VAE decode (CPU): done")
        return img
