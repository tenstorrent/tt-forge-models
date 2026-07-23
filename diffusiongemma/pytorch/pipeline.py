# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma 26B block-diffusion text-generation pipeline on Tenstorrent.

Both the encoder (prefill) and the decoder (denoising loop) run on TT (sharded, SPMD). The
model can't fit on device twice, so residency is STAGED: the encoder is loaded as an
independent model, prefills the KV cache, is freed, then the decoder is loaded -- only one
component is device-resident at a time.

The host driver (sampler/stopping/cache/RNG) reuses the model's own generate() helpers, so it
is bit-identical to generate(); only the two NN components run on TT. ``manual_generate`` and
the ``TT*`` wrappers are module-level so a PCC test can reuse them with injected forwards.

NOTE: the caller must install transformers>=5.11 (the version swap lives on the tt-xla side,
which this repo can't import); ``setup()`` re-registers tt_moe against that swapped-in version.
"""

import copy
import gc
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from tt_torch.moe_backend import TT_MOE_BACKEND_NAME, register_tt_moe_backend

from .loader import ModelLoader, ModelVariant

PROMPT = "Why is the sky blue?"
MAX_NEW_TOKENS = 256  # one canvas block
SEED = 0


def enable_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def make_mesh(mesh_shape, mesh_names) -> Mesh:
    device_ids = np.array(range(xr.global_runtime_device_count()))
    return Mesh(device_ids, mesh_shape, mesh_names)


def to_device(obj, device):
    """Recursively move tensors in a tensor / dict / list / tuple to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def cache_to_device(cache, device):
    """Deep-copy a DynamicCache and move its per-layer key/value tensors to ``device``."""
    dev = copy.deepcopy(cache)
    for layer in dev.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
    return dev


def free_tt_graphs():
    """Fully release baked TT graph weights (`.to('cpu')`/`del` free only activations): reset
    dynamo, null the torch_xla GraphInputMatcher tensors, clear the C++ computation cache.
    Safe because each staged component is an independent model (nothing else pins its weights).
    """
    from torch_xla._dynamo.dynamo_bridge import GraphInputMatcher

    torch._dynamo.reset()
    for obj in gc.get_objects():
        try:
            if (
                isinstance(obj, torch.fx.GraphModule)
                and getattr(obj, "xla_args", None) is not None
            ):
                obj.xla_args = None
            if isinstance(obj, GraphInputMatcher):
                for ref in gc.get_referrers(obj):
                    if isinstance(ref, tuple):
                        for d in gc.get_referrers(ref):
                            if isinstance(d, dict):
                                d.clear()
        except ReferenceError:
            continue
    xr.clear_computation_cache()
    gc.collect()


class TTEncoder(torch.nn.Module):
    """torch.compile needs tensor I/O: returns last_hidden_state (cache updated in place)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        mm_token_type_ids=None,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
        ).last_hidden_state


class TTDecoder(torch.nn.Module):
    """torch.compile needs tensor I/O: hardcodes input_ids=None (decoder path), returns logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        decoder_input_ids,
        decoder_position_ids,
        self_conditioning_logits,
        decoder_attention_mask,
        self_conditioning_mask,
        past_key_values,
    ):
        return self.model(
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        ).logits


@torch.no_grad()
def manual_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    *,
    encoder_forward=None,
    decoder_forward=None,
    **model_kwargs,
):
    """With-cache mimic of generate() (verified bit-identical). Only the encoder prefill and
    decoder forward are swappable (default to the model's own) for TT injection."""
    encoder_forward = encoder_forward or model.model.encoder
    decoder_forward = decoder_forward or model.forward

    gen_cfg, model_kwargs = model._prepare_generation_config(
        None, max_new_tokens=max_new_tokens, **model_kwargs
    )
    batch_size, cur_len = input_ids.shape
    max_length, max_new_tokens = model._prepare_generated_length(gen_cfg, cur_len)
    max_new_canvases = math.ceil(max_new_tokens / model.config.canvas_length)
    # Single canvas block only: encoder_forward starts a fresh KV cache each block, so multi-block
    # would drop the prior block's context. Multi-block needs cross-block cache carry-over.
    assert (
        max_new_canvases == 1
    ), f"pipeline supports a single canvas block (=1) for now; got max_new_canvases={max_new_canvases} (multi-block needs KV-cache carry-over)"

    device = input_ids.device
    canvas_length = model.config.canvas_length
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past_key_values = model._prepare_cache_for_generation(
        generation_config=gen_cfg,
        batch_size=batch_size,
        max_length=max_length - canvas_length,
    )
    eos_tensor = (
        torch.tensor(gen_cfg.eos_token_id, device=device)
        if gen_cfg.eos_token_id is not None
        else None
    )
    encoder_position_ids = torch.arange(
        cur_len - input_ids.shape[1], cur_len, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_position_ids = torch.arange(
        cur_len, cur_len + canvas_length, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_attention_mask = torch.nn.functional.pad(
        attention_mask, (0, canvas_length), value=True
    )

    sampler = model._prepare_sampler(gen_cfg)
    logits_processor = model._prepare_logits_processor(gen_cfg, None)
    ar_stopping = model._prepare_ar_stopping_criteria(gen_cfg, None)
    diffusion_stopping = model._prepare_diffusion_stopping_criteria(gen_cfg)

    is_prefill = True
    for block in range(max_new_canvases):
        unprocessed_input_ids, encoder_mask_mapping = model._prepare_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            canvas_length=canvas_length,
            batch_size=batch_size,
            **model_kwargs,
        )
        encoder_outputs = encoder_forward(
            input_ids=unprocessed_input_ids,
            attention_mask=encoder_mask_mapping,
            past_key_values=past_key_values,
            position_ids=encoder_position_ids,
            **model_kwargs,
        )
        past_key_values = encoder_outputs.past_key_values
        is_prefill = False

        (
            current_canvas,
            self_conditioning_logits,
            mask_mapping,
            finished_denoising,
        ) = model._prepare_denoiser_inputs(
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            sampler=sampler,
            diffusion_stopping_criteria=diffusion_stopping,
            batch_size=batch_size,
            device=device,
            model_kwargs=model_kwargs,
        )
        argmax_canvas = current_canvas

        for cur_step in reversed(range(1, gen_cfg.max_denoising_steps + 1)):
            (
                current_canvas,
                argmax_canvas,
                self_conditioning_logits,
                finished_denoising,
            ) = model._denoising_step(
                decoder_forward=decoder_forward,
                current_canvas=current_canvas,
                argmax_canvas=argmax_canvas,
                input_ids=input_ids,
                decoder_position_ids=decoder_position_ids,
                self_conditioning_logits=self_conditioning_logits,
                mask_mapping=mask_mapping,
                past_key_values=past_key_values,
                finished_denoising=finished_denoising,
                cur_step=cur_step,
                sampler=sampler,
                logits_processor=logits_processor,
                diffusion_stopping_criteria=diffusion_stopping,
                **model_kwargs,
            )
            if torch.all(finished_denoising):
                break

        logger.info("block {}/{} done", block + 1, max_new_canvases)
        input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
        input_ids, finished_sequences = model._finalize_canvas(
            input_ids=input_ids,
            finished_sequences=finished_sequences,
            generation_config=gen_cfg,
            stopping_criteria=ar_stopping,
            canvas_length=canvas_length,
            eos_tensor=eos_tensor,
        )
        if torch.all(finished_sequences):
            break
        (
            cur_len,
            decoder_attention_mask,
            attention_mask,
            encoder_position_ids,
            decoder_position_ids,
        ) = model._prepare_kwargs_for_next_canvas(
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            canvas_length=canvas_length,
            cur_len=cur_len,
            is_compiling=False,
        )

    return input_ids


class DiffusionGemmaConfig:
    def __init__(self, max_new_tokens: int = MAX_NEW_TOKENS, seed: int = SEED):
        self.max_new_tokens = max_new_tokens
        self.seed = seed


class DiffusionGemmaPipeline:
    """Staged both-on-TT block-diffusion text generation.

    ``setup()`` enables SPMD, re-registers tt_moe (against the caller's swapped-in transformers),
    loads the host driver model, and builds the device mesh. ``generate()`` runs the staged
    encoder/decoder on TT and returns the decoded text.
    """

    def __init__(self, config: DiffusionGemmaConfig = None):
        self.config = config or DiffusionGemmaConfig()
        self.loader = None
        self.cpu_model = None
        self.mesh = None
        self.xla = None

    def setup(self):
        enable_spmd()
        # Re-register tt_moe against the live transformers (the caller's version swap wiped
        # moe_backend's import-time registration). See tenstorrent/tt-xla#5424.
        register_tt_moe_backend()
        self.loader = ModelLoader()
        # CPU model: host driver only (sampler/stopping/cache/positions); runs no NN forward.
        self.cpu_model = self.loader.load_model(dtype_override=torch.bfloat16)
        self.cpu_model.eval()
        self.cpu_model.config._experts_implementation = TT_MOE_BACKEND_NAME
        self.mesh = make_mesh(
            *self.loader.get_mesh_config(xr.global_runtime_device_count())
        )
        self.xla = xm.xla_device()

    def _load_sharded(self, variant):
        # Consumer applies the loader's spec: loader returns load_shard_spec; we .to() + shard.
        vl = ModelLoader(variant)
        model = vl.load_model(dtype_override=torch.bfloat16)
        vl.config._experts_implementation = TT_MOE_BACKEND_NAME
        model = model.to(self.xla)
        xs.set_global_mesh(self.mesh)  # tt_moe reads get_global_mesh() for the EP axis
        for tensor, spec in vl.load_shard_spec(model).items():
            xs.mark_sharding(tensor, self.mesh, spec)
        return model

    def _staged_forwards(self, vocab_size):
        """Build the drive-with-TT encoder/decoder forwards; one component resident at a time."""
        from transformers import DynamicCache  # 5.12.0, after the caller's swap

        xla = self.xla
        tt_pkv = {
            "host": None,
            "pkv": None,
        }  # host: CPU cache handed encoder -> decoder
        stage = {
            "dec_tt": None,
            "dec_model": None,
        }  # decoder graph during the decode loop

        def encoder_forward(**kw):
            # Free the previous block's decoder (if any) so only one model is resident.
            if stage["dec_tt"] is not None:
                stage["dec_tt"] = stage["dec_model"] = None
                free_tt_graphs()
            enc_model = self._load_sharded(ModelVariant.ENCODER)
            enc_tt = torch.compile(TTEncoder(enc_model), backend="tt")
            pkv = DynamicCache()
            lhs = enc_tt(
                to_device(kw["input_ids"], xla),
                to_device(kw["attention_mask"], xla),
                to_device(kw["position_ids"], xla),
                pkv,
                to_device(kw.get("mm_token_type_ids"), xla),
            )
            # Cache to host + FREE the encoder; the decoder loads lazily in decoder_forward.
            xm.mark_step()
            tt_pkv["host"] = cache_to_device(pkv, "cpu")
            del enc_tt, enc_model, pkv
            free_tt_graphs()
            return SimpleNamespace(
                last_hidden_state=lhs.to("cpu"), past_key_values=tt_pkv["host"]
            )

        def decoder_forward(**kw):
            # First decode step: encoder is freed, so load the decoder now (vocab-shard
            # lm_head/embed so decoder + logits fit) and restore the KV cache from host.
            if stage["dec_tt"] is None:
                dec_model = self._load_sharded(ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT)
                xs.mark_sharding(dec_model.lm_head.weight, self.mesh, ("model", None))
                stage["dec_model"] = dec_model
                stage["dec_tt"] = torch.compile(TTDecoder(dec_model), backend="tt")
                tt_pkv["pkv"] = cache_to_device(tt_pkv["host"], xla)
            # Consistent self-conditioning (zeros + mask=False on step 1) -> one TT graph.
            bs, canvas = kw["decoder_input_ids"].shape
            if kw.get("self_conditioning_logits") is None:
                kw["self_conditioning_logits"] = torch.zeros(
                    bs, canvas, vocab_size, dtype=self.cpu_model.dtype
                )
                scm = torch.zeros(bs, dtype=torch.bool)
            else:
                scm = torch.ones(bs, dtype=torch.bool)
            logits = stage["dec_tt"](
                to_device(kw["decoder_input_ids"], xla),
                to_device(kw["decoder_position_ids"], xla),
                to_device(kw["self_conditioning_logits"], xla),
                to_device(kw["decoder_attention_mask"], xla),
                to_device(scm, xla),
                tt_pkv["pkv"],
            )
            return SimpleNamespace(logits=logits.to("cpu"))  # drive with the TT output

        return encoder_forward, decoder_forward

    def generate(
        self, prompt: str = PROMPT, max_new_tokens: int = None, seed: int = None
    ) -> str:
        """Generate text with both encoder and decoder on TT (staged); return decoded output."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        seed = self.config.seed if seed is None else seed

        inputs = self.loader.load_inputs(dtype_override=torch.bfloat16, prompt=prompt)
        # generate()'s extra inputs (e.g. mm_token_type_ids), minus decoder_input_ids.
        extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "decoder_input_ids")
        }
        vocab_size = self.cpu_model.config.text_config.vocab_size
        encoder_forward, decoder_forward = self._staged_forwards(vocab_size)

        torch.manual_seed(seed)
        output = manual_generate(
            self.cpu_model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            encoder_forward=encoder_forward,
            decoder_forward=decoder_forward,
            **extra_kwargs,
        )
        return self.loader.processor.decode(output[0], skip_special_tokens=True)
