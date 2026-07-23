# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llasa-8B (HKUSTAudio/Llasa-8B) — end-to-end text-to-speech generation pipeline.

Llasa-8B is a Llama-3.1-8B TTS fine-tune (vocab 193800: text + XCodec2 speech
tokens). The 16 GB weights do not fit one Wormhole chip, so the LM runs
**tensor-parallel across a multi-chip mesh** (Megatron-1D over a
``(None, "model")`` mesh — the shard spec the model-runner
``tensor_parallel-inference`` test validates).

``generate()`` runs the FULL autoregressive decode loop **on device**: a fixed
``[1, WINDOW]`` LM forward that compiles once and reruns per step, sampling each
next speech token on the host. It returns the emitted speech-token *codes*
(``token_id - <|s_0|>``); turning those into a waveform is XCodec2's job, which
lives in a separate torch-2.5 env and is deliberately NOT part of this graph.

Three things make eager TP decode work on TT (see the model-runner path):

* ``CONVERT_SHLO_TO_SHARDY=1`` before SPMD init, so tt-mlir gets Shardy
  annotations (else "GSPMD presharded argument missing @Sharding custom call").
* No ``attention_mask`` into the forward — HF ``_ignore_causal_mask_sdpa`` does a
  data-dependent ``.all()`` that materializes a sharded tensor mid-graph
  ("Can't get a single buffer from host storage distributed over mesh"). Plain
  causal masking is correct for right-padded static-window decode: we read
  logits at real position ``cur-1``, which never attends to padded positions.
* ``lm_head`` is kept replicated (dropped from the shard spec) so the output
  logits are not vocab-sharded and the per-step host copy is a single buffer.

Per-generate timing is recorded into ``self._perf``::

    _perf = {
        "components": {},                    # (none — single monolithic LM)
        "steps": [seconds, ...],             # per-token decode times
        "step_metric_name": "decode_step",
        "total": seconds,                    # full generate() wall time
    }
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .loader import ModelLoader, ModelVariant


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


class LlasaConfig:
    """Configuration for the Llasa-8B TTS generation pipeline."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        window: int = 464,
        max_new_tokens: int = 400,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        self.dtype = dtype
        self.window = window
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


class LlasaPipeline:
    """Llasa-8B pipeline: the LM runs tensor-parallel on the mesh; sampling on host."""

    def __init__(self, config: Optional[LlasaConfig] = None):
        self.config = config or LlasaConfig()
        self.loader = None
        self.model = None
        self.tokenizer = None
        self.mesh = None
        self._device = None
        self._speech_end = None
        self._speech_base = None
        self._perf = None

    def setup(self):
        """Build the mesh, load Llasa-8B, shard (lm_head replicated) on the device."""
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()

        self.loader = ModelLoader(ModelVariant.LLASA_8B)
        self.model = self.loader.load_model(dtype_override=self.config.dtype)
        self.tokenizer = self.loader.tokenizer or self.loader._load_tokenizer()

        # <|SPEECH_GENERATION_END|> stops decode; <|s_k|> == speech_base + k.
        self._speech_end = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )
        self._speech_base = self.tokenizer.convert_tokens_to_ids("<|s_0|>")

        mesh_shape, mesh_names = self.loader.get_mesh_config(num_devices)
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
        xs.set_global_mesh(self.mesh)
        logger.info(f"[Llasa] mesh {mesh_shape} {mesh_names} over {num_devices} chips")

        device = torch_xla.device(0)
        self.model = self.model.to(device)
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        self._device = device

        # Build shard spec AFTER moving to device (keys are on-device params),
        # then keep lm_head replicated so output logits are not vocab-sharded.
        shard_specs = self.loader.load_shard_spec(self.model)
        shard_specs.pop(self.model.lm_head.weight, None)
        for param, spec in shard_specs.items():
            xs.mark_sharding(param, self.mesh, spec)
        logger.info(
            f"[Llasa] sharded {len(shard_specs)} params (Megatron-1D, lm_head replicated)"
        )

    def _build_prompt_ids(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.apply_chat_template(
            self.loader._build_tts_prompt(text),
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )
        return encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]

    def generate(
        self,
        text: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> List[int]:
        """Autoregressively generate speech-token codes on device.

        Returns the list of XCodec2 codes (``token_id - <|s_0|>``) up to the
        model's ``<|SPEECH_GENERATION_END|>`` or ``max_new_tokens``.
        """
        assert self.model is not None, "Call setup() before generate()."
        text = text if text is not None else self.loader.sample_text
        n_new = max_new_tokens or self.config.max_new_tokens
        window = self.config.window

        prompt_ids = self._build_prompt_ids(text)
        prompt_len = prompt_ids.shape[1]
        assert prompt_len + n_new <= window, "window too small for prompt + max_new_tokens"

        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "decode_step",
            "total": None,
        }

        # Static right-padded window: the [1, window] forward keeps a fixed shape
        # so it compiles once and reruns per token.
        input_ids = torch.zeros((1, window), dtype=torch.long)
        input_ids[0, :prompt_len] = prompt_ids[0]
        cur = prompt_len

        gen: List[int] = []
        g = torch.Generator().manual_seed(seed if seed is not None else 0)
        code_max = self._speech_base + 65535

        t_total = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_new):
                t0 = time.perf_counter()
                out = self.model(input_ids=input_ids.to(self._device))
                logits_cpu = out.logits.to("cpu").float()  # fixed shape -> stable graph
                torch_xla.sync()
                row = logits_cpu[0, cur - 1, :] / self.config.temperature
                probs = torch.softmax(row, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                sp[(sp.cumsum(-1) - sp) > self.config.top_p] = 0.0
                sp /= sp.sum()
                nxt = si[torch.multinomial(sp, 1, generator=g)].item()
                self._perf["steps"].append(time.perf_counter() - t0)
                if nxt == self._speech_end or cur >= window:
                    break
                gen.append(nxt)
                input_ids[0, cur] = nxt
                cur += 1
        self._perf["total"] = time.perf_counter() - t_total

        return [t - self._speech_base for t in gen if self._speech_base <= t <= code_max]
