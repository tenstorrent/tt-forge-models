# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-Mini-3B-2507 — end-to-end audio+text -> text generation pipeline.

Voxtral is Mistral's audio-understanding model: a Whisper-style audio encoder +
projector feeding a 30-layer Ministral-3B causal LM (vocab 131072). It fits a
single Wormhole chip, so this pipeline runs **single-device** (no mesh/sharding).

The HF forward merges audio embeddings into the text sequence with a
data-dependent ``masked_scatter`` that the compiler can't lower. The loader
already avoids that by precomputing merged ``inputs_embeds`` on the host (audio
tower + text embed + static merge); this pipeline feeds those embeddings to the
LM, so the on-device graph is a plain static-shape language model.

``generate()`` runs the FULL autoregressive decode loop **on device**: a fixed
``[1, WINDOW, hidden]`` LM forward that compiles once and reruns per step,
greedily picking each next token on the host and appending its (host-side)
embedding. Decoding is greedy, so the result is deterministic and matches an
equivalent CPU run token-for-token. Returns the decoded answer string.

No ``attention_mask`` is passed into the forward: HF ``_ignore_causal_mask_sdpa``
does a data-dependent ``.all()`` on it, and plain causal masking is correct for
right-padded static-window decode (we read logits at real position ``cur-1``,
which never attends to padded positions).

Per-generate timing is recorded into ``self._perf``::

    _perf = {
        "components": {"prefill_tokens": L},   # host-precomputed prefill length
        "steps": [seconds, ...],               # per-token decode times
        "step_metric_name": "decode_step",
        "total": seconds,                      # full generate() wall time
    }
"""

import time
from typing import Optional

import torch
import torch_xla
from loguru import logger

from .loader import ModelLoader, ModelVariant


class VoxtralConfig:
    """Configuration for the Voxtral audio->text generation pipeline."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16, max_new_tokens: int = 256):
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens


class VoxtralPipeline:
    """Voxtral pipeline: audio tower + merge on host, the 3B LM greedy-decodes on device."""

    def __init__(self, config: Optional[VoxtralConfig] = None):
        self.config = config or VoxtralConfig()
        self.loader = None
        self.model = None
        self.tokenizer = None
        self._device = None
        self._prefill = None
        self._embed_w = None
        self._eos = None
        self._perf = None

    def setup(self):
        """Load Voxtral, precompute merged inputs_embeds on host, move the LM to device."""
        self.loader = ModelLoader(ModelVariant.VOXTRAL_MINI_3B)
        self.model = self.loader.load_model(dtype_override=self.config.dtype)
        self.tokenizer = self.loader._load_processor().tokenizer

        # Host-side prefill: audio tower + text embed + static merge (model on CPU).
        inputs = self.loader.load_inputs(dtype_override=self.config.dtype)
        self._prefill = inputs["inputs_embeds"]  # [1, L, H] cpu
        L, H = self._prefill.shape[1], self._prefill.shape[2]

        # CPU copy of the token embeddings so generated tokens embed on the host.
        self._embed_w = (
            self.model.get_input_embeddings().weight.detach().to(self.config.dtype).cpu()
        )
        eos = self.model.generation_config.eos_token_id
        self._eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

        self._device = torch_xla.device(0)
        self.model = self.model.to(self._device)
        logger.info(f"[Voxtral] single-device LM; prefill L={L} H={H}")

    @torch.no_grad()
    def generate(self, max_new_tokens: Optional[int] = None) -> str:
        """Greedily decode the answer on device. Returns the decoded text string."""
        assert self.model is not None, "Call setup() before generate()."
        n_new = max_new_tokens or self.config.max_new_tokens
        L, H = self._prefill.shape[1], self._prefill.shape[2]
        window = L + n_new

        self._perf = {
            "components": {"prefill_tokens": L},
            "steps": [],
            "step_metric_name": "decode_step",
            "total": None,
        }

        # Static right-padded window of embeddings -> the forward compiles once.
        buf = torch.zeros((1, window, H), dtype=self.config.dtype)
        buf[0, :L, :] = self._prefill[0]
        cur = L

        gen_ids = []
        t_total = time.perf_counter()
        for _ in range(n_new):
            t0 = time.perf_counter()
            out = self.model(inputs_embeds=buf.to(self._device))
            logits = out.logits.to("cpu").float()  # fixed shape -> stable graph
            torch_xla.sync()
            nxt = int(logits[0, cur - 1, :].argmax())
            self._perf["steps"].append(time.perf_counter() - t0)
            if nxt in self._eos or cur >= window:
                break
            gen_ids.append(nxt)
            buf[0, cur, :] = self._embed_w[nxt]
            cur += 1
        self._perf["total"] = time.perf_counter() - t_total

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
