# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FIBO (briaai/FIBO) loading and preprocessing.

FIBO is BRIA AI's 8B-parameter DiT-based, flow-matching text-to-image model.
It uses ``SmolLM3-3B`` as the text encoder, ``Wan 2.2`` as the VAE, and a
``DimFusion`` conditioning architecture trained on structured JSON captions.
The full pipeline is exposed as ``BriaFiboPipeline`` in upstream diffusers
(git-main; not yet in the 0.37.x line).

The FIBO transformer's exact positional signature evolves with diffusers
revisions, so this loader follows the same pattern as ``bria_2_3``: it loads
the full pipeline, then drives one short ``pipe(prompt=...)`` call with a
monkey-patched ``transformer.forward`` to capture the exact positional
tensors the transformer consumes. This makes the loader robust to schema
drift in upstream diffusers without us pinning to one specific version.

Reference: https://huggingface.co/briaai/FIBO
"""

from typing import Any, Dict, Optional, Tuple

import torch

# Minimal JSON-style prompt — FIBO is trained on structured JSON captions but
# also accepts plain text via VLM expansion. For a single bringup forward we
# use a stub structured prompt; the model's __call__ will tokenize it through
# SmolLM3 the same way regardless of whether the JSON is meaningful or not.
BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)

# Default inference hyperparameters lifted from the FIBO model card's Generate
# example — see https://huggingface.co/briaai/FIBO#generate.
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0


class _ShortCircuit(Exception):
    """Raised inside the patched transformer to abort the pipeline after step 0."""


def load_pipe(pretrained_model_name: str, dtype_override=None):
    """Load the FIBO pipeline.

    Args:
        pretrained_model_name: HuggingFace repo id, e.g. ``"briaai/FIBO"``.
            Note: this repo is gated — accept the bria-fibo license on HF and
            authenticate via ``HF_TOKEN`` before calling this.
        dtype_override: Optional ``torch.dtype`` cast applied to the pipeline.
            ``torch.bfloat16`` matches the model card's reference settings.

    Returns:
        diffusers.DiffusionPipeline: FIBO pipeline on CPU, ``eval()`` mode,
        ``requires_grad`` disabled.
    """
    try:
        from diffusers import BriaFiboPipeline  # type: ignore
    except ImportError:
        # Fall back to the auto-resolver; this works if a newer diffusers is
        # available (model_index.json in briaai/FIBO declares the class).
        from diffusers import DiffusionPipeline as BriaFiboPipeline  # type: ignore

    pipe_kwargs: Dict[str, Any] = {}
    if dtype_override is not None:
        pipe_kwargs["torch_dtype"] = dtype_override

    pipe = BriaFiboPipeline.from_pretrained(pretrained_model_name, **pipe_kwargs)

    pipe.to("cpu")

    for attr in ("text_encoder", "transformer", "vae"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def capture_transformer_inputs(
    pipe,
    prompt: str = BRINGUP_PROMPT,
    *,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = 42,
) -> Dict[str, Any]:
    """Drive ``pipe(prompt=...)`` for one transformer step and capture inputs.

    Monkey-patches the FIBO transformer's ``forward`` to record the exact
    positional args + kwargs it's invoked with, then short-circuits the
    denoising loop. This avoids hardcoding a signature that may change as
    upstream diffusers evolves.

    Returns:
        dict: ``{"args": tuple, "kwargs": dict}`` of the first forward call.
    """
    if not hasattr(pipe, "transformer"):
        raise RuntimeError(
            "FIBO pipeline does not expose a .transformer attribute. "
            "Upstream diffusers may have renamed it; inspect pipe.components."
        )

    capture: Dict[str, Any] = {}
    original_forward = pipe.transformer.forward

    def patched_forward(*args, **kwargs):
        capture["args"] = args
        capture["kwargs"] = kwargs
        out = original_forward(*args, **kwargs)
        capture["output"] = out
        raise _ShortCircuit()

    pipe.transformer.forward = patched_forward
    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            try:
                pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",
                )
            except _ShortCircuit:
                pass
    finally:
        pipe.transformer.forward = original_forward

    if "args" not in capture:
        raise RuntimeError(
            "FIBO pipeline never invoked the patched transformer — "
            "capture is empty. Check the pipeline class and prompt format."
        )
    return capture


def positional_inputs_from_capture(capture: Dict[str, Any]) -> Tuple[Any, ...]:
    """Flatten a captured (args, kwargs) call into a positional tuple.

    The order is ``args`` first (positional in the original call), then the
    kwargs in their iteration order. Non-tensor values are kept as-is so the
    auto-runner can decide how to handle them; the wrapper below normalizes
    common cases (e.g. ``dict`` → kept as keyword on replay).
    """
    flat = list(capture["args"])
    for key, value in capture["kwargs"].items():
        flat.append(value)
    return tuple(flat)


class FiboTransformerWrapper(torch.nn.Module):
    """Wrap the FIBO transformer so it accepts the captured positional inputs.

    The auto-runner (``DynamicTorchModelTester``) calls ``model(*inputs)``
    positionally. The FIBO transformer's native ``forward`` mixes positional
    and keyword arguments; we capture both at construction time and replay
    them with the same call shape.

    On forward, ``inputs`` are split back into ``args + kwargs`` using the
    captured signature, then forwarded to the underlying transformer.
    """

    def __init__(self, transformer: torch.nn.Module, capture: Dict[str, Any]) -> None:
        super().__init__()
        self.transformer = transformer
        self._num_args = len(capture["args"])
        self._kwarg_keys: Tuple[str, ...] = tuple(capture["kwargs"].keys())

    def forward(self, *inputs):
        if len(inputs) != self._num_args + len(self._kwarg_keys):
            raise ValueError(
                f"FiboTransformerWrapper expected "
                f"{self._num_args + len(self._kwarg_keys)} positional inputs "
                f"(got {len(inputs)})."
            )
        args = inputs[: self._num_args]
        kwargs = dict(zip(self._kwarg_keys, inputs[self._num_args :]))
        out = self.transformer(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        return out
