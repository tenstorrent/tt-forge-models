# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import queue
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import increment_steps

# ======================================================
# Queue wrapper to capture latents
# ======================================================


class LatentTapQueue(queue.Queue):
    def __init__(self, collector):
        super().__init__()
        self.collector = collector

    def put(self, item, *args, **kwargs):
        if torch.is_tensor(item):
            self.collector.append(item.detach().cpu())
        return super().put(item, *args, **kwargs)


@torch.no_grad()
def forward(
    self,
    model_state: dict,
    text_to_generate: str,
    frames_after_eos=None,
    copy_state=True,
):
    """
    Runs full Pocket-TTS inference.

    - generates latents
    - decodes audio (unchanged)
    - stores decoded audio internally

    Returns:
        latents: [1, T, ldim]
    """

    collected_latents = []

    # store final audio here
    self._cached_audio = None

    original_queue = queue.Queue

    def queue_factory(*args, **kwargs):
        if not hasattr(self, "_latent_queue_created"):
            self._latent_queue_created = True
            return LatentTapQueue(collected_latents)
        return original_queue(*args, **kwargs)

    queue.Queue = queue_factory

    try:
        audio = self.generate_audio(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
        )

        # cache decoded output
        self._cached_audio = audio

    finally:
        queue.Queue = original_queue
        if hasattr(self, "_latent_queue_created"):
            del self._latent_queue_created

    latents = torch.cat(collected_latents, dim=1)

    return latents


def post_process(self):
    """
    Returns decoded audio that was already produced
    during forward().
    """

    if self._cached_audio is None:
        raise RuntimeError("post_process() called before forward().")

    return self._cached_audio


def _run_flow_lm_and_increment_step(
    self,
    model_state: dict,
    text_tokens: torch.Tensor | None = None,
    backbone_input_latents: torch.Tensor | None = None,
    audio_conditioning: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First one is the backbone output, second one is the audio decoding output."""
    if text_tokens is not None:
        device = text_tokens.device
    elif backbone_input_latents is not None:
        device = backbone_input_latents.device
    elif audio_conditioning is not None:
        device = audio_conditioning.device
    else:
        device = self.flow_lm.device
    if text_tokens is None:
        text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=device)
    if backbone_input_latents is None:
        backbone_input_latents = torch.empty(
            (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=device
        )
    if audio_conditioning is None:
        audio_conditioning = torch.empty(
            (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=device
        )
    output = self._run_flow_lm(
        text_tokens=text_tokens,
        backbone_input_latents=backbone_input_latents,
        model_state=model_state,
        audio_conditioning=audio_conditioning,
    )
    increment_by = (
        text_tokens.shape[1]
        + backbone_input_latents.shape[1]
        + audio_conditioning.shape[1]
    )
    increment_steps(self.flow_lm, model_state, increment=increment_by)
    return output


TTSModel._run_flow_lm_and_increment_step = _run_flow_lm_and_increment_step
TTSModel.forward = forward
TTSModel.post_process = post_process
