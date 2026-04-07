# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import queue
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.models.flow_lm import FlowLMModel
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


@property
def device(self) -> str:
    """
    Return the full torch.device instead of just the device type.

    This ensures correct device resolution for backends such as XLA
    (e.g., 'xla:0' instead of 'xla').
    """
    return str(next(self.parameters()).device)


TTSModel.forward = forward
TTSModel.post_process = post_process
FlowLMModel.device = device
