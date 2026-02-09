# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from collections import deque
from torch import Tensor
from lerobot.policies.pi0 import PI0Policy


@torch.no_grad()
def preprocess_for_sampling(self, batch: dict[str, Tensor]):
    """
    Preprocess a batch for action sampling during inference.

    This method extracts and prepares all inputs required by the
    inference-time `forward` method, replicating the preprocessing
    logic originally used inside `select_action`.

    Args:
        batch (dict[str, Tensor]): Dictionary containing environment observations.

    Returns:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks indicating valid pixels in images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        state (Tensor): State vector including proprioception / joint positions.
    """
    # Images and corresponding masks
    images, img_masks = self._preprocess_images(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    state = self.prepare_state(batch)

    return images, img_masks, lang_tokens, lang_masks, state


@torch.no_grad()
def forward(
    self,
    images: Tensor,
    img_masks: Tensor,
    lang_tokens: Tensor,
    lang_masks: Tensor,
    state: Tensor,
    **kwargs
) -> Tensor:

    """
    Forward pass replicating `select_action` behavior using preprocessed inputs.

    This method overrides the default training-related `forward` of PI0Policy.
    It handles `n_action_steps > 1` via an internal action queue:
    if the queue is empty, it samples a chunk of actions from the model,
    truncates them to the configured action dimension, and populates the queue.
    Each call returns the next action in the queue, effectively mimicking
    `select_action` for inference.

    Args:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks for images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        state (Tensor): State vector including proprioception / joint positions.
        **kwargs: Additional keyword arguments passed to `sample_actions`.

    Returns:
        Tensor: A single action from the model (shape: [batch_size, action_dim]).
    """
    if not hasattr(self, "_action_queue"):
        self._action_queue = deque()

    if len(self._action_queue) == 0:
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, **kwargs
        )

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = actions[:, : self.config.n_action_steps]

        self._action_queue.extend(actions.transpose(0, 1))

    return self._action_queue.popleft()


PI0Policy.preprocess_for_sampling = preprocess_for_sampling
PI0Policy.forward = forward
