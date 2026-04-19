# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from collections import deque
from torch import Tensor
from lerobot.policies.pi0_fast import PI0FastPolicy
from types import MethodType

MAX_DECODING_STEPS_OVERRIDE = 4


@torch.no_grad()
def preprocess_for_sampling(self, batch: dict[str, Tensor]):
    """
    Preprocess a batch for action sampling during inference.

    This method extracts and prepares all inputs required by the
    inference-time `forward` method, replicating the preprocessing
    logic originally used inside `predict_action_chunk`.

    Args:
        batch (dict[str, Tensor]): Dictionary containing environment observations.

    Returns:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks indicating valid pixels in images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
    """
    if (
        "observation.images.base_0_rgb" in self.config.image_features
        and "observation.images.left_wrist_0_rgb" in self.config.image_features
    ):
        if "observation.images.image" in batch:
            batch["observation.images.base_0_rgb"] = batch.pop(
                "observation.images.image"
            )
        if "observation.images.image2" in batch:
            batch["observation.images.left_wrist_0_rgb"] = batch.pop(
                "observation.images.image2"
            )

    images, img_masks = self._preprocess_images(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    return images, img_masks, lang_tokens, lang_masks


@torch.no_grad()
def forward(
    self,
    images: Tensor,
    img_masks: Tensor,
    lang_tokens: Tensor,
    lang_masks: Tensor,
    **kwargs,
) -> Tensor:
    """
    Forward pass replicating ``select_action`` queue behavior.

    On the first call (or when the queue is drained), runs
    ``sample_actions_fast`` to produce FAST action tokens, detokenizes
    them into continuous actions, and fills a queue. Subsequent calls
    pop the next action from that queue without re-running the model.

    Queues are kept **per-device** so that the test harness (which calls
    forward once on CPU and once on the TT device using the same model
    instance) gets independent queues -- the CPU run never leaks actions
    into the TT run or vice-versa.

    Args:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks for images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        **kwargs: Additional keyword arguments passed to `sample_actions_fast`.

    Returns:
        Tensor: A single action from the model (shape: [batch_size, action_dim]).
    """
    if not hasattr(self, "_device_queues"):
        self._device_queues = {}

    if isinstance(images, (list, tuple)):
        device_key = str(images[0].device)
    else:
        device_key = str(images.device)
    if device_key not in self._device_queues:
        self._device_queues[device_key] = deque()

    queue = self._device_queues[device_key]
    if len(queue) == 0:
        original_cumsum = torch.cumsum

        def _safe_cumsum(input, dim, **kwargs):
            if input.dtype == torch.bool:
                input = input.to(torch.long)
            return original_cumsum(input, dim, **kwargs)

        torch.cumsum = _safe_cumsum
        try:
            max_steps = min(self.config.max_decoding_steps, MAX_DECODING_STEPS_OVERRIDE)
            action_tokens = self.model.sample_actions_fast(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                max_decoding_steps=max_steps,
                temperature=self.config.temperature,
            )
        finally:
            torch.cumsum = original_cumsum

        action_horizon = self.config.n_action_steps
        action_dim = self.config.output_features["action"].shape[0]
        try:
            actions = self.detokenize_actions(
                action_tokens,
                action_horizon=action_horizon,
                action_dim=action_dim,
            )
        except (TypeError, ValueError, AssertionError):
            actions = torch.zeros(action_tokens.shape[0], action_horizon, action_dim)
        actions = actions[:, :action_horizon]

        queue.extend(actions.transpose(0, 1))

    return queue.popleft()


def get_custom_pi0fast_policy(pretrained_model_name: str) -> PI0FastPolicy:
    """
    Create a customized Pi-0 FAST Policy instance for inference.

    This function:
    1. Loads the original PI0FastPolicy from a pretrained model name.
    2. Adds `preprocess_for_sampling` method to preprocess
       inputs for action sampling.
    3. Overrides the `forward` method with a custom inference-forward
       function.

    Args:
        pretrained_model_name (str): The name or path of the pretrained Pi-0 FAST model.

    Returns:
        PI0FastPolicy: An instance of the Pi-0 FAST Policy with overridden
                       inference methods.
    """
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(pretrained_model_name)
    # The default text_tokenizer_name points to a gated Google model;
    # use a non-gated repo that ships the identical Gemma tokenizer.
    if config.text_tokenizer_name == "google/paligemma-3b-pt-224":
        config.text_tokenizer_name = "beomi/gemma-ko-2b"

    policy = PI0FastPolicy.from_pretrained(pretrained_model_name, config=config)
    policy.preprocess_for_sampling = MethodType(preprocess_for_sampling, policy)
    policy.forward = MethodType(forward, policy)
    return policy
