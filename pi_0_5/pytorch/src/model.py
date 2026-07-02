# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from collections import deque
from torch import Tensor
from lerobot.policies.pi05 import PI05Policy
from types import MethodType


def _cast_linear_input_to_weight_dtype(module, args):
    """Forward pre-hook: cast a Linear's floating-point input to the weight dtype.

    Pi-0.5's flow-matching sampler builds the timestep (and its sinusoidal
    ``time_emb``) with a hard-coded ``dtype=torch.float32`` inside
    ``sample_actions`` / ``embed_suffix``. On the Tenstorrent device the module
    weights are placed in bfloat16, but that internally-created fp32 ``time_emb``
    is *not* downcast (explicit-dtype tensors survive device transfer), so
    ``time_mlp_in(time_emb)`` hits ``mat1 and mat2 must have the same dtype,
    but got Float and BFloat16``. Casting each Linear's float input to its own
    weight dtype makes every matmul self-consistent. On CPU (weights fp32) the
    cast is a no-op, so the golden reference is unchanged.
    """
    if not args:
        return None
    weight_dtype = module.weight.dtype
    x = args[0]
    if isinstance(x, torch.Tensor) and x.dtype.is_floating_point and x.dtype != weight_dtype:
        return (x.to(weight_dtype),) + args[1:]
    return None


def _register_linear_dtype_hooks(model: nn.Module) -> None:
    """Register the input-dtype-matching pre-hook on every nn.Linear."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_pre_hook(_cast_linear_input_to_weight_dtype)


@torch.no_grad()
def preprocess_for_sampling(self, batch: dict[str, Tensor]):
    """
    Preprocess a batch for action sampling during inference.

    This method extracts and prepares all inputs required by the
    inference-time `forward` method, replicating the preprocessing
    logic originally used inside `select_action` / `predict_action_chunk`.

    Unlike Pi-0, Pi-0.5 does **not** take a separate proprioceptive ``state``
    tensor into the model. The state is discretized and folded into the
    language prompt by ``Pi05PrepareStateTokenizerProcessorStep`` during
    preprocessing, so ``sample_actions`` consumes only images and language.

    Args:
        batch (dict[str, Tensor]): Dictionary containing environment observations.

    Returns:
        images (list[Tensor]): Preprocessed image tensors (one per camera).
        img_masks (list[Tensor]): Masks indicating valid cameras.
        lang_tokens (Tensor): Tokenized language observations (task + state).
        lang_masks (Tensor): Attention masks for language tokens.
    """
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
    noise: Tensor = None,
    **kwargs,
) -> Tensor:
    """
    Forward pass replicating ``select_action`` queue behavior.

    On the first call (or when the queue is drained), runs
    ``sample_actions`` to produce a chunk of actions and fills a queue.
    Subsequent calls pop the next action from that queue without
    re-running the model.

    Queues are kept **per-device** so that the test harness (which calls
    forward once on CPU and once on the TT device using the same model
    instance) gets independent queues -- the CPU run never leaks actions
    into the TT run or vice-versa.

    Args:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks for images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        noise (Tensor, optional): Pre-generated noise tensor for deterministic
            flow-matching sampling. When provided, both CPU and device runs use
            the same starting noise, ensuring reproducible PCC comparison.
        **kwargs: Additional keyword arguments passed to `sample_actions`.

    Returns:
        Tensor: A single action from the model (shape: [batch_size, action_dim]).
    """
    if not hasattr(self, "_device_queues"):
        self._device_queues = {}

    device_key = str(lang_tokens.device)
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
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, noise=noise, **kwargs
            )
        finally:
            torch.cumsum = original_cumsum

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = actions[:, : self.config.n_action_steps]

        queue.extend(actions.transpose(0, 1))

    return queue.popleft()


def get_custom_pi05_policy(pretrained_model_name: str) -> PI05Policy:
    """
    Create a customized Pi-0.5 Policy instance for inference.

    This function:
    1. Loads the original PI05Policy from a pretrained model name.
    2. adds `preprocess_for_sampling` method to preprocess
       inputs for action sampling.
    3. Overrides the `forward` method with a custom inference-forward
       function.

    Args:
        pretrained_model_name (str): The name or path of the pretrained Pi-0.5 model.

    Returns:
        PI05Policy: An instance of the Pi-0.5 Policy with overridden
                    inference methods.
    """
    policy = PI05Policy.from_pretrained(pretrained_model_name)
    policy.preprocess_for_sampling = MethodType(preprocess_for_sampling, policy)
    policy.forward = MethodType(forward, policy)
    _register_linear_dtype_hooks(policy)
    return policy
