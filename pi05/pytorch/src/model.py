# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from collections import deque
from torch import Tensor
from lerobot.policies.pi05 import PI05Policy
from types import MethodType


@torch.no_grad()
def preprocess_for_sampling(self, batch: dict[str, Tensor]):
    """
    Preprocess a batch for action sampling during inference.

    This method extracts and prepares all inputs required by the
    inference-time `forward` method, replicating the preprocessing
    logic originally used inside `predict_action_chunk`.

    Unlike pi0, pi05 does not pass a separate proprioceptive state tensor to
    ``sample_actions`` -- the state is discretized into the language prompt by
    the policy preprocessor, so only images and language tokens are returned.

    Args:
        batch (dict[str, Tensor]): Dictionary containing environment observations
            after the policy preprocessor has run (tokens + image tensors).

    Returns:
        images (list[Tensor]): Preprocessed image tensors (one per camera view).
        img_masks (list[Tensor]): Masks indicating valid/padded camera views.
        lang_tokens (Tensor): Tokenized language observation (includes state).
        lang_masks (Tensor): Attention mask for the language tokens.
    """
    # The libero dataset exposes cameras as `image`/`image2`, while pi05's
    # config expects `base_0_rgb`/`left_wrist_0_rgb` (and a third
    # `right_wrist_0_rgb` view). Remap the two available views; the missing
    # third view is auto-padded with -1 by `_preprocess_images`.
    if (
        "observation.images.base_0_rgb" in self.config.image_features
        and "observation.images.left_wrist_0_rgb" in self.config.image_features
        and "observation.images.image" in batch
    ):
        batch["observation.images.base_0_rgb"] = batch.pop(
            "observation.images.image"
        )
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
    images,
    img_masks,
    lang_tokens: Tensor,
    lang_masks: Tensor,
    noise: Tensor = None,
    **kwargs,
) -> Tensor:
    """
    Forward pass replicating ``select_action`` queue behavior for pi05.

    On the first call (or when the queue is drained), runs ``sample_actions``
    to produce a chunk of actions and fills a queue. Subsequent calls pop the
    next action from that queue without re-running the model.

    Queues are kept **per-device** so that the test harness (which calls
    forward once on CPU and once on the TT device using the same model
    instance) gets independent queues -- the CPU run never leaks actions into
    the TT run or vice-versa.

    Args:
        images (list[Tensor]): Preprocessed image tensors.
        img_masks (list[Tensor]): Masks for images.
        lang_tokens (Tensor): Tokenized language observation.
        lang_masks (Tensor): Attention mask for the language tokens.
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


def _cast_input_to_weight_dtype(module, args):
    """Forward pre-hook: cast a Linear's input to its weight dtype.

    pi05's flow-matching denoise path hardcodes float32 in several places
    (the sinusoidal timestep is built as float32, and ``denoise_step`` casts
    the expert output back to float32 before ``action_out_proj``). These
    float32 activations are designed for a float32 model, but to fit on a
    single 12 GB Wormhole chip the weights must be bfloat16. Casting each
    projection's input to its own weight dtype reconciles the two without
    touching the (unchanged) reference modeling code. The cast is a no-op when
    the model already runs in float32 (e.g. the CPU correctness check).
    """
    if not args:
        return args
    x = args[0]
    w_dtype = module.weight.dtype
    if torch.is_tensor(x) and x.is_floating_point() and x.dtype != w_dtype:
        x = x.to(w_dtype)
        return (x,) + args[1:]
    return args


def _install_dtype_reconcile_hooks(policy: PI05Policy) -> None:
    """Cast inputs of the action/time projections to their weight dtype.

    Covers every Linear in the flow-matching suffix path that can receive a
    float32 activation while holding bfloat16 weights.
    """
    model = policy.model
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        module = getattr(model, name, None)
        if module is not None:
            module.register_forward_pre_hook(_cast_input_to_weight_dtype)


def get_custom_pi05_policy(pretrained_model_name: str) -> PI05Policy:
    """
    Create a customized Pi-0.5 Policy instance for inference.

    This function:
    1. Loads the original PI05Policy from a pretrained model name.
    2. Adds `preprocess_for_sampling` to prepare inputs for action sampling.
    3. Overrides `forward` with a custom inference-forward function.

    Args:
        pretrained_model_name (str): The name or path of the pretrained pi05 model.

    Returns:
        PI05Policy: An instance of the pi05 Policy with overridden inference methods.
    """
    policy = PI05Policy.from_pretrained(pretrained_model_name)
    _install_dtype_reconcile_hooks(policy)
    policy.preprocess_for_sampling = MethodType(preprocess_for_sampling, policy)
    policy.forward = MethodType(forward, policy)
    return policy
