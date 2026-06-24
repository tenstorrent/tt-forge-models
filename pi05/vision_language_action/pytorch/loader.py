# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi0.5 (pi05) vision-language-action model loader implementation for action prediction.

pi05 is a flow-matching VLA policy from Physical Intelligence (LeRobot port). It pairs a
PaliGemma (gemma_2b) vision-language backbone with a smaller Gemma (gemma_300m) action
expert and predicts continuous action chunks via an iterative flow-matching denoising loop.
See: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ....base import ForgeModel


def _patch_torch_cumsum_bool_to_int64() -> None:
    """torch.compile/XLA can keep bool dtype through cumsum; subtracting 1 then fails.

    Eager PyTorch promotes bool cumsum to int64. Casting bool inputs here matches that
    behavior and fixes the lerobot ``torch.cumsum(pad_masks, dim=1) - 1`` pattern used to
    build attention position ids.
    """
    if getattr(torch, "_tt_xla_cumsum_bool_patch_applied", False):
        return
    _orig_cumsum = torch.cumsum

    def cumsum(input, *args, **kwargs):  # noqa: A002
        if isinstance(input, torch.Tensor) and input.dtype == torch.bool:
            input = input.to(dtype=torch.int64)
        return _orig_cumsum(input, *args, **kwargs)

    torch.cumsum = cumsum  # type: ignore[method-assign]
    torch._tt_xla_cumsum_bool_patch_applied = True  # type: ignore[attr-defined]


from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _materialize_meta_buffers(model) -> None:
    """Recompute the small non-persistent buffers that meta-device construction leaves empty.

    pi05 builds on the meta device to avoid a ~30 GB host-RAM peak (see ``_load_policy_low_mem``),
    which leaves a handful of deterministic buffers unmaterialized: the SigLIP vision-tower
    ``position_ids`` (``arange``) and the Gemma rotary ``inv_freq``/``original_inv_freq``. These
    are absent from the checkpoint, so recompute them from each module's own config.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for _, module in model.named_modules():
        inv_freq = getattr(module, "inv_freq", None)
        if (
            isinstance(inv_freq, torch.Tensor)
            and inv_freq.is_meta
            and hasattr(module, "rope_type")
            and hasattr(module, "config")
        ):
            if module.rope_type == "default" and hasattr(
                module, "compute_default_rope_parameters"
            ):
                new_inv_freq, scaling = module.compute_default_rope_parameters(
                    module.config, device="cpu"
                )
            else:
                new_inv_freq, scaling = ROPE_INIT_FUNCTIONS[module.rope_type](
                    module.config, device="cpu"
                )
            module.register_buffer("inv_freq", new_inv_freq, persistent=False)
            module.register_buffer(
                "original_inv_freq", new_inv_freq.clone(), persistent=False
            )
            module.attention_scaling = scaling
        position_ids = getattr(module, "position_ids", None)
        if isinstance(position_ids, torch.Tensor) and position_ids.is_meta:
            n = position_ids.shape[-1]
            module.register_buffer(
                "position_ids", torch.arange(n).expand((1, -1)), persistent=False
            )


def _load_policy_low_mem(model_id: str):
    """Load PI05Policy with a low host-RAM peak.

    The stock ``PI05Policy.from_pretrained`` first allocates the full ~16.6 GB float32 model
    (with random init) and then loads an equally large state dict, peaking around ~30 GB and
    OOM-killing the runner host. Instead, construct the model on the meta device (no weight
    storage), materialize the few non-persistent buffers, then load the checkpoint with
    ``assign=True`` so the loaded tensors become the parameters directly. Peak stays near the
    size of one weight copy (~5-15 GB). The same key remapping the stock loader applies
    (``_fix_pytorch_state_dict_keys`` + ``model.`` prefixing) is reused verbatim.
    """
    import gc

    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(model_id)
    # Build on the meta device. The constructor calls ``self.model.to(config.device)``; setting
    # device to "meta" keeps that a no-op instead of crashing on meta tensors.
    config.device = "meta"
    with init_empty_weights():
        model = PI05Policy(config)
    _materialize_meta_buffers(model)

    weights_file = cached_file(model_id, "model.safetensors")
    state_dict = load_file(weights_file)
    fixed = model._fix_pytorch_state_dict_keys(state_dict, model.config)
    remapped = {
        (k if k.startswith("model.") else f"model.{k}"): v for k, v in fixed.items()
    }
    del state_dict, fixed
    gc.collect()
    missing, unexpected = model.load_state_dict(remapped, strict=False, assign=True)
    del remapped
    gc.collect()
    if missing:
        raise RuntimeError(
            f"pi05 checkpoint missing {len(missing)} parameters (e.g. {missing[:3]})"
        )
    return model


def _patch_deterministic_noise(policy, seed: int = 0) -> None:
    """Make pi05's flow-matching start from a fixed initial noise.

    pi05 is a flow-matching policy: ``sample_actions`` denoises from ``x_T = noise`` where
    ``noise`` defaults to ``sample_noise`` (an unseeded ``torch.normal``). The model is correct
    on device, but the framework's golden CPU run and the device run each draw *different*
    noise, so their action chunks are two equally-valid-but-unrelated samples and PCC collapses
    to ~0. Replace ``sample_noise`` with a deterministic constant (one fixed-seed CPU draw,
    moved to the requested device) so both runs share the same ``x_T`` and are comparable. The
    constant is precomputed once, so nothing stochastic enters the compiled graph.
    """
    inner = policy.model  # PI05Pytorch, owns sample_noise / sample_actions
    config = policy.config
    actions_shape = (1, config.chunk_size, config.max_action_dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fixed_noise = torch.normal(
        mean=0.0, std=1.0, size=actions_shape, dtype=torch.float32, generator=generator
    )
    inner._tt_fixed_noise = fixed_noise

    def deterministic_sample_noise(self, shape, device):
        noise = self._tt_fixed_noise
        if tuple(shape) != tuple(noise.shape):
            # Batch sizes other than 1: tile the single fixed draw along the batch dim so the
            # result is still deterministic for the requested shape.
            if len(shape) == 3 and shape[1:] == noise.shape[1:]:
                noise = noise.expand(shape[0], -1, -1).contiguous()
            else:  # Unexpected shape; fall back to a fresh fixed-seed draw.
                g = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.normal(
                    mean=0.0, std=1.0, size=tuple(shape), dtype=torch.float32, generator=g
                )
        return noise.to(device=device, dtype=torch.float32)

    inner.sample_noise = types.MethodType(deterministic_sample_noise, inner)


def _patch_preprocess_images_for_dynamo(policy) -> None:
    """Make ``PI05Policy._preprocess_images`` traceable by torch.compile / dynamo.

    The method starts with ``device = next(self.parameters()).device``. Tracing that
    ``parameters()`` generator crashes torch 2.10's dynamo with an internal
    ``NameError: cannot access free variable 'named_children'`` before the graph ever reaches
    the device. The model device is the same as the device the input images already live on
    (the runner places both together), so rebuild the method from its installed source with
    that one line rewritten to read the device from the first present image tensor — a plain
    tensor-attribute access dynamo handles fine. Everything else in the body is kept verbatim
    so the patch tracks whatever lerobot version is installed.
    """
    import textwrap

    from lerobot.policies.pi05 import modeling_pi05

    cls = type(policy)
    original = cls._preprocess_images
    src = textwrap.dedent(inspect.getsource(original))
    needle = "device = next(self.parameters()).device"
    if needle not in src:
        # Source changed upstream; leave the original in place rather than guess.
        return
    replacement = (
        "device = batch[\n"
        "            next(k for k in self.config.image_features if k in batch)\n"
        "        ].device"
    )
    src = src.replace(needle, replacement)
    namespace: dict = {}
    exec(compile(src, modeling_pi05.__file__, "exec"), modeling_pi05.__dict__, namespace)
    patched = namespace["_preprocess_images"]
    policy._preprocess_images = types.MethodType(patched, policy)


def _alias_relative_actions_processor() -> None:
    """Register the ``relative_actions_processor`` step name expected by the pi05 checkpoint.

    The published ``lerobot/pi05_base`` ``policy_preprocessor.json`` references a step named
    ``relative_actions_processor``. In lerobot 0.5.1 the same ``RelativeActionsProcessorStep``
    class is registered under the legacy name ``delta_actions_processor`` instead, so pipeline
    deserialization fails with a KeyError. Add an alias mapping the checkpoint's name to the
    installed class so the preprocessor loads unchanged.
    """
    from lerobot.processor.pipeline import ProcessorStepRegistry

    if "relative_actions_processor" in ProcessorStepRegistry._registry:
        return
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep,
    )

    ProcessorStepRegistry._registry[
        "relative_actions_processor"
    ] = RelativeActionsProcessorStep


def _setup_policies_namespace() -> None:
    """Register lerobot.policies in sys.modules so subpackage imports work when this loader
    is imported outside the normal lerobot package context (e.g. via tt-forge-models dynamic
    import). Without this, 'from lerobot.policies.pi05...' can fail with import errors.
    """
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return
    policies_path = Path(spec.origin).resolve().parent / "policies"
    if not policies_path.exists():
        return
    if "lerobot.policies" in sys.modules:
        return
    policies_module = types.ModuleType("lerobot.policies")
    policies_module.__path__ = [str(policies_path)]
    sys.modules["lerobot.policies"] = policies_module


class Pi05InferenceWrapper(torch.nn.Module):
    """Wraps PI05Policy to use predict_action_chunk (inference) instead of forward (training).

    PI05Policy.forward() computes training loss; for inference we use predict_action_chunk,
    which runs the flow-matching denoising loop and returns an action chunk.
    See: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py
    """

    def __init__(self, policy: "PI05Policy"):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> torch.Tensor:
        """Run inference via predict_action_chunk. Returns action tensor (B, n_steps, action_dim)."""
        return self.policy.predict_action_chunk(batch)


class ModelVariant(StrEnum):
    """Available pi05 model variants."""

    PI05_BASE = "pi05_base"


class ModelLoader(ForgeModel):
    """pi05 model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.PI05_BASE: ModelConfig(
            pretrained_model_name="lerobot/pi05_base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PI05_BASE

    sample_task = "pick the red block"
    robot_type = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Pi05",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self, device: torch.device):
        _patch_torch_cumsum_bool_to_int64()
        _setup_policies_namespace()
        import lerobot.policies.pi05.processor_pi05  # noqa: F401

        _alias_relative_actions_processor()
        from lerobot.processor import PolicyProcessorPipeline

        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        _patch_torch_cumsum_bool_to_int64()
        _setup_policies_namespace()

        # pi05: always use float32. torch.compile/inductor on CPU has dtype consistency
        # issues with bfloat16 (mat1/mat2 mismatch). Pretrained weights are float32
        # (config dtype="float32"); preprocess produces float32 - explicit float32 keeps
        # model and inputs aligned.
        #
        # The model is ~16.6 GB in float32; the stock from_pretrained peaks ~30 GB host RAM
        # (model + state dict co-resident) and OOM-kills the runner. Use a meta-device load
        # that keeps the peak near a single weight copy. See ``_load_policy_low_mem``.
        model = _load_policy_low_mem(self._variant_config.pretrained_model_name)
        model.to(device)
        model = model.to(dtype=torch.float32)
        model.eval()
        # Rewrite the one ``next(self.parameters())`` device lookup that crashes dynamo.
        _patch_preprocess_images_for_dynamo(model)
        # Flow-matching: start denoising from a fixed noise so CPU golden and device runs match.
        _patch_deterministic_noise(model)
        self.config = model.config
        if self.preprocess is None:
            self._load_processors(torch.device(device))
        # Wrap so model(**inputs) runs predict_action_chunk (inference) not forward (training).
        return Pi05InferenceWrapper(model)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        _setup_policies_namespace()
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.utils import prepare_observation_for_inference

        if self.config is None:
            self.config = PI05Config.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self.preprocess is None:
            self._load_processors(torch.device(device))

        dummy_observation = build_dummy_observation(self.config.input_features or {})
        obs_frame = prepare_observation_for_inference(
            observation=dummy_observation,
            device=torch.device(device),
            task=self.sample_task,
            robot_type=self.robot_type,
        )

        inputs = self.preprocess(obs_frame)

        if batch_size > 1:
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        return {"batch": inputs}

    def unpack_forward_output(self, fwd_output):
        """predict_action_chunk returns action tensor (B, n_steps, action_dim) directly."""
        return fwd_output


def build_dummy_observation(input_features: dict) -> dict[str, np.ndarray]:
    from lerobot.configs.types import FeatureType

    observation: dict[str, np.ndarray] = {}
    for key, feature in input_features.items():
        if not key.startswith("observation."):
            continue
        if feature.type == FeatureType.VISUAL:
            channels, height, width = feature.shape
            observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
        elif feature.type in (FeatureType.STATE, FeatureType.ENV):
            observation[key] = np.zeros(feature.shape, dtype=np.float32)
    return observation
