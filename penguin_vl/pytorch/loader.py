# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Penguin-VL model loader implementation for multimodal visual question answering.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional

# transformers >= 5.x removed the 'default' entry from ROPE_INIT_FUNCTIONS, but
# Penguin-VL's vision encoder still uses it. Add it back before importing the model.
def _patch_rope_init_functions():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config=None, device=None, seq_len=None, **rope_kwargs
        ):
            if rope_kwargs:
                base = rope_kwargs.get("base", 10000.0)
                dim = rope_kwargs["dim"]
            else:
                base = getattr(config, "rope_theta", 10000.0)
                partial = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                dim = int(head_dim * partial)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


_patch_rope_init_functions()


def _patch_init_weights_for_default_rope():
    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers import initialization as init

    _orig_init_weights = PreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        # VisualRotaryEmbedding (and similar custom modules) use rope_type="default" but
        # don't implement compute_default_rope_parameters required by transformers >= 5.x.
        # Fall back to ROPE_INIT_FUNCTIONS["default"] in that case.
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and hasattr(module, "original_inv_freq")
            and getattr(module, "rope_type", None) == "default"
            and not hasattr(module, "compute_default_rope_parameters")
            and "default" in ROPE_INIT_FUNCTIONS
        ):
            buffer_value, _ = ROPE_INIT_FUNCTIONS["default"](module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
            return
        return _orig_init_weights(self, module)

    PreTrainedModel._init_weights = _patched_init_weights


_patch_init_weights_for_default_rope()

from ...tools.utils import get_file
from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Penguin-VL model variants."""

    PENGUIN_VL_2B = "2B"
    PENGUIN_VL_8B = "8B"


class ModelLoader(ForgeModel):
    """Penguin-VL model loader implementation for multimodal visual question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PENGUIN_VL_2B: ModelConfig(
            pretrained_model_name="tencent/Penguin-VL-2B",
        ),
        ModelVariant.PENGUIN_VL_8B: ModelConfig(
            pretrained_model_name="tencent/Penguin-VL-8B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PENGUIN_VL_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Penguin-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Penguin-VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Penguin-VL model instance.
        """
        import sys
        import inspect

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["trust_remote_code"] = True
        model_kwargs |= kwargs

        try:
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
        except TypeError as e:
            if "_get_arguments_from_pretrained" not in str(e):
                raise
            # Patch model's processor class for compatibility with transformers >= 5.2.0,
            # which passes processor_dict as a positional arg that older model code doesn't accept.
            from transformers.processing_utils import ProcessorMixin

            for module in list(sys.modules.values()):
                cls = getattr(module, "PenguinVLQwen3Processor", None)
                if (
                    cls is not None
                    and inspect.isclass(cls)
                    and issubclass(cls, ProcessorMixin)
                    and "_get_arguments_from_pretrained" in cls.__dict__
                ):
                    _orig = cls._get_arguments_from_pretrained.__func__

                    @classmethod
                    def _patched(c, name, processor_dict=None, **kw):
                        return _orig(c, name, **kw)

                    cls._get_arguments_from_pretrained = _patched
                    break
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )

        # Penguin-VL's _merge_kwargs was written for transformers==4.51.3 which had
        # 'common_kwargs' in ProcessingKwargs.__annotations__. In transformers >= 5.x
        # this was removed. Add dummy entries so the loop in _merge_kwargs doesn't KeyError.
        class _EmptyKwargs:
            __annotations__ = {}

        for _module in list(sys.modules.values()):
            _pkwargs_cls = getattr(_module, "PenguinVLQwen3ProcessorKwargs", None)
            if _pkwargs_cls is not None and inspect.isclass(_pkwargs_cls):
                for _missing in ("common_kwargs",):
                    if _missing not in _pkwargs_cls.__annotations__:
                        _pkwargs_cls.__annotations__[_missing] = _EmptyKwargs
                break

        # Penguin-VL calls from_pretrained inside __init__ for its vision encoder, which is
        # incompatible with the meta device context that transformers >= 5.x always sets up
        # during model initialization. Temporarily disable meta device init to allow this.
        from transformers.modeling_utils import PreTrainedModel

        _orig_get_init_context = PreTrainedModel.get_init_context

        @classmethod
        def _get_init_context_no_meta(cls, dtype, is_quantized, _is_ds_init_called):
            import torch

            contexts = _orig_get_init_context.__func__(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [
                c
                for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        PreTrainedModel.get_init_context = _get_init_context_no_meta
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        except TypeError as e:
            if "pretrained_model_name_or_path" not in str(e):
                raise
            # Penguin-VL's _load_pretrained_model was written for transformers==4.51.3.
            # In transformers >= 5.x, _load_pretrained_model is a @staticmethod that takes
            # (model, state_dict, checkpoint_files, load_config) where load_config is a
            # LoadStateDictConfig object instead of individual keyword arguments.
            # Patch the model class to use the new signature.
            from transformers.modeling_utils import (
                PreTrainedModel as _PTM,
                load_state_dict as _load_sd,
            )

            for _module in list(sys.modules.values()):
                _cls = getattr(_module, "PenguinVLQwen3ForCausalLM", None)
                if (
                    _cls is not None
                    and inspect.isclass(_cls)
                    and issubclass(_cls, _PTM)
                    and "_load_pretrained_model" in _cls.__dict__
                ):

                    @staticmethod
                    def _compat_load_pretrained_model(
                        model, state_dict, checkpoint_files, load_config
                    ):
                        def _remap(sd):
                            prefix = "model.vision_encoder.vision_encoder."
                            if not any(k.startswith(prefix) for k in sd):
                                return sd
                            return {
                                (
                                    k.replace(prefix, "model.vision_encoder.")
                                    if k.startswith(prefix)
                                    else k
                                ): v
                                for k, v in sd.items()
                            }

                        if state_dict is not None:
                            state_dict = _remap(state_dict)
                        elif checkpoint_files is not None:
                            merged = {}
                            files = (
                                checkpoint_files
                                if isinstance(checkpoint_files, list)
                                else [checkpoint_files]
                            )
                            for ckpt in files:
                                merged.update(
                                    _load_sd(
                                        ckpt,
                                        map_location="cpu",
                                        weights_only=load_config.weights_only,
                                    )
                                )
                            state_dict = _remap(merged)
                            checkpoint_files = None
                        return _PTM._load_pretrained_model(
                            model, state_dict, checkpoint_files, load_config
                        )

                    _cls._load_pretrained_model = _compat_load_pretrained_model
                    break
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context

        # Penguin-VL's vision encoder unconditionally uses flash_attn_varlen_func in
        # PenguinVLAttention.forward, but only imports it when flash_attn is available.
        # Inject a PyTorch fallback so the encoder runs on CPU / compile-only environments.
        # This must happen after model loading so that modeling_penguinvl_encoder is in
        # sys.modules (it is imported by modeling_penguinvl_qwen3, not by the processor).
        import torch.nn.functional as _F

        def _flash_attn_varlen_fallback(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0.0,
            causal=False,
            **_kw
        ):
            outputs = []
            for i in range(cu_seqlens_q.shape[0] - 1):
                qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                ks, ke = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
                # shapes: [seq, num_heads, head_dim] → [1, num_heads, seq, head_dim]
                qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
                ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
                vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
                # GQA: expand KV heads to match Q heads
                if ki.shape[1] != qi.shape[1]:
                    repeat_factor = qi.shape[1] // ki.shape[1]
                    ki = ki.repeat_interleave(repeat_factor, dim=1)
                    vi = vi.repeat_interleave(repeat_factor, dim=1)
                out_i = _F.scaled_dot_product_attention(
                    qi, ki, vi, dropout_p=dropout_p, is_causal=causal
                )
                outputs.append(out_i.squeeze(0).transpose(0, 1))
            return torch.cat(outputs, dim=0)

        for _module in list(sys.modules.values()):
            _mod_name = getattr(_module, "__name__", "") or ""
            if "modeling_penguinvl_encoder" in _mod_name and not getattr(
                _module, "flash_attn_varlen_func", None
            ):
                _module.flash_attn_varlen_func = _flash_attn_varlen_fallback
                break

        model.eval()
        self.model = model

        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Penguin-VL model.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )

        image_file = str(
            get_file(
                "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            )
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_file}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]

        inputs = self.processor(
            conversation=conversation,
            return_tensors="pt",
        )

        if self.model is not None:
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs.
            input_length: Optional length of input tokens to slice from output.

        Returns:
            str: Decoded output text.
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.processor.decode(next_token_id, skip_special_tokens=True)
