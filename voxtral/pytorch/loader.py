# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral model loader implementation.

Voxtral-Mini-3B-2507 is Mistral's audio+text -> text model: a Whisper-style
audio encoder + multi-modal projector feeding a 30-layer Ministral-3B causal
language model (vocab 131072).

The HF ``VoxtralForConditionalGeneration.forward`` merges the audio embeddings
into the text-embedding sequence with ``inputs_embeds.masked_scatter(...)``. That
op is data-dependent and lowers to a dynamic-shape ``set_dimension_size`` which
the TT/Shardy compiler cannot handle. To keep the on-device graph static we
precompute ``inputs_embeds`` on the host (text embedding + audio tower + the
masked_scatter merge) inside ``load_inputs`` and feed those embeddings straight
to the model. With ``inputs_embeds`` supplied and ``input_features`` omitted,
``forward`` skips the dynamic merge and runs only the language model — a static
graph suitable for compilation.
"""

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from typing import Optional


# Default audio prompt used to build the multimodal inputs. Two short clips so
# the bring-up exercises multi-chunk audio handling, plus a text question.
_DEFAULT_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
            },
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
            },
            {
                "type": "text",
                "text": "What sport and what nursery rhyme are referenced?",
            },
        ],
    }
]


def _patch_chat_template_guard():
    """Work around a transformers 5.5.x bug.

    ``VoxtralProcessor.apply_chat_template`` calls ``_get_template_variables(
    chat_template)`` unconditionally, but Voxtral uses the mistral_common backend
    and has no Jinja ``chat_template`` (it is ``None``), which makes jinja raise
    "Can't compile non template nodes". Guard the ``None`` case idempotently.
    """
    import transformers.models.voxtral.processing_voxtral as _vox

    if getattr(_vox._get_template_variables, "_tt_patched", False):
        return

    _orig = _vox._get_template_variables

    def _safe_get_template_variables(chat_template):
        if chat_template is None:
            return frozenset()
        return _orig(chat_template)

    _safe_get_template_variables._tt_patched = True
    _vox._get_template_variables = _safe_get_template_variables


class ModelVariant(StrEnum):
    """Available Voxtral model variants."""

    VOXTRAL_MINI_3B = "Voxtral-Mini-3B-2507"


class ModelLoader(ForgeModel):
    """Voxtral model loader implementation."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_MINI_3B: ModelConfig(
            pretrained_model_name="mistralai/Voxtral-Mini-3B-2507",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_MINI_3B

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model_name = self._variant_config.pretrained_model_name
        self.model = None
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Voxtral",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        if self.processor is None:
            _patch_chat_template_guard()
            self.processor = AutoProcessor.from_pretrained(self._model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Voxtral model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. Bring-up uses torch.bfloat16.

        Returns:
            VoxtralForConditionalGeneration: The model instance.
        """
        _patch_chat_template_guard()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VoxtralForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    @torch.no_grad()
    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Build static-shape model inputs (precomputed ``inputs_embeds``).

        The audio tower, text embedding and audio/text merge all run here on the
        host, producing the merged ``inputs_embeds`` so the on-device graph is
        just the language model — see the module docstring. Extra runner kwargs
        (e.g. ``run_phase``, ``seq_len``) are accepted and ignored.

        Args:
            dtype_override: Optional torch.dtype for the embeddings (e.g. bfloat16).
            batch_size: Batch size (replicated along dim 0). ``None`` -> 1.

        Returns:
            dict: ``{"inputs_embeds": Tensor, "attention_mask": Tensor}``.
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)
        processor = self._load_processor()

        batch_size = batch_size or 1
        target_dtype = dtype_override or next(self.model.parameters()).dtype
        # Precompute on the host so the dynamic merge never reaches the device.
        device = next(self.model.parameters()).device

        inputs = processor.apply_chat_template(_DEFAULT_CONVERSATION).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].cpu()
        input_features = inputs["input_features"].to(target_dtype)

        # Text embeddings + audio tower output merged at the audio placeholder
        # positions. Use a static index_copy (not masked_scatter) so this is
        # safe even if the model is already on a device with no dynamic-shape
        # support; it is numerically identical to masked_scatter.
        embeds = self.model.get_input_embeddings()(input_ids).to(target_dtype)
        audio_embeds = self.model.get_audio_features(
            input_features, return_dict=True
        ).pooler_output.to(target_dtype)
        audio_positions = (input_ids[0] == self.model.config.audio_token_id).nonzero(
            as_tuple=True
        )[0]
        inputs_embeds = embeds.index_copy(
            1, audio_positions, audio_embeds.unsqueeze(0)
        ).cpu()

        if batch_size != 1:
            inputs_embeds = inputs_embeds.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    def decode_output(self, outputs, dtype_override=None):
        """Greedy-decode the next token from logits for a human-readable check."""
        processor = self._load_processor()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        next_id = logits[:, -1, :].argmax(dim=-1)
        return processor.tokenizer.decode(next_id.tolist())

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    @staticmethod
    def _get_language_model(model):
        """Get the language_model sub-module, handling nested model wrapping."""
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Cannot find language_model on the model")

    def load_shard_spec(self, model):
        """Megatron-style tensor-parallel shard spec for the language model.

        Used for multichip bring-up when the model is weight-bound on a single
        device. Column-parallel on q/k/v/gate/up, row-parallel on o/down.
        """
        shard_specs = {}
        language_model = self._get_language_model(model)
        for layer in language_model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs

    # ------------------------------------------------------------------ #
    # Audio tower (Whisper-style encoder + projector) on TT.
    #
    # The LM runs on TT; the audio tower still ran on CPU (host-precomputed
    # inputs_embeds). ``encode_audio`` runs it on device: the Whisper encoder
    # (audio_tower) on TT via torch.compile(backend="tt"); the reshape +
    # multi_modal_projector on CPU. Build ``input_features`` with the processor's
    # WhisperFeatureExtractor (no chat-template/tokenizer needed for the audio
    # path). See tt-xla #5537.
    # ------------------------------------------------------------------ #

    def encode_audio(self, input_features, backbone_on_tt=False):
        """Mel ``input_features`` [B, n_mels, frames] -> audio embeds [N, hidden].

        With ``backbone_on_tt=True`` the Whisper audio tower runs on TT
        (optimization_level=1); the reshape + projector stay on CPU.
        """
        model = self.model if self.model is not None else self.load_model()
        audio_tower = model.audio_tower
        projector = model.multi_modal_projector
        inter = model.config.audio_config.intermediate_size

        def _project(last_hidden):
            return projector(last_hidden.reshape(-1, inter))

        with torch.no_grad():
            if not backbone_on_tt:
                lh = audio_tower(input_features, return_dict=True).last_hidden_state
                return _project(lh)

            import torch_xla
            import torch_xla.core.xla_model as xm

            torch_xla.set_custom_compile_options({"optimization_level": 1})
            dev = xm.xla_device()
            at = audio_tower.to(dtype=torch.bfloat16).to(dev)
            compiled = torch.compile(
                lambda f: at(f, return_dict=True).last_hidden_state, backend="tt"
            )
            lh = compiled(input_features.to(dtype=torch.bfloat16).to(dev))
            lh = lh.to("cpu").to(torch.float32)
            return _project(lh)  # reshape + projector on CPU
