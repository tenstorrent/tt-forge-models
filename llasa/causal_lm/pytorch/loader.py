# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa (Llama3-8B TTS) model loader implementation.

Llasa-8B is a text-to-speech model that is initialized from Llama 3.1 8B and
fine-tuned with an expanded vocabulary (text tokens + XCodec2 speech tokens,
vocab 193800). Architecturally it is a plain ``LlamaForCausalLM``: given a text
prompt wrapped in Llasa's TTS chat format, it autoregressively predicts speech
tokens which an external XCodec2 decoder turns into a waveform.

Because the model is a standard causal LM, the entire on-device graph is just the
language-model forward (static shapes) -- there is no multimodal merge or
dynamic op to special-case. ``load_model`` covers that LM forward.

The XCodec2 audio decoder that turns the predicted speech tokens into a waveform
is exposed via ``load_audio_decoder`` / ``decode_speech_tokens`` (below): its
conv/attention backbone runs on Tenstorrent while the iSTFT head stays on CPU
(complex/FFT). See tt-xla #5537.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Llasa model variants."""

    LLASA_8B = "Llasa-8B"


class ModelLoader(ForgeModel):
    """Llasa (Llama3-8B TTS) loader implementation for causal LM TTS tasks."""

    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Sample text to be synthesized into speech.
    sample_text = "Tenstorrent builds hardware and software for AI."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llasa",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. Bring-up uses torch.bfloat16.

        Returns:
            torch.nn.Module: The Llasa (LlamaForCausalLM) model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def _build_tts_prompt(self, text: str):
        """Build the Llasa TTS chat prompt for ``text``.

        Mirrors Llasa's documented usage: the text to synthesize is wrapped in
        ``<|TEXT_UNDERSTANDING_START|>...<|TEXT_UNDERSTANDING_END|>`` and the
        assistant turn is primed with ``<|SPEECH_GENERATION_START|>`` so the
        model continues by emitting speech tokens.
        """
        return [
            {
                "role": "user",
                "content": (
                    "Convert the text to speech:"
                    f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
                ),
            },
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Build static-shape inputs for the Llasa causal LM TTS forward.

        Extra runner kwargs (e.g. ``run_phase``, ``seq_len``) are accepted and
        ignored.

        Args:
            dtype_override: Optional torch.dtype (floating inputs only; ids are
                left as int).
            batch_size: Batch size (replicated along dim 0). ``None`` -> 1.

        Returns:
            dict: ``{"input_ids": Tensor, "attention_mask": Tensor}``.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        batch_size = batch_size or 1

        encoded = self.tokenizer.apply_chat_template(
            self._build_tts_prompt(self.sample_text),
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )
        # apply_chat_template may return a bare tensor or a BatchEncoding/dict
        # depending on the transformers version.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad to a static length so the on-device graph has fixed shapes.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Greedy-decode the next speech token from logits for a quick check."""
        if self.tokenizer is None:
            self._load_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        next_id = logits[:, self.seq_len - 1, :].argmax(dim=-1)
        return self.tokenizer.decode(next_id.tolist())

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style tensor-parallel shard spec for the Llasa LM.

        Column-parallel on q/k/v/gate/up, row-parallel on o/down. Used only for
        multichip bring-up; single-device runs ignore this.
        """
        shard_specs = {}
        shard_specs[model.lm_head.weight] = ("model", "batch")
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llasa model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    # ------------------------------------------------------------------ #
    # XCodec2 acoustic decoder (speech tokens -> waveform).
    #
    # The LM (above) emits XCodec2 speech-token indices; this decoder turns
    # them into audio. It is loaded here so the Llasa TTS model is complete in
    # one place. Only the *decoder* is built (`generator` Vocos vocoder +
    # `fc_post_a`) -- `decode` never touches the Wav2Vec2Bert semantic encoder
    # -- so xcodec2 runs in this env (torch>=2.11, installed --no-deps) without
    # its old torch-2.5 encoder stack. The heavy xcodec2 imports are lazy so
    # test discovery (which imports this loader before requirements install)
    # never needs the package.
    # ------------------------------------------------------------------ #

    XCODEC2_REPO = "HKUSTAudio/xcodec2"

    @staticmethod
    def _stub_encoder_only_deps():
        """Stub xcodec2's encoder-only imports (``torchaudio`` dead import;
        ``torchtune`` RoPE) so the package imports for the decode path without
        pulling torch-ABI-locked extensions (torchaudio/torchao)."""
        import importlib.machinery
        import sys
        import types

        def _stub(name, **attrs):
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules.setdefault(name, m)
            return sys.modules[name]

        _stub("torchaudio", __version__="0.0.0-stub")

        class _RoPE(torch.nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x, *a, **k):
                return x

        _tt = _stub("torchtune")
        _tt.modules = _stub("torchtune.modules", RotaryPositionalEmbeddings=_RoPE)

    def load_audio_decoder(self):
        """Build + weight-load the XCodec2 acoustic decoder (generator + fc_post_a).

        Returns ``(generator, fc_post_a)`` and caches them on the loader.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self._stub_encoder_only_deps()
        from xcodec2.vq.codec_decoder_vocos import CodecDecoderVocos

        generator = CodecDecoderVocos().eval()
        fc_post_a = torch.nn.Linear(2048, 1024)
        full = load_file(hf_hub_download(repo_id=self.XCODEC2_REPO, filename="model.safetensors"))
        generator.load_state_dict(
            {k[len("generator."):]: v for k, v in full.items() if k.startswith("generator.")},
            strict=False,
        )
        fc_post_a.load_state_dict(
            {k[len("fc_post_a."):]: v for k, v in full.items() if k.startswith("fc_post_a.")},
            strict=False,
        )
        self.audio_generator = generator.to(torch.float32)
        self.audio_fc_post_a = fc_post_a.to(torch.float32)
        return self.audio_generator, self.audio_fc_post_a

    def decode_speech_tokens(self, vq_code, backbone_on_tt=False):
        """Decode XCodec2 speech-token indices ``[B, 1, T]`` to a waveform ``[B, 1, time]``.

        With ``backbone_on_tt=True`` the Vocos conv/attention **backbone** runs on
        Tenstorrent via ``torch.compile(backend="tt")``; the **iSTFT head** stays
        on CPU because it uses complex/FFT ops (``torch.fft.irfft``) the TT backend
        cannot lower. ``T`` should be a multiple of 32 (tile alignment) for the TT
        path. ``load_audio_decoder`` must be called first.
        """
        gen, fc = self.audio_generator, self.audio_fc_post_a
        with torch.no_grad():
            emb = gen.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            emb = emb.transpose(1, 2)
            emb = fc(emb.transpose(1, 2)).transpose(1, 2)   # [B, 1024, T]
            feats = emb.transpose(1, 2).contiguous()        # [B, T, 1024]

            if not backbone_on_tt:
                return gen(feats, vq=False)[0]

            import torch_xla
            import torch_xla.core.xla_model as xm

            torch_xla.set_custom_compile_options({"optimization_level": 1})  # backbone GroupNorm
            dev = xm.xla_device()
            backbone = gen.backbone.to(dtype=torch.bfloat16).to(dev)  # head left on CPU/fp32
            compiled = torch.compile(lambda x: backbone(x), backend="tt")
            bb = compiled(feats.to(dtype=torch.bfloat16).to(dev)).to("cpu").to(torch.float32)
            return gen.head(bb)[0]  # iSTFT head: complex/FFT -> CPU only
