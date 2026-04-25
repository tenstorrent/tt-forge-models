# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ultravox model loader implementation for speech language modeling.
"""

import json
import os
import shutil
import tempfile

import numpy as np
from typing import Optional

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
    """Available Ultravox model variants."""

    V0_3 = "v0_3"
    V0_4 = "v0_4"
    V0_5_LLAMA_3_2_1B = "v0_5_Llama_3_2_1B"
    V0_5_LLAMA_3_1_8B = "v0_5_Llama_3_1_8B"
    HAUSA_STAGE2_LAST = "hausa-ultravox-stage2-last"


class ModelLoader(ForgeModel):
    """Ultravox model loader implementation for speech language modeling tasks."""

    _VARIANTS = {
        ModelVariant.V0_3: ModelConfig(
            pretrained_model_name="fixie-ai/ultravox-v0_3",
        ),
        ModelVariant.V0_4: ModelConfig(
            pretrained_model_name="fixie-ai/ultravox-v0_4",
        ),
        ModelVariant.V0_5_LLAMA_3_2_1B: ModelConfig(
            pretrained_model_name="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        ),
        ModelVariant.V0_5_LLAMA_3_1_8B: ModelConfig(
            pretrained_model_name="fixie-ai/ultravox-v0_5-llama-3_1-8b",
        ),
        ModelVariant.HAUSA_STAGE2_LAST: ModelConfig(
            pretrained_model_name="vaghawan/hausa-ultravox-stage2-last",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_5_LLAMA_3_2_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._patched_dir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ultravox",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    _TEXT_CONFIGS = {
        ModelVariant.V0_5_LLAMA_3_2_1B: {
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
        },
        ModelVariant.V0_5_LLAMA_3_1_8B: {
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
        },
    }

    def _get_text_config(self):
        """Return the text model config for the current variant."""
        return self._TEXT_CONFIGS[self._variant]

    def _get_patched_model_dir(self):
        """Create a local directory with a patched config.json that avoids gated repo access.

        The Ultravox custom config tries to fetch the gated Llama config from
        HuggingFace at init time. We create a patched local copy with
        text_model_id set to null and text_config provided inline.
        """
        if self._patched_dir is not None:
            return self._patched_dir

        from huggingface_hub import hf_hub_download, model_info

        pretrained_model_name = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        # Nullify text_model_id to prevent fetching the gated Llama config.
        # Provide a text_config dict so the custom UltravoxConfig uses it.
        config_dict["text_model_id"] = None
        if "text_config" not in config_dict or not isinstance(
            config_dict.get("text_config"), dict
        ):
            # Provide inline text_config to avoid fetching gated Llama configs.
            text_configs = {
                ModelVariant.V0_5_LLAMA_3_2_1B: {
                    "model_type": "llama",
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_attention_heads": 32,
                    "num_hidden_layers": 16,
                    "num_key_value_heads": 8,
                    "vocab_size": 128256,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 500000.0,
                },
                ModelVariant.V0_5_LLAMA_3_1_8B: {
                    "model_type": "llama",
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "num_attention_heads": 32,
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "vocab_size": 128256,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 500000.0,
                },
                ModelVariant.HAUSA_STAGE2_LAST: {
                    "model_type": "llama",
                    "hidden_size": 8192,
                    "intermediate_size": 28672,
                    "num_attention_heads": 64,
                    "num_hidden_layers": 80,
                    "num_key_value_heads": 8,
                    "vocab_size": 128256,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 500000.0,
                },
            }
            config_dict["text_config"] = text_configs[self._variant]

        tmpdir = tempfile.mkdtemp()

        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config_dict, f)

        # Copy .py and tokenizer files; symlink only model weight files from the
        # HF cache so we can load from patched_dir directly (with patched code).
        # Exclude training artifacts (optimizer states, FSDP checkpoints, etc.)
        # to avoid accidentally downloading huge files we don't need.
        _WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}
        _TRAINING_ARTIFACTS = {
            "optimizer.bin",
            "pytorch_model_fsdp.bin",
            "trainer_state.json",
            "training_args.bin",
            "scheduler.pt",
        }
        info = model_info(pretrained_model_name)
        for sibling in info.siblings:
            fname = sibling.rfilename
            if fname == "config.json":
                # Already wrote the patched version above.
                continue
            if fname in _TRAINING_ARTIFACTS or fname.startswith("rng_state"):
                # Skip large training-only artifacts.
                continue
            src = hf_hub_download(pretrained_model_name, fname)
            if (
                fname.endswith(".py")
                or "tokenizer" in fname
                or fname.endswith(".model")
            ):
                shutil.copy2(src, os.path.join(tmpdir, fname))
            elif any(fname.endswith(ext) for ext in _WEIGHT_EXTS):
                # Symlink weight files to avoid copying large files.
                os.symlink(src, os.path.join(tmpdir, fname))
            # Skip other files (preprocessor_config, gitattributes, etc.)

        # Patch ultravox_model.py for transformers >= 5.x:
        # tie_weights() is now called with recompute_mapping=False.
        _model_py = os.path.join(tmpdir, "ultravox_model.py")
        if os.path.exists(_model_py):
            with open(_model_py) as f:
                _code = f.read()
            _code = _code.replace(
                "def tie_weights(self):",
                "def tie_weights(self, **kwargs):",
            )
            with open(_model_py, "w") as f:
                f.write(_code)

        # Patch ultravox_processing.py for transformers >= 5.x:
        # ProcessorMixin.__init__ now requires audio_processor to be a
        # FeatureExtractionMixin, not a full WhisperProcessor.
        _proc_py = os.path.join(tmpdir, "ultravox_processing.py")
        if os.path.exists(_proc_py):
            with open(_proc_py) as f:
                _code = f.read()
            # Update the accepted class to FeatureExtractionMixin subclass.
            _code = _code.replace(
                'audio_processor_class = ("WhisperProcessor",)',
                'audio_processor_class = ("WhisperFeatureExtractor",)',
            )
            # Extract feature_extractor from WhisperProcessor after loading.
            _code = _code.replace(
                "        if audio_processor is None:\n"
                "            audio_processor = transformers.AutoProcessor.from_pretrained(\n"
                '                "openai/whisper-tiny"\n'
                "            )\n"
                "\n"
                "        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)",
                "        if audio_processor is None:\n"
                "            audio_processor = transformers.AutoProcessor.from_pretrained(\n"
                '                "openai/whisper-tiny"\n'
                "            )\n"
                "        if hasattr(audio_processor, 'feature_extractor'):\n"
                "            audio_processor = audio_processor.feature_extractor\n"
                "\n"
                "        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)",
            )
            # Also extract feature_extractor in from_pretrained before passing to cls().
            _code = _code.replace(
                "        audio_processor = transformers.AutoProcessor.from_pretrained(\n"
                "            config.audio_model_id\n"
                "            or config.audio_config._name_or_path\n"
                '            or "openai/whisper-tiny"\n'
                "        )\n"
                "\n"
                "        tokenizer = transformers.AutoTokenizer.from_pretrained(",
                "        audio_processor = transformers.AutoProcessor.from_pretrained(\n"
                "            config.audio_model_id\n"
                "            or config.audio_config._name_or_path\n"
                '            or "openai/whisper-tiny"\n'
                "        )\n"
                "        if hasattr(audio_processor, 'feature_extractor'):\n"
                "            audio_processor = audio_processor.feature_extractor\n"
                "\n"
                "        tokenizer = transformers.AutoTokenizer.from_pretrained(",
            )
            # Update hop_length access: audio_processor is now the feature extractor.
            _code = _code.replace(
                "self.audio_processor.feature_extractor.hop_length",
                "self.audio_processor.hop_length",
            )
            with open(_proc_py, "w") as f:
                f.write(_code)

        self._patched_dir = tmpdir
        return tmpdir

    def _cleanup_patched_dir(self):
        if self._patched_dir is not None:
            shutil.rmtree(self._patched_dir, ignore_errors=True)
            self._patched_dir = None

    def _load_processor(self):
        """Load processor for the current variant."""
        from transformers import AutoProcessor

        patched_dir = self._get_patched_model_dir()
        self.processor = AutoProcessor.from_pretrained(
            patched_dir,
            trust_remote_code=True,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ultravox model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ultravox model instance.
        """
        import transformers
        import transformers.modeling_utils

        # transformers >= 5.x removed _init_weights; the custom ultravox_model.py
        # uses it to decide whether to load from pretrained (True) vs create with
        # empty/meta weights (False). We emulate the old behavior: return False when
        # inside a meta-device context (set by low_cpu_mem_usage from_pretrained).
        if not hasattr(transformers.modeling_utils, "_init_weights"):

            class _InitWeightsSentinel:
                def __bool__(self):
                    try:
                        import torch

                        return torch.get_default_device().type != "meta"
                    except Exception:
                        return True

            transformers.modeling_utils._init_weights = _InitWeightsSentinel()

        patched_dir = self._get_patched_model_dir()

        config = transformers.AutoConfig.from_pretrained(
            patched_dir, trust_remote_code=True
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load from patched_dir so the patched ultravox_model.py code is used.
        # patched_dir has symlinks to the HF-cached weights, so weights load correctly.
        model = transformers.AutoModel.from_pretrained(
            patched_dir,
            config=config,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Ultravox model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic 3-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 3
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        turns = [
            {
                "role": "user",
                "content": "<|audio|>\nDescribe what you hear in this audio.",
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=text,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
