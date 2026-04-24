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

        # Copy all .py and tokenizer files from the repo
        info = model_info(pretrained_model_name)
        for sibling in info.siblings:
            fname = sibling.rfilename
            if (
                fname.endswith(".py")
                or "tokenizer" in fname
                or fname.endswith(".model")
            ):
                src = hf_hub_download(pretrained_model_name, fname)
                dst = os.path.join(tmpdir, fname)
                shutil.copy2(src, dst)
                # Patch ultravox_model.py: replace removed transformers attribute with
                # a meta-device context check (the original intent of _init_weights).
                if fname.endswith("ultravox_model.py"):
                    with open(dst) as f:
                        content = f.read()
                    content = content.replace(
                        "transformers.modeling_utils._init_weights",
                        "(torch.tensor([]).device.type != 'meta')",
                    )
                    with open(dst, "w") as f:
                        f.write(content)
                # Patch ultravox_processing.py for transformers 5.x:
                # MODALITY_TO_BASE_CLASS_MAPPING["audio_processor"] now requires
                # FeatureExtractionMixin, so extract feature_extractor from the
                # WhisperProcessor returned by AutoProcessor.from_pretrained().
                if fname.endswith("ultravox_processing.py"):
                    with open(dst) as f:
                        content = f.read()
                    content = content.replace(
                        "        audio_processor = transformers.AutoProcessor.from_pretrained(\n"
                        "            config.audio_model_id\n"
                        "            or config.audio_config._name_or_path\n"
                        '            or "openai/whisper-tiny"\n'
                        "        )",
                        "        _audio_proc = transformers.AutoProcessor.from_pretrained(\n"
                        "            config.audio_model_id\n"
                        "            or config.audio_config._name_or_path\n"
                        '            or "openai/whisper-tiny"\n'
                        "        )\n"
                        "        audio_processor = getattr(_audio_proc, 'feature_extractor', _audio_proc)",
                    )
                    content = content.replace(
                        "            hop_length = self.audio_processor.feature_extractor.hop_length",
                        "            _fe = getattr(self.audio_processor, 'feature_extractor', self.audio_processor)\n"
                        "            hop_length = _fe.hop_length",
                    )
                    with open(dst, "w") as f:
                        f.write(content)

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

    def _patch_hf_module_cache(self):
        """Patch the HuggingFace cached ultravox_model.py to fix transformers API changes.

        In transformers 5.x, modeling_utils._init_weights was removed. The cached
        ultravox_model.py uses it to detect whether model init is running inside a
        meta-device context (transformers now always wraps cls() in torch.device('meta')).
        Replace all references with an equivalent runtime check.
        """
        import glob
        import sys

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        local_hf_home = os.path.join(
            os.path.dirname(__file__), "..", "..", ".cache", "huggingface"
        )

        replacements = [
            # transformers 5.x removed _init_weights module attribute; replace with
            # equivalent meta-device context check (models must not call from_pretrained
            # while a meta device is active, so we use the empty-weights path instead).
            (
                "transformers.modeling_utils._init_weights",
                "(torch.tensor([]).device.type != 'meta')",
            ),
            # transformers 5.x added recompute_mapping kwarg to tie_weights(); the old
            # UltravoxModel.tie_weights() override does not accept it.
            (
                "    def tie_weights(self):",
                "    def tie_weights(self, **kwargs):",
            ),
            # transformers 5.x removed layer_head_mask from WhisperEncoderLayer.forward().
            (
                "                    layer_outputs = encoder_layer(\n"
                "                        hidden_states,\n"
                "                        attention_mask,\n"
                "                        layer_head_mask=(\n"
                "                            head_mask[idx] if head_mask is not None else None\n"
                "                        ),\n"
                "                        output_attentions=output_attentions,\n"
                "                    )",
                "                    layer_outputs = encoder_layer(\n"
                "                        hidden_states,\n"
                "                        attention_mask,\n"
                "                        output_attentions=output_attentions,\n"
                "                    )",
            ),
        ]

        for search_root in [local_hf_home, hf_home]:
            pattern = os.path.join(
                search_root,
                "modules",
                "transformers_modules",
                "**",
                "ultravox_model.py",
            )
            for cached_file in glob.glob(pattern, recursive=True):
                with open(cached_file) as f:
                    content = f.read()
                patched = content
                for old, new in replacements:
                    patched = patched.replace(old, new)
                if patched != content:
                    with open(cached_file, "w") as f:
                        f.write(patched)
                    # Remove stale .pyc so Python reloads the patched source.
                    pycache = os.path.join(
                        os.path.dirname(cached_file),
                        "__pycache__",
                    )
                    if os.path.isdir(pycache):
                        shutil.rmtree(pycache)
                    # Invalidate any already-imported module.
                    mod_name = next(
                        (k for k in sys.modules if "ultravox_model" in k), None
                    )
                    if mod_name:
                        del sys.modules[mod_name]

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ultravox model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ultravox model instance.
        """
        import transformers

        self._patch_hf_module_cache()

        pretrained_model_name = self._variant_config.pretrained_model_name
        patched_dir = self._get_patched_model_dir()

        config = transformers.AutoConfig.from_pretrained(
            patched_dir, trust_remote_code=True
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = transformers.AutoModel.from_pretrained(
            pretrained_model_name,
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
