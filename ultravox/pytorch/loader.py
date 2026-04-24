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

    # Training-only files from the HF repo that are not needed for inference.
    _TRAINING_ARTIFACTS = frozenset(
        {
            "optimizer.bin",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            ".gitattributes",
        }
    )

    def _get_patched_model_dir(self):
        """Create a local directory with patched model files for transformers compat.

        Patches applied:
        - config.json: text_model_id=null to avoid gated Llama download; inline
          text_config so the custom UltravoxConfig doesn't need to fetch it.
        - ultravox_model.py: compatibility fixes for transformers>=5 (removed
          _init_weights attribute, tie_weights new **kwargs signature).

        Weight files (.safetensors/.bin) are symlinked from the HF hub cache so
        from_pretrained can be called with this local dir instead of the remote
        model name.
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

        # Copy/symlink all non-training files from the repo
        weight_exts = (".safetensors", ".bin", ".pt", ".pth")
        info = model_info(pretrained_model_name)
        for sibling in info.siblings:
            fname = sibling.rfilename
            if fname in self._TRAINING_ARTIFACTS or fname.startswith("rng_state"):
                continue
            src = hf_hub_download(pretrained_model_name, fname)
            dst = os.path.join(tmpdir, fname)
            if any(fname.endswith(ext) for ext in weight_exts):
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)

        self._patch_ultravox_model(tmpdir)

        self._patched_dir = tmpdir
        return tmpdir

    def _patch_ultravox_model(self, tmpdir):
        """Apply transformers>=5 compatibility patches to ultravox_model.py."""
        model_py = os.path.join(tmpdir, "ultravox_model.py")
        if not os.path.exists(model_py):
            return
        with open(model_py) as f:
            content = f.read()
        # _init_weights module attribute removed in transformers>=5
        content = content.replace(
            "transformers.modeling_utils._init_weights",
            "getattr(transformers.modeling_utils, '_init_weights', True)",
        )
        # tie_weights gained a **kwargs parameter in transformers>=5
        content = content.replace(
            "    def tie_weights(self):\n        return self.language_model.tie_weights()",
            "    def tie_weights(self, **kwargs):\n        return self.language_model.tie_weights(**kwargs)",
        )
        with open(model_py, "w") as f:
            f.write(content)

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
        import transformers.initialization as _tf_init

        patched_dir = self._get_patched_model_dir()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers>=5 unconditionally uses torch.device("meta") in get_init_context,
        # but the custom UltravoxModel calls from_pretrained inside __init__ for its
        # audio tower, which fails inside a meta device context.  Temporarily override
        # get_init_context to skip the meta device so nested from_pretrained works.
        original_get_init_context = transformers.PreTrainedModel.get_init_context

        @classmethod  # type: ignore[misc]
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            from transformers.modeling_utils import local_torch_dtype

            return [local_torch_dtype(dtype, cls.__name__), _tf_init.no_tie_weights()]

        transformers.PreTrainedModel.get_init_context = _no_meta_get_init_context
        try:
            # Load from patched_dir: patched config.json + patched .py files +
            # symlinked weight files — no remote fetch needed.
            model = transformers.AutoModel.from_pretrained(
                patched_dir,
                trust_remote_code=True,
                **model_kwargs,
            )
        finally:
            transformers.PreTrainedModel.get_init_context = original_get_init_context

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
