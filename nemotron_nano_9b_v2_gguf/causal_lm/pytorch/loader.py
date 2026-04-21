# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 9B v2 GGUF model loader implementation for causal language modeling.

The Nemotron-H architecture (hybrid Mamba2+attention) is not supported by the
transformers GGUF loader, so we load from the base HuggingFace model using
trust_remote_code=True.  The variant name preserves the original GGUF intent;
the actual weights source is the nvidia/NVIDIA-Nemotron-Nano-9B-v2 repo.
"""
import contextlib
import importlib.metadata
import importlib.util
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _ensure_gguf_importable():
    """Patch gguf.__version__ and clear is_gguf_available cache for transformers compatibility.

    When RequirementsManager installs gguf after transformers is already imported,
    transformers' PACKAGE_DISTRIBUTION_MAPPING is stale. is_gguf_available() falls
    back to getattr(gguf, '__version__', 'N/A'), which returns 'N/A' since gguf has
    no __version__ attribute. version.parse('N/A') then raises InvalidVersion.
    Setting gguf.__version__ from importlib.metadata and clearing the lru_cache
    lets is_gguf_available() re-evaluate correctly.
    """
    if importlib.util.find_spec("gguf") is None:
        return
    import gguf

    if not hasattr(gguf, "__version__"):
        try:
            gguf.__version__ = importlib.metadata.version("gguf")
        except importlib.metadata.PackageNotFoundError:
            return

    from transformers.utils.import_utils import is_gguf_available

    is_gguf_available.cache_clear()


def _patch_cuda_no_op():
    """Make torch.cuda.stream a no-op when CUDA is unavailable (e.g. Tenstorrent hardware).

    NemotronH's naive Mamba fallback path calls torch.cuda.stream() unconditionally
    (to avoid NaN on multi-GPU).  Without this patch the call raises AssertionError
    on non-CUDA builds.  When CUDA is absent the stream is meaningless anyway, so
    replacing it with a null context is safe.
    """
    if torch.cuda.is_available():
        return

    @contextlib.contextmanager
    def _null_stream(stream=None):
        yield

    def _null_default_stream(device=None):
        return None

    torch.cuda.stream = _null_stream
    torch.cuda.default_stream = _null_default_stream


_patch_cuda_no_op()


class ModelVariant(StrEnum):
    """Available NVIDIA Nemotron Nano 9B v2 GGUF model variants for causal language modeling."""

    NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF = "9B_v2_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """NVIDIA Nemotron Nano 9B v2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF

    # Original GGUF source; not used for loading since nemotron_h GGUF architecture
    # is not supported by transformers GGUF utilities.
    GGUF_FILE = "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NVIDIA Nemotron Nano 9B v2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _ensure_gguf_importable()
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        _ensure_gguf_importable()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
