# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-TTS (zai-org/GLM-TTS) model loader implementation for text-to-speech
tasks.

GLM-TTS is a two-stage zero-shot TTS system whose first stage is a
Llama-based causal language model that converts text into speech-token
sequences, and whose second stage is a flow-matching model that produces
mel-spectrograms consumed by a vocoder. This loader targets the
Llama-based LLM backbone, which is stored in the ``llm/`` subfolder of
the HuggingFace repository alongside the flow-matching and vocoder
components.
"""
import os
import shutil
import tempfile
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GLM-TTS model variants."""

    GLM_TTS = "GLM-TTS"


class ModelLoader(ForgeModel):
    """GLM-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.GLM_TTS: ModelConfig(
            pretrained_model_name="zai-org/GLM-TTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_TTS

    _LLM_SUBFOLDER = "llm"
    _TOKENIZER_SUBFOLDER = "vq32k-phoneme-tokenizer"

    sample_text = "Hello, how are you today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLM-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # zai-org/GLM-TTS does not ship tokenization_chatglm.py or tokenizer.model at
    # the repo root, so AutoTokenizer with trust_remote_code cannot resolve the
    # ChatGLM4Tokenizer class.  We bootstrap from THUDM/glm-4-9b-chat, which
    # provides a compatible implementation, then combine it with the GLM-TTS
    # phoneme tokenizer_config.json so the vocabulary is correct.
    _FALLBACK_TOKENIZER_REPO = "THUDM/glm-4-9b-chat"

    def _load_tokenizer(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self._variant_config.pretrained_model_name,
                    subfolder=self._TOKENIZER_SUBFOLDER,
                    trust_remote_code=True,
                )
            except OSError:
                from huggingface_hub import hf_hub_download

                tmpdir = tempfile.mkdtemp(prefix="glm_tts_tok_")
                code_src = hf_hub_download(
                    self._FALLBACK_TOKENIZER_REPO, "tokenization_chatglm.py"
                )
                vocab_src = hf_hub_download(
                    self._FALLBACK_TOKENIZER_REPO, "tokenizer.model"
                )
                config_src = hf_hub_download(
                    self._variant_config.pretrained_model_name,
                    "tokenizer_config.json",
                    subfolder=self._TOKENIZER_SUBFOLDER,
                )
                for src in (code_src, vocab_src):
                    shutil.copy(src, tmpdir)
                shutil.copy(config_src, os.path.join(tmpdir, "tokenizer_config.json"))
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tmpdir, trust_remote_code=True
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"subfolder": self._LLM_SUBFOLDER}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        tokenizer = self._load_tokenizer()
        # add_special_tokens=False prevents high-ID role/BOS tokens (e.g. >98303)
        # from being prepended, which would exceed the model's vocab_size.
        inputs = tokenizer(
            self.sample_text, return_tensors="pt", add_special_tokens=False
        )
        return inputs
