# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arc1-Mini GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
from huggingface_hub import hf_hub_download
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.auto_factory as _auto_factory
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _find_gguf_load_with_model_to_load(fn):
    """Search closure chain and module globals for load_gguf_checkpoint accepting model_to_load.

    Other loaders patch load_gguf_checkpoint at module level (not via closures), so we
    must search __globals__ of each patched function to find the original that has the
    model_to_load parameter required by transformers 5.x.
    """
    seen = set()
    queue = [fn]
    while queue:
        f = queue.pop(0)
        fid = id(f)
        if fid in seen:
            continue
        seen.add(fid)
        try:
            sig = inspect.signature(f)
            if "model_to_load" in sig.parameters:
                return f
        except (ValueError, TypeError):
            pass
        # Search closures (for closure-captured variables)
        if hasattr(f, "__closure__") and f.__closure__:
            for cell in f.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        queue.append(val)
                except ValueError:
                    pass
        # Search module globals for other load_gguf variants (module-level _orig references)
        if hasattr(f, "__globals__"):
            for name, val in f.__globals__.items():
                if callable(val) and "load_gguf" in name and id(val) not in seen:
                    queue.append(val)
    return None


from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Arc1-Mini GGUF model variants for causal language modeling."""

    ARC1_MINI_GGUF = "Arc1_Mini_GGUF"


class ModelLoader(ForgeModel):
    """Arc1-Mini GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ARC1_MINI_GGUF: LLMModelConfig(
            pretrained_model_name="meissosisai/arc1-mini",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ARC1_MINI_GGUF

    GGUF_FILE = "arc1-mini.gguf"

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

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
            model="Arc1-Mini GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        # Load from JSON files (not GGUF) so the chat_template from
        # tokenizer_config.json is included; GGUF tokenizer data omits it.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _from_pretrained_no_adapter(self, pretrained_model_name, **kwargs):
        # arc1-mini has an adapter_config.json (it's a LoRA repo) but the GGUF
        # already has merged weights. We need two fixes:
        # 1. Suppress adapter detection so transformers does not redirect model
        #    loading to the base model repo.
        # 2. Restore a load_gguf_checkpoint that accepts model_to_load, since
        #    other loaders in this test session may have registered an older patch
        #    that lacks this parameter (required by transformers 5.x).
        _orig_adapter = _auto_factory.find_adapter_config_file
        _auto_factory.find_adapter_config_file = lambda *a, **kw: None

        _current_gguf = _gguf_utils.load_gguf_checkpoint
        _orig_gguf = _find_gguf_load_with_model_to_load(_current_gguf)
        if _orig_gguf and _orig_gguf is not _current_gguf:
            _gguf_utils.load_gguf_checkpoint = _orig_gguf

        try:
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **kwargs
            ).eval()
        finally:
            _auto_factory.find_adapter_config_file = _orig_adapter
            _gguf_utils.load_gguf_checkpoint = _current_gguf

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Pre-load config using just the GGUF filename so cached_file can resolve
        # it by repo ID + filename (the normal HF Hub path).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # Pre-download the GGUF so we have a local absolute path. Passing the
        # absolute path as gguf_file lets transformers use os.path.isfile() to
        # resolve the checkpoint file instead of going through cached_file() with
        # the repo ID, which returns None for this repo (the repo has an
        # adapter_config.json that causes commit-hash mismatches in the lookup).
        local_gguf = hf_hub_download(
            repo_id=pretrained_model_name,
            filename=self.GGUF_FILE,
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["config"] = config
        model_kwargs["gguf_file"] = local_gguf

        model = self._from_pretrained_no_adapter(pretrained_model_name, **model_kwargs)

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
