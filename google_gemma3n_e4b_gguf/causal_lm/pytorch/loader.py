# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Google Gemma 3n E4B GGUF (bartowski) model loader implementation for causal language modeling.
"""

import importlib.metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _ensure_gguf_version():
    """Set gguf.__version__ so transformers' is_gguf_available() can parse it.

    transformers captures importlib.metadata.packages_distributions() at import
    time (module-level). When gguf is installed dynamically (via RequirementsManager
    after transformers is already imported), the stale mapping causes a KeyError and
    the fallback reads getattr(gguf, '__version__', 'N/A') — which is 'N/A' because
    the gguf package does not define __version__. version.parse('N/A') then raises
    InvalidVersion. Setting gguf.__version__ here fixes the fallback path.
    """
    try:
        import gguf

        if not hasattr(gguf, "__version__") or gguf.__version__ == "N/A":
            gguf.__version__ = importlib.metadata.version("gguf")
    except Exception:
        pass


def _patch_gemma3n_support():
    """Register gemma3n GGUF architecture in transformers.

    transformers 5.x has Gemma3nForCausalLM but lacks GGUF loading support for
    the gemma3n architecture. We bridge that gap by aliasing gemma3n onto the
    existing gemma3 config/tensor mappings and remapping model_type to gemma3n_text
    (which is what Gemma3nForCausalLM expects).
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        Gemma2TensorProcessor,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGemmaConverter,
    )

    if "gemma3n" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gemma3n"] = dict(
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["gemma3"]
    )
    TENSOR_PROCESSORS["gemma3n"] = Gemma2TensorProcessor
    GGUF_TO_FAST_CONVERTERS.setdefault("gemma3n", GGUFGemmaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("gemma3n_text", GGUFGemmaConverter)

    # Also teach get_gguf_hf_weights_map to map gemma3n_text → gemma3n so that
    # tensor loading works when the model's config.model_type is "gemma3n_text".
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, **kwargs
    ):
        effective_type = (
            hf_model.config.model_type
            if model_type is None and hf_model is not None
            else model_type
        )
        if effective_type == "gemma3n_text":
            model_type = "gemma3n"
        return _orig_get_map(hf_model, processor, model_type=model_type, **kwargs)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

    # Walk the patch chain to find the real transformers load_gguf_checkpoint,
    # identified by having "model_to_load" in its signature.  Other models patch
    # this function with (gguf_path, return_tensors=False) and don't forward
    # model_to_load; by calling the real function directly we avoid TypeError
    # while still passing model_to_load through for correct tensor mapping.
    import inspect

    _real_load = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        try:
            if "model_to_load" in inspect.signature(_real_load).parameters:
                break
        except (ValueError, TypeError):
            break
        closure = getattr(_real_load, "__closure__", None) or []
        found = None
        for cell in closure:
            try:
                val = cell.cell_contents
                if callable(val) and id(val) not in seen:
                    seen.add(id(val))
                    found = val
                    break
            except ValueError:
                pass
        if found is None:
            break
        _real_load = found

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        result = _real_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        if result.get("config", {}).get("model_type") == "gemma3n":
            result["config"]["model_type"] = "gemma3n_text"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


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
    """Available Google Gemma 3n E4B GGUF model variants for causal language modeling."""

    GOOGLE_GEMMA_3N_E4B_IT_GGUF = "google_gemma_3n_E4B_IT_GGUF"


class ModelLoader(ForgeModel):
    """Google Gemma 3n E4B GGUF (bartowski) model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GOOGLE_GEMMA_3N_E4B_IT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/google_gemma-3n-E4B-it-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOOGLE_GEMMA_3N_E4B_IT_GGUF

    GGUF_FILE = "google_gemma-3n-E4B-it-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Google Gemma 3n E4B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _ensure_gguf_version()
        _patch_gemma3n_support()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_gguf_version()
        _patch_gemma3n_support()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
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
        _ensure_gguf_version()
        _patch_gemma3n_support()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
