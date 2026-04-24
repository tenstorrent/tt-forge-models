# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
solidrust Llama-3.1-8B-Lexi-Uncensored-V2 AWQ model loader implementation for causal language modeling.
"""
# gptqmodel 4.x hardcodes `from transformers.modeling_utils import no_init_weights`
# but that symbol moved to `transformers.initialization` in transformers 5.x.
# Patch all affected gptqmodel source files before import so the fix survives
# module-cache purges and reimports.
import importlib
import importlib.util
import os as _os
import re as _re
import sys as _sys


def _patch_gptqmodel_for_transformers5():
    spec = importlib.util.find_spec("gptqmodel")
    if spec is None:
        return
    gptqmodel_dir = _os.path.dirname(spec.origin)
    _target = "from transformers.initialization import no_init_weights"
    _pattern = _re.compile(
        r"^(\s*)from transformers\.modeling_utils import no_init_weights",
        _re.MULTILINE,
    )
    patched_any = False
    for dirpath, _dirs, files in _os.walk(gptqmodel_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = _os.path.join(dirpath, fname)
            with open(fpath) as _f:
                src = _f.read()
            if _target in src or not _pattern.search(src):
                continue

            def _make_replacement(m):
                indent = m.group(1)
                return (
                    f"{indent}try:\n"
                    f"{indent}    from transformers.modeling_utils import no_init_weights\n"
                    f"{indent}except ImportError:\n"
                    f"{indent}    from transformers.initialization import no_init_weights"
                )

            new_src = _pattern.sub(_make_replacement, src)
            if new_src != src:
                with open(fpath, "w") as _f:
                    _f.write(new_src)
                patched_any = True
    if patched_any:
        for _k in list(_sys.modules.keys()):
            if _k.split(".")[0] == "gptqmodel":
                del _sys.modules[_k]
        importlib.invalidate_caches()


_patch_gptqmodel_for_transformers5()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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
    """Available solidrust Llama-3.1-8B-Lexi-Uncensored-V2 AWQ model variants for causal language modeling."""

    SOLIDRUST_LLAMA_3_1_8B_LEXI_UNCENSORED_V2_AWQ = (
        "Llama_3.1_8B_Lexi_Uncensored_V2_AWQ"
    )


class ModelLoader(ForgeModel):
    """solidrust Llama-3.1-8B-Lexi-Uncensored-V2 AWQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SOLIDRUST_LLAMA_3_1_8B_LEXI_UNCENSORED_V2_AWQ: LLMModelConfig(
            pretrained_model_name="solidrust/Llama-3.1-8B-Lexi-Uncensored-V2-AWQ",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SOLIDRUST_LLAMA_3_1_8B_LEXI_UNCENSORED_V2_AWQ

    sample_text = "Hey how are you doing today?"

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
            model="solidrust Llama-3.1-8B-Lexi-Uncensored-V2 AWQ",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["device_map"] = "cpu"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

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

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
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
