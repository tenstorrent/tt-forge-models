# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling.
"""
import zipfile
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
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
    """Available Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model variants for causal language modeling."""

    META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE = (
        "Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile"
    )


class ModelLoader(ForgeModel):
    """Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="mozilla-ai/Meta-Llama-3.1-70B-Instruct-llamafile",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE

    GGUF_FILE = "Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile"
    GGUF_EXTRACTED_NAME = "Meta-Llama-3.1-70B-Instruct.Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

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
            model="Meta-Llama-3.1-70B-Instruct-llamafile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _extract_gguf(self) -> tuple[str, str]:
        """Download llamafile and extract embedded GGUF, returning (snapshot_dir, gguf_filename)."""
        llamafile_path = Path(
            hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename=self.GGUF_FILE,
            )
        )
        snapshot_dir = llamafile_path.parent
        gguf_path = snapshot_dir / self.GGUF_EXTRACTED_NAME

        if not gguf_path.exists():
            with zipfile.ZipFile(llamafile_path) as zf:
                gguf_members = [f for f in zf.namelist() if f.endswith(".gguf")]
                if not gguf_members:
                    raise ValueError(f"No GGUF file found inside {self.GGUF_FILE}")
                zf.extract(gguf_members[0], snapshot_dir)

        return str(snapshot_dir), self.GGUF_EXTRACTED_NAME

    def _load_tokenizer(self, dtype_override=None):
        snapshot_dir, gguf_filename = self._extract_gguf()
        tokenizer_kwargs = {"gguf_file": gguf_filename}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        snapshot_dir, gguf_filename = self._extract_gguf()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": gguf_filename}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(snapshot_dir, gguf_file=gguf_filename)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        snapshot_dir, gguf_filename = self._extract_gguf()
        self.config = AutoConfig.from_pretrained(snapshot_dir, gguf_file=gguf_filename)
        return self.config
