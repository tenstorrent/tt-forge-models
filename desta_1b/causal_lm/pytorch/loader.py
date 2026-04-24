# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DESTA-1B model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download
from typing import Optional


class _SentencePieceTokenizer:
    """Thin sentencepiece wrapper; AutoTokenizer (transformers>=5) can't load this model's sp vocab."""

    def __init__(self, sp_model_path, bos_token_id=1, eos_token_id=2, pad_token_id=3):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.pad_token = self.sp.IdToPiece(pad_token_id)
        self.eos_token = self.sp.IdToPiece(eos_token_id)

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=None,
    ):
        texts = [text] if isinstance(text, str) else text
        encoded = [self.sp.Encode(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [e[:max_length] for e in encoded]
        max_len = max((len(e) for e in encoded), default=0)
        input_ids, attention_mask = [], []
        for e in encoded:
            pad_len = (max_len - len(e)) if padding else 0
            input_ids.append(e + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(e) + [0] * pad_len)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.sp.Decode(token_ids)


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
    """Available DESTA-1B model variants."""

    DESTA_1B = "desta_1b"


class ModelLoader(ForgeModel):
    """DESTA-1B model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DESTA_1B: LLMModelConfig(
            pretrained_model_name="mewaeltsegay/desta_1b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DESTA_1B

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DESTA-1B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        sp_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "sentencepiece.model"
        )
        self.tokenizer = _SentencePieceTokenizer(sp_path)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        # DynamicCache removed in transformers>=5.0; disable cache.
        config.use_cache = False
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

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
