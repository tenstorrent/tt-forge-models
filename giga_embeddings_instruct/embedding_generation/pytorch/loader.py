# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Giga-Embeddings-instruct model loader for text embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional


def _compute_default_rope_parameters(config, device, seq_len=None, **kwargs):
    # Standard RoPE without scaling — transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available model variants for Giga-Embeddings-instruct."""

    GIGA_EMBEDDINGS_INSTRUCT_4BIT_NF4 = "iMiW/Giga-Embeddings-instruct-4bit-nf4"


class ModelLoader(ForgeModel):
    """Giga-Embeddings-instruct model loader for text embedding generation."""

    _VARIANTS = {
        ModelVariant.GIGA_EMBEDDINGS_INSTRUCT_4BIT_NF4: LLMModelConfig(
            pretrained_model_name="iMiW/Giga-Embeddings-instruct-4bit-nf4",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGA_EMBEDDINGS_INSTRUCT_4BIT_NF4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Giga-Embeddings-instruct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self._patch_autocast(model)

        self.model = model

        return model

    @staticmethod
    def _patch_autocast(model):
        # The model's forward hardcodes torch.autocast('cuda', ...) which fails without a CUDA device.
        # Patch it to use the input tensor's device type instead.
        from transformers.modeling_outputs import BaseModelOutputWithPast

        original_forward = model.forward

        def _forward(input_ids, attention_mask, return_embeddings=False, **kw):
            kw.pop("token_type_ids", None)
            device_type = input_ids.device.type
            if device_type not in ("cpu", "cuda", "mps", "xpu"):
                device_type = "cpu"
            with torch.autocast(
                device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
            ):
                outputs = model.model(
                    input_ids=input_ids, attention_mask=attention_mask, **kw
                )
                last_hidden = model.latent_attention_model(
                    outputs.last_hidden_state, attention_mask
                )
            if return_embeddings:
                return model.mean_pool(last_hidden, attention_mask)
            return BaseModelOutputWithPast(last_hidden_state=last_hidden)

        model.forward = _forward

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "Instruct: Retrieve semantically similar text\nQuery: This is an example sentence for generating text embeddings"

        max_length = getattr(self._variant_config, "max_length", 256)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
