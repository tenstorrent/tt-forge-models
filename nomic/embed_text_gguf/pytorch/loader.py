# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Text v1 GGUF model loader implementation for sentence embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional


def _patch_nomic_bert_gguf():
    """Register nomic-bert GGUF architecture in transformers.

    The nomic-ai/nomic-embed-text-v1-GGUF repo ships a GGUF checkpoint whose
    general.architecture is 'nomic-bert', which is not in GGUF_CONFIG_MAPPING.
    This patch adds the field-name mapping and injects the auto_map so that
    AutoConfig and AutoModel can resolve NomicBertConfig / NomicBertModel from
    the nomic-ai/nomic-bert-2048 remote-code repo.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "nomic-bert" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nomic-bert")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nomic-bert"] = {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        # NomicBertConfig extends GPT2Config and uses config.n_inner for MLP size,
        # not config.intermediate_size.  Map feed_forward_length → n_inner directly.
        "feed_forward_length": "n_inner",
        "attention.head_count": "num_attention_heads",
        # GPT2Config (parent of NomicBertConfig) uses layer_norm_epsilon, not
        # layer_norm_eps.  Using the wrong key leaves epsilon at the GPT2 default
        # of 1e-5 instead of the correct 1e-12, causing layer-norm precision loss.
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "rope.freq_base": "rotary_emb_base",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "nomic-bert":
            config["model_type"] = "nomic_bert"
            # The GGUF file does not encode activation_function; nomic-bert
            # uses SwiGLU (fc11 + fc12 gate projection).  Without this the
            # model falls back to a standard single fc1 MLP and the gate
            # weights (blk.N.ffn_gate) cannot be loaded, causing PCC~0.42.
            config.setdefault("activation_function", "swiglu")
            # nomic-embed-text-v1 uses full RoPE (rotary_emb_fraction=1.0).
            # Without this, NomicBertEmbeddings creates randomly-initialised
            # absolute position embeddings (rotary_emb_fraction defaults to
            # 0.0 in NomicBertConfig), which are not in the GGUF and cause
            # catastrophic PCC degradation.
            config.setdefault("rotary_emb_fraction", 1.0)
            # nomic-embed-text-v1 uses no biases in attention or MLP.
            # NomicBertConfig defaults qkv_proj_bias/mlp_fc*_bias to True;
            # without these overrides the model allocates random bias params
            # that are absent from the GGUF checkpoint (they silently retain
            # random init values under strict=False), corrupting output.
            config.setdefault("qkv_proj_bias", False)
            config.setdefault("mlp_fc1_bias", False)
            config.setdefault("mlp_fc2_bias", False)
            # GPT2Config defaults all dropouts to 0.1; the trained model uses
            # 0.0 (no dropout at inference).  These don't affect eval-mode
            # output, but setting them consistently avoids surprises.
            config.setdefault("attn_pdrop", 0.0)
            config.setdefault("resid_pdrop", 0.0)
            config.setdefault("embd_pdrop", 0.0)
            config.setdefault(
                "auto_map",
                {
                    "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
                    "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
                },
            )
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_nomic_bert_gguf()

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nomic Embed Text v1 GGUF model variants for embedding generation."""

    NOMIC_EMBED_TEXT_V1_GGUF = "nomic-embed-text-v1-GGUF"


class ModelLoader(ForgeModel):
    """Nomic Embed Text v1 GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_TEXT_V1_GGUF: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-embed-text-v1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_TEXT_V1_GGUF

    GGUF_FILE = "nomic-embed-text-v1.Q4_K_M.gguf"
    # The GGUF repo has no tokenizer files; load tokenizer from the base model.
    TOKENIZER_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"

    sample_sentences = [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nomic-Embed-Text-v1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_MODEL_NAME, trust_remote_code=True, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        import numpy as np
        import torch
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.utils.hub import cached_file
        from transformers.modeling_gguf_pytorch_utils import (
            get_gguf_hf_weights_map,
            TensorProcessor,
        )
        from gguf import GGUFReader, dequantize
        import transformers.modeling_gguf_pytorch_utils as gguf_utils

        pretrained_model_name = self._variant_config.pretrained_model_name

        gguf_path = cached_file(pretrained_model_name, self.GGUF_FILE)

        # Load config from GGUF (our patch handles "nomic-bert" arch registration
        # and injects auto_map so NomicBertConfig can be resolved).
        config_dict = gguf_utils.load_gguf_checkpoint(gguf_path, return_tensors=False)[
            "config"
        ]

        NomicBertConfig = get_class_from_dynamic_module(
            "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
            pretrained_model_name,
        )
        NomicBertModel = get_class_from_dynamic_module(
            "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
            pretrained_model_name,
        )

        config, _ = NomicBertConfig.from_dict(config_dict, return_unused_kwargs=True)
        # gguf-py MODEL_ARCH_NAMES uses "nomic-bert" (hyphen), not "nomic_bert".
        # get_gguf_hf_weights_map looks up model_type in MODEL_ARCH_NAMES so the
        # model_type on the config must match the hyphenated form.
        config.model_type = "nomic-bert"

        if dtype_override is not None:
            config.torch_dtype = dtype_override

        # Create the model directly.  NomicBertModel.from_pretrained is a custom
        # implementation that only loads PyTorch/SafeTensors weights — it cannot
        # load GGUF weights and will raise OSError on a GGUF-only repo.
        model = NomicBertModel(config, add_pooling_layer=False)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        # Load GGUF tensors directly with the gguf-py reader to avoid the
        # monkey-patched load_gguf_checkpoint chain (other loaders install
        # narrow-signature wrappers that reject the model_to_load kwarg).
        with torch.device("meta"):
            dummy = NomicBertModel(config, add_pooling_layer=False)

        processor = TensorProcessor(config=config_dict)
        tensor_key_mapping = get_gguf_hf_weights_map(dummy, processor)

        reader = GGUFReader(gguf_path)
        state_dict = {}
        for tensor in reader.tensors:
            weights = dequantize(tensor.data, tensor.tensor_type)
            result = processor.process(
                weights=weights,
                name=tensor.name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters={},
            )
            if result.name not in tensor_key_mapping:
                continue
            hf_name = tensor_key_mapping[result.name]
            state_dict[hf_name] = torch.from_numpy(np.copy(result.weights))

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

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
