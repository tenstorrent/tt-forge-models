# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 8B GGUF model loader implementation for image to text.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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


def _register_qwen3vl_gguf_support():
    """Register qwen3vl in transformers GGUF architecture tables.

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF loading
    support for the qwen3vl architecture.  Registers the config field mapping and
    tokenizer converter so that load_gguf_checkpoint can parse the metadata.
    Tensor loading is handled separately in load_model() to bypass the
    model_to_load keyword argument incompatibility in other loaders' patches.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3_vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


_register_qwen3vl_gguf_support()

_TEXT_CONFIG_KEYS = [
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "vocab_size",
    "tie_word_embeddings",
]


def _build_qwen3vl_gguf_tensor_mapping(n_layers):
    """Return a {gguf_name: hf_param_name} dict for Qwen3VL text backbone."""
    m = {
        "token_embd.weight": "model.language_model.embed_tokens.weight",
        "output_norm.weight": "model.language_model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    for i in range(n_layers):
        g = f"blk.{i}."
        h = f"model.language_model.layers.{i}."
        m.update(
            {
                f"{g}attn_q.weight": f"{h}self_attn.q_proj.weight",
                f"{g}attn_k.weight": f"{h}self_attn.k_proj.weight",
                f"{g}attn_v.weight": f"{h}self_attn.v_proj.weight",
                f"{g}attn_output.weight": f"{h}self_attn.o_proj.weight",
                f"{g}attn_q_norm.weight": f"{h}self_attn.q_norm.weight",
                f"{g}attn_k_norm.weight": f"{h}self_attn.k_norm.weight",
                f"{g}ffn_gate.weight": f"{h}mlp.gate_proj.weight",
                f"{g}ffn_up.weight": f"{h}mlp.up_proj.weight",
                f"{g}ffn_down.weight": f"{h}mlp.down_proj.weight",
                f"{g}attn_norm.weight": f"{h}input_layernorm.weight",
                f"{g}ffn_norm.weight": f"{h}post_attention_layernorm.weight",
            }
        )
    return m


class ModelVariant(StrEnum):
    """Available MediX R1 8B GGUF model variants for image to text."""

    MEDIX_R1_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-8B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: "MediX-R1-8B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_8B_Q4_K_M

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Install a wide-sig load_gguf_checkpoint wrapper that handles the
        # model_to_load kwarg that other loaders' narrow-sig patches ignore.
        # For config loading (return_tensors=False): delegate to whatever is
        # currently installed (safe for narrow-sig chain).
        # For tensor loading (return_tensors=True): load directly via GGUFReader,
        # bypassing the narrow-sig chain entirely.
        prev_load = gguf_utils.load_gguf_checkpoint

        def _qwen3vl_load_gguf(*args, **kw):
            model_to_load = kw.pop("model_to_load", None)
            return_tensors = kw.get("return_tensors", False)
            if len(args) > 1:
                return_tensors = args[1]

            # Config pass: call through the existing chain (narrow-sig compatible)
            config_kw = dict(kw)
            config_kw["return_tensors"] = False
            config_args = list(args)
            if len(config_args) > 1:
                config_args[1] = False
            result = prev_load(*config_args, **config_kw)

            # Translate qwen3vl flat config → nested Qwen3VLConfig structure
            if result.get("config", {}).get("model_type") == "qwen3vl":
                config = result["config"]
                text_config = {}
                for k in _TEXT_CONFIG_KEYS:
                    if k in config:
                        text_config[k] = config.pop(k)
                config["text_config"] = text_config
                config["model_type"] = "qwen3_vl"

            if return_tensors and model_to_load is not None:
                # Tensor pass: load directly from GGUF, bypassing narrow-sig chain
                gguf_path = args[0] if args else kw.get("gguf_checkpoint_path")
                result["tensors"] = _load_qwen3vl_tensors(gguf_path, model_to_load)

            return result

        gguf_utils.load_gguf_checkpoint = _qwen3vl_load_gguf
        for mod in (config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = _qwen3vl_load_gguf

        try:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs["gguf_file"] = gguf_file
            model_kwargs |= kwargs

            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
            )

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            gguf_utils.load_gguf_checkpoint = prev_load
            for mod in (config_utils, modeling_utils):
                if hasattr(mod, "load_gguf_checkpoint"):
                    mod.load_gguf_checkpoint = prev_load

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs


def _load_qwen3vl_tensors(gguf_path, model_to_load):
    """Load GGUF text-backbone tensors directly, bypassing the patching chain.

    Uses a hard-coded qwen3vl → HF parameter name mapping so we are not
    affected by the get_gguf_hf_weights_map multi-submodule traversal issue
    (visual encoder submodules would otherwise claim 'output_norm' first).
    """
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize
    from tqdm.auto import tqdm

    n_layers = model_to_load.config.text_config.num_hidden_layers
    tensor_key_mapping = _build_qwen3vl_gguf_tensor_mapping(n_layers)

    reader = GGUFReader(gguf_path)
    state_dict = {}
    for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
        name = tensor.name
        if name not in tensor_key_mapping:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        hf_name = tensor_key_mapping[name]
        state_dict[hf_name] = torch.from_numpy(np.copy(weights))

    return state_dict
