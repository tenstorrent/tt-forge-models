# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling.
"""
import torch
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


def _patch_transformers_falcon_h1_gguf():
    """Monkey-patch transformers to add falcon-h1 GGUF architecture support.

    Transformers 5.x has FalconH1ForCausalLM but lacks GGUF loading support
    for the falcon-h1 architecture. The gguf library already knows about
    falcon-h1 tensor names, so we only need to bridge transformers' config
    and tokenizer processing layer.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "falcon-h1" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "rope.freq_base": "rope_theta",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.inner_size": "mamba_d_ssm",
        "ssm.state_size": "mamba_d_state",
        "ssm.group_count": "mamba_n_groups",
        "ssm.time_step_rank": None,
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    for arch_key in ("falcon-h1", "falcon_h1"):
        if arch_key not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[arch_key] = GGUFGPTConverter

    # Patch load_gguf_checkpoint to convert "falcon-h1" model_type to "falcon_h1"
    # and set rope_parameters. We patch all modules that use a local binding.
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "falcon-h1":
            config["model_type"] = "falcon_h1"
            rope_theta = config.pop("rope_theta", None)
            if rope_theta is not None:
                config["rope_parameters"] = {
                    "rope_theta": float(rope_theta),
                    "rope_type": "default",
                }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_falcon_h1_gguf()


def _load_falcon_h1_direct(
    pretrained_model_name, gguf_file, gguf_path, dtype_override, num_layers=None
):
    """Load FalconH1 directly from GGUF, bypassing AutoModelForCausalLM.from_pretrained.

    Other loaders in the test suite overwrite gguf_utils.load_gguf_checkpoint
    with functions that strip model_to_load, breaking from_pretrained's tensor
    loading for every model.  We bypass that path here by:
      1. Loading the config via AutoConfig (config-only GGUF read; works fine).
      2. Creating the model directly from the config.
      3. Loading tensors directly via gguf-py.
    """
    import numpy as np
    from gguf import GGUFReader, dequantize
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers import FalconH1ForCausalLM

    # Step 1: Config
    config = AutoConfig.from_pretrained(pretrained_model_name, gguf_file=gguf_file)
    if num_layers is not None:
        config.num_hidden_layers = num_layers

    # Step 2: Open GGUF and correct vocab_size from the actual embedding tensor shape.
    # GGUF metadata vocab_size may be the base vocab only; the embedding tensor
    # (which includes special tokens) is the authoritative source. GGUF stores
    # shapes reversed vs PyTorch: token_embd.weight shape is (hidden_size, vocab_size),
    # so shape[-1] gives the true vocab_size.
    reader = GGUFReader(gguf_path, "r")
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            actual_vocab = int(tensor.shape[-1])
            if actual_vocab != config.vocab_size:
                config.vocab_size = actual_vocab
            break

    # Step 3: Model with correct architecture (weights loaded below)
    torch_dtype = dtype_override if dtype_override is not None else torch.bfloat16
    model = FalconH1ForCausalLM(config).to(dtype=torch_dtype)

    # Step 4: Build GGUF tensor name → HF state-dict name mapping.
    # Pass model_type="falcon-h1" explicitly since MODEL_ARCH_NAMES uses the
    # hyphenated form, while config.model_type is "falcon_h1" (underscore).
    processor = gguf_utils.TensorProcessor(config={})
    num_hidden = config.num_hidden_layers
    tensor_key_mapping = gguf_utils.get_gguf_hf_weights_map(
        model, processor, model_type="falcon-h1", num_layers=num_hidden
    )

    # Step 5: Read, dequantize, and load tensors from the GGUF file.
    loaded = {}
    state_dict = model.state_dict()
    for tensor in reader.tensors:
        gguf_name = tensor.name
        if gguf_name not in tensor_key_mapping:
            continue
        hf_name = tensor_key_mapping[gguf_name]
        if hf_name not in state_dict:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.copy(weights)).to(dtype=torch_dtype)
        expected_shape = state_dict[hf_name].shape
        if t.ndim >= 2 and t.shape == tuple(reversed(expected_shape)):
            t = t.T
        if t.shape == expected_shape:
            loaded[hf_name] = t

    model.load_state_dict(loaded, strict=False)
    return model.eval()


class ModelVariant(StrEnum):
    """Available Falcon-H1-34B-Instruct GGUF model variants for causal language modeling."""

    FALCON_H1_34B_INSTRUCT_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-34B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M

    GGUF_FILE = "Falcon-H1-34B-Instruct-Q4_K_M.gguf"

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
            model="Falcon-H1-34B-Instruct GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )

        model = _load_falcon_h1_direct(
            pretrained_model_name=pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            gguf_path=gguf_path,
            dtype_override=dtype_override,
            num_layers=self.num_layers,
        )

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
