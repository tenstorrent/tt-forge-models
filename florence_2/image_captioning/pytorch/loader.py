# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Florence-2 image captioning model loader implementation (PyTorch).
"""

import glob
import importlib.util
import os
import sys

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Florence2ForConditionalGeneration,
)
from typing import Optional
from PIL import Image

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
from ....tools.utils import get_file

_SD3_REPO = "gokaygokay/Florence-2-SD3-Captioner"


def _find_sd3_cache_dirs():
    """Find all possible cache directories for SD3-Captioner files."""
    from huggingface_hub import constants

    hf_home = constants.HF_HOME
    dirs = []
    for pattern in [
        os.path.join(
            hf_home,
            "modules",
            "transformers_modules",
            "gokaygokay",
            "Florence_hyphen_2_hyphen_SD3_hyphen_Captioner",
            "*",
        ),
        os.path.join(
            hf_home,
            "hub",
            "models--gokaygokay--Florence-2-SD3-Captioner",
            "snapshots",
            "*",
        ),
    ]:
        dirs.extend(glob.glob(pattern))
    return dirs


def _patch_sd3_config():
    """Patch the config module for transformers 5.x compatibility."""
    try:
        AutoConfig.from_pretrained(_SD3_REPO, trust_remote_code=True)
    except (AttributeError, Exception):
        pass
    for name, mod in sys.modules.items():
        if (
            "Florence_hyphen_2_hyphen_SD3_hyphen_Captioner" in name
            and "configuration" in name
        ):
            if hasattr(mod, "Florence2LanguageConfig"):
                mod.Florence2LanguageConfig.forced_bos_token_id = None


def _patch_sd3_modeling_file():
    """Patch the cached modeling file for transformers 5.x compatibility."""
    for cache_dir in _find_sd3_cache_dirs():
        model_file = os.path.join(cache_dir, "modeling_florence2.py")
        if not os.path.exists(model_file):
            continue
        with open(model_file) as f:
            source = f.read()
        modified = False

        old_linspace = (
            "[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]"
        )
        new_linspace = (
            "[drop_path_rate * i / max(sum(depths) * 2 - 1, 1)"
            " for i in range(sum(depths) * 2)]"
        )
        if old_linspace in source:
            source = source.replace(old_linspace, new_linspace)
            modified = True

        old_output_fields = (
            "    last_hidden_state: torch.FloatTensor = None\n"
            "    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None\n"
            "    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    encoder_last_hidden_state: Optional[torch.FloatTensor] = None\n"
            "    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None"
        )
        new_output_fields = (
            "    loss: Optional[torch.FloatTensor] = None\n"
            "    logits: torch.FloatTensor = None\n"
            "    last_hidden_state: torch.FloatTensor = None\n"
            "    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None\n"
            "    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    encoder_last_hidden_state: Optional[torch.FloatTensor] = None\n"
            "    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None\n"
            "    image_hidden_states: Optional[torch.FloatTensor] = None"
        )
        if old_output_fields in source:
            source = source.replace(old_output_fields, new_output_fields)
            modified = True

        if modified:
            with open(model_file, "w") as f:
                f.write(source)


def _load_sd3_captioner_processor():
    """Load the SD3-Captioner processor with compatibility patches."""
    tok = AutoTokenizer.from_pretrained(_SD3_REPO)
    tok.__dict__["additional_special_tokens"] = []
    ip = AutoImageProcessor.from_pretrained(_SD3_REPO, use_fast=False)

    proc_file = None
    for cache_dir in _find_sd3_cache_dirs():
        candidate = os.path.join(cache_dir, "processing_florence2.py")
        if os.path.exists(candidate):
            proc_file = candidate
            break

    if proc_file is None:
        raise RuntimeError("Could not find SD3-Captioner processing_florence2.py")

    spec = importlib.util.spec_from_file_location("sd3_processing", proc_file)
    proc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc_mod)
    return proc_mod.Florence2Processor(ip, tok)


def _load_sd3_captioner_model(model_kwargs):
    """Load the SD3-Captioner model with compatibility patches."""
    _patch_sd3_config()
    _patch_sd3_modeling_file()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            _SD3_REPO,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
    except AttributeError:
        for name, mod in sys.modules.items():
            if (
                "Florence_hyphen_2_hyphen_SD3_hyphen_Captioner" in name
                and "modeling" in name
            ):
                if hasattr(mod, "Florence2ForConditionalGeneration"):
                    mod.Florence2ForConditionalGeneration._supports_sdpa = True
        model = AutoModelForCausalLM.from_pretrained(
            _SD3_REPO,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
    return model


class ModelVariant(StrEnum):
    """Available Florence-2 image captioning model variants."""

    BASE = "Base"
    BASE_FT = "Base_Ft"
    LARGE = "Large"
    SD3_CAPTIONER = "SD3-Captioner"
    COMMUNITY_BASE_FT = "Community_Base_Ft"


_DESCRIPTION_VARIANTS = {ModelVariant.SD3_CAPTIONER}

_COMMUNITY_VARIANTS = {ModelVariant.SD3_CAPTIONER}


class ModelLoader(ForgeModel):
    """Florence-2 image captioning model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-base",
        ),
        ModelVariant.BASE_FT: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-base-ft",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="microsoft/Florence-2-large",
        ),
        ModelVariant.SD3_CAPTIONER: ModelConfig(
            pretrained_model_name=_SD3_REPO,
        ),
        ModelVariant.COMMUNITY_BASE_FT: ModelConfig(
            pretrained_model_name="florence-community/Florence-2-base-ft",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Florence-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_CAPT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_community(self):
        return self._variant in _COMMUNITY_VARIANTS

    def _load_processor(self):
        if self._is_community():
            self.processor = _load_sd3_captioner_processor()
        else:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self._is_community():
            model = _load_sd3_captioner_model(model_kwargs)
        else:
            model = Florence2ForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                attn_implementation="eager",
                **model_kwargs,
            )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = (
            "<DESCRIPTION>" if self._variant in _DESCRIPTION_VARIANTS else "<CAPTION>"
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        decoder_start_token_id = self.processor.tokenizer.bos_token_id or 2
        inputs["decoder_input_ids"] = torch.full(
            (1, 1), decoder_start_token_id, dtype=torch.long
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.processor is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.processor.decode(token_ids[0], skip_special_tokens=True)
