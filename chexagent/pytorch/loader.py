# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXagent model loader implementation for chest X-ray vision-language tasks.
"""

import os
import re
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _apply_transformers5_compat() -> None:
    """Stub is_tf_available removed from transformers 5.x.

    transformers 5.x removed TensorFlow support and dropped is_tf_available
    from transformers.utils and transformers.utils.import_utils.  The
    CheXagent tokenizer's remote code imports it and uses it in a
    TYPE_CHECKING guard.  Adding a False-returning stub here allows:
      - check_imports() to skip the guarded `import tensorflow` block, and
      - the `from transformers.utils import is_tf_available` line to succeed.
    """
    import transformers.utils as tu
    import transformers.utils.import_utils as tiu

    if not hasattr(tiu, "is_tf_available"):
        tiu.is_tf_available = lambda: False
        tu.is_tf_available = lambda: False


def _patch_modeling_visual(model_name: str) -> None:
    """Remove strict transformers==4.40.0 assertion from modeling_visual.py.

    The file shipped with the model hard-asserts the exact transformers
    version at module import time.  The assertion is purely a developer
    warning and has no functional effect; removing it lets the model load
    with transformers 5.x.
    """
    from huggingface_hub import hf_hub_download

    # Patch the HF hub blob (and follow the symlink to the real file).
    hub_path = Path(hf_hub_download(model_name, "modeling_visual.py"))
    real_path = Path(os.path.realpath(hub_path))
    _remove_version_assert(real_path)

    # Also patch the transformers_modules cache copy when it already exists,
    # so subsequent runs (which skip the blob-copy step) still use the fix.
    _patch_modules_cache_copy(model_name, "modeling_visual.py")


def _remove_version_assert(path: Path) -> None:
    content = path.read_text()
    if 'assert transformers.__version__' not in content:
        return
    patched = re.sub(
        r'^assert transformers\.__version__ == "[^"]+",.*\n',
        "",
        content,
        flags=re.MULTILINE,
    )
    path.write_text(patched)


def _patch_modules_cache_copy(model_name: str, filename: str) -> None:
    """Patch a file in the transformers_modules cache, if it is already there."""
    from huggingface_hub import hf_hub_download

    try:
        hub_path = Path(hf_hub_download(model_name, "config.json"))
        snapshot_hash = hub_path.parent.name
    except Exception:
        return

    org, model_id = model_name.split("/", 1)
    encoded_model = model_id.replace("-", "_hyphen_")
    modules_file = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
        / org
        / encoded_model
        / snapshot_hash
        / filename
    )
    if modules_file.exists():
        _remove_version_assert(modules_file)


class ModelVariant(StrEnum):
    """Available CheXagent model variants."""

    CHEXAGENT_2_3B = "chexagent_2_3b"


class ModelLoader(ForgeModel):
    """CheXagent model loader implementation for chest X-ray vision-language tasks."""

    _VARIANTS = {
        ModelVariant.CHEXAGENT_2_3B: ModelConfig(
            pretrained_model_name="StanfordAIMI/CheXagent-2-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEXAGENT_2_3B

    sample_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )
    sample_text = "Describe the findings in this chest X-ray."
    sample_system_prompt = "You are a helpful assistant."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CheXagent",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        _apply_transformers5_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        _apply_transformers5_compat()
        _patch_modeling_visual(pretrained_model_name)

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        query = self.tokenizer.from_list_format(
            [
                {"image": self.sample_image_url},
                {"text": self.sample_text},
            ]
        )

        conv = [
            {"from": "system", "value": self.sample_system_prompt},
            {"from": "human", "value": query},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )

        return {"input_ids": input_ids}
