# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Magi (The Manga Whisperer) model loader implementation for manga panel, character,
and text detection.
"""

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, ConditionalDetrConfig
from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_relative_import_files

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Magi model variants."""

    MAGI = "magi"


class ModelLoader(ForgeModel):
    """Magi (The Manga Whisperer) model loader for manga detection and transcription."""

    _VARIANTS = {
        ModelVariant.MAGI: ModelConfig(
            pretrained_model_name="ragavsachdeva/magi",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAGI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Magi",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _apply_magi_patches(self, pretrained_model_name: str) -> None:
        # Load remote config module into cache so we can patch before model instantiation.
        AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        config_module_key = next(
            k for k in sys.modules if "configuration_magi" in k and "ragavsachdeva" in k
        )
        modelling_module_key = config_module_key.replace("configuration_magi", "modelling_magi")
        modelling_module = importlib.import_module(modelling_module_key)

        def _set_module_hash(module_key: str, module) -> None:
            # Compute the same hash used by get_class_in_module so the module
            # is not re-executed (which would overwrite our patches).
            file_path = Path(HF_MODULES_CACHE) / (module_key.replace(".", "/") + ".py")
            files = [file_path] + sorted(
                map(Path, get_relative_import_files(str(file_path)))
            )
            module.__transformers_module_hash__ = hashlib.sha256(
                b"".join(bytes(f) + f.read_bytes() for f in files)
            ).hexdigest()

        config_module = sys.modules[config_module_key]
        _set_module_hash(config_module_key, config_module)
        _set_module_hash(modelling_module_key, modelling_module)

        # Fix 1: In transformers 5.x, PretrainedConfig.from_dict(dict) always
        # returns a base PretrainedConfig regardless of model_type.  The magi
        # config relies on typed sub-configs, so we reconstruct detection_model_config
        # as a ConditionalDetrConfig (which handles the backbone kwarg consolidation).
        MagiConfig = config_module.MagiConfig
        _orig_config_init = MagiConfig.__init__

        def _patched_config_init(
            self, detection_model_config=None, ocr_model_config=None, crop_embedding_model_config=None, **kw
        ):
            det_config = (
                ConditionalDetrConfig(**detection_model_config)
                if isinstance(detection_model_config, dict)
                else detection_model_config
            )
            _orig_config_init(
                self,
                detection_model_config=None,
                ocr_model_config=ocr_model_config,
                crop_embedding_model_config=crop_embedding_model_config,
                **kw,
            )
            if det_config is not None:
                self.detection_model_config = det_config

        MagiConfig.__init__ = _patched_config_init

        # Fix 2: MagiModel.__init__ does not call self.post_init().
        # In transformers 5.x, post_init() is required to set all_tied_weights_keys.
        MagiModel = modelling_module.MagiModel
        _orig_model_init = MagiModel.__init__

        def _patched_model_init(self, config):
            _orig_model_init(self, config)
            self.post_init()

        MagiModel.__init__ = _patched_model_init

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._apply_magi_patches(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"].convert("L").convert("RGB")
        images = [np.array(image)] * batch_size

        inputs = self.model.processor.preprocess_inputs_for_detection(images)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
