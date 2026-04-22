# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIDO.RNA model loader implementation for embedding generation on RNA sequences.
"""
import sys
import types
from typing import Optional

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


def _mock_lamindb_setup():
    """Pre-populate sys.modules with a stub lamindb_setup.

    modelgenerator -> bionty -> lamindb_setup -> Django is the import chain.
    Django's lazy proxy objects conflict with the test runner's module cleanup,
    so we short-circuit the chain here. The actual model (AIDO.RNA) never uses
    lamindb functionality; bionty only checks _check_instance_setup() which
    returns False when no lamin instance is configured.
    """
    if "lamindb_setup" in sys.modules:
        return

    from pathlib import Path

    def _make(name, parent=None):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        sys.modules[name] = mod
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], mod)
        return mod

    pkg = _make("lamindb_setup")
    pkg.__version__ = "1.9.1"

    check = _make("lamindb_setup._check_setup", pkg)
    check._check_instance_setup = lambda *a, **kw: False

    core = _make("lamindb_setup.core", pkg)

    upath = _make("lamindb_setup.core.upath", core)
    upath.UPath = Path

    errors = _make("lamindb_setup.errors", pkg)
    errors.InstanceNotSetupError = type("InstanceNotSetupError", (Exception,), {})

    _make("lamindb_setup.types", pkg)
    _make("lamindb_setup._django", pkg)
    _make("lamindb_setup._connect_instance", pkg)
    _make("lamindb_setup._delete", pkg)
    _make("lamindb_setup._disconnect", pkg)
    _make("lamindb_setup._entry_points", pkg)


class ModelVariant(StrEnum):
    """Available AIDO.RNA model variants."""

    AIDO_RNA_1B600M = "genbio-ai/AIDO.RNA-1.6B"
    AIDO_RNA_650M_CDS = "genbio-ai/AIDO.RNA-650M-CDS"


class ModelLoader(ForgeModel):
    """AIDO.RNA model loader for embedding generation on RNA sequences."""

    _VARIANTS = {
        ModelVariant.AIDO_RNA_1B600M: ModelConfig(
            pretrained_model_name="genbio-ai/AIDO.RNA-1.6B",
        ),
        ModelVariant.AIDO_RNA_650M_CDS: ModelConfig(
            pretrained_model_name="genbio-ai/AIDO.RNA-650M-CDS",
        ),
    }

    _VARIANT_TO_BACKBONE = {
        ModelVariant.AIDO_RNA_1B600M: "aido_rna_1b600m",
        ModelVariant.AIDO_RNA_650M_CDS: "aido_rna_650m_cds",
    }

    DEFAULT_VARIANT = ModelVariant.AIDO_RNA_1B600M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_instance = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AIDO.RNA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _mock_lamindb_setup()

        from modelgenerator.tasks import Embed

        backbone = self._VARIANT_TO_BACKBONE[self._variant]
        model = Embed.from_config({"model.backbone": backbone}).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model_instance = model
        return model

    def load_inputs(self, dtype_override=None):
        if self._model_instance is None:
            self.load_model(dtype_override=dtype_override)

        # Sample RNA sequence
        rna_sequence = "ACGUACGUACGUACGU"

        transformed_batch = self._model_instance.transform(
            {"sequences": [rna_sequence]}
        )

        return (transformed_batch,)
