# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.8 model loader implementation for causal language modeling.

Qwen 3.8-27B is architecturally identical to Qwen 3.5-27B -- the published
checkpoints keep ``model_type: qwen3_5`` -- so this loader is a thin
subclass of the Qwen 3.5 causal-LM loader. Only variant registration, an
architecture guard, and Qwen 3.8-specific checkpoint auditing live here;
tokenization, model loading, sharding and mixed-precision discovery are
inherited.
"""
import logging
import os
from dataclasses import replace
from typing import Optional

from transformers import AutoConfig

from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....qwen_3_5.causal_lm.pytorch.loader import (
    ModelLoader as Qwen35ModelLoader,
)

logger = logging.getLogger(__name__)


class ModelVariant(StrEnum):
    """Available Qwen 3.8 model variants for causal language modeling."""

    QWEN_3_8_27B = "27B"


class ModelLoader(Qwen35ModelLoader):
    """Qwen 3.8 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_8_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.8-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_8_27B

    # The published Qwen3.8-27B checkpoints reuse the Qwen 3.5 architecture
    # (config ``model_type: qwen3_5``): a 64-layer hybrid of Gated DeltaNet
    # linear-attention and full-attention layers with an untied lm_head.
    _EXPECTED_MODEL_TYPE = "qwen3_5"

    # Checkpoint tensors outside the causal LM: the multi-token-prediction
    # heads (``mtp.*``) and the visual tower (``visual.*``). The checkpoints
    # declare these in ``_keys_to_ignore_on_load_unexpected``; load_model
    # verifies that none survive into the instantiated model.
    _IGNORED_KEY_PREFIXES = ("mtp.", "visual.")

    # Qwen 3.8-specific config keys. They exist only in Qwen 3.8 configs and
    # are audited (logged) only -- no loader behavior derives from them.
    # (``text_config`` nesting itself is audited alongside them below.)
    _QWEN38_ONLY_CONFIG_KEYS = ("output_gate_type", "language_model_only")

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ) -> None:
        super().__init__(variant, num_layers)
        # Optional local-checkpoint override; by default the loader resolves
        # the published Hugging Face repo id like every other variant.
        env_path = os.environ.get("QWEN_3_8_MODEL_PATH")
        if env_path:
            self._variant_config = replace(
                self._variant_config, pretrained_model_name=env_path
            )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen 3.8",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _guard_model_type(self, config) -> None:
        """Fail fast unless the checkpoint is the expected qwen3_5 architecture."""
        text_cfg = getattr(config, "text_config", config)
        model_type = getattr(config, "model_type", None) or getattr(
            text_cfg, "model_type", None
        )
        if model_type != self._EXPECTED_MODEL_TYPE:
            raise ValueError(
                f"{type(self).__name__} expects a checkpoint with model_type "
                f"{self._EXPECTED_MODEL_TYPE!r} (the architecture Qwen 3.8 "
                f"shares with Qwen 3.5), got {model_type!r}."
            )

    def _audit_config(self, config) -> None:
        """Log Qwen 3.8-specific config keys; no behavior depends on them."""
        text_cfg = getattr(config, "text_config", None)
        for key in self._QWEN38_ONLY_CONFIG_KEYS:
            for scope, cfg in (("config", config), ("text_config", text_cfg)):
                if cfg is not None and getattr(cfg, key, None) is not None:
                    logger.info(
                        "Qwen 3.8 config audit: %s.%s = %r (informational)",
                        scope,
                        key,
                        getattr(cfg, key),
                    )
        if text_cfg is not None:
            logger.info(
                "Qwen 3.8 config audit: nested text_config present; the outer "
                "config mirrors it and the model builder reads the nested one"
            )

    def _report_checkpoint_coverage(self, model) -> None:
        """Verify ignored-prefix tensors are absent and report tensor counts.

        Mirrors the checkpoints' ``_keys_to_ignore_on_load_unexpected``
        semantics: ``mtp.*`` / ``visual.*`` tensors must not survive into the
        instantiated model.
        """
        state = model.state_dict()
        leaked = [k for k in state if k.startswith(self._IGNORED_KEY_PREFIXES)]
        if leaked:
            logger.warning(
                "Qwen 3.8 checkpoint coverage: %d tensors with ignored "
                "prefixes %s survived load, first few: %s",
                len(leaked),
                self._IGNORED_KEY_PREFIXES,
                leaked[:10],
            )
        else:
            logger.info(
                "Qwen 3.8 checkpoint coverage: %d tensors loaded, 0 with "
                "ignored prefixes %s",
                len(state),
                self._IGNORED_KEY_PREFIXES,
            )

    def load_config(self):
        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        self._guard_model_type(config)
        self._audit_config(config)
        self.config = config
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        self._guard_model_type(self.load_config())
        model = super().load_model(dtype_override=dtype_override, **kwargs)
        self._audit_config(model.config)
        self._report_checkpoint_coverage(model)
        return model
