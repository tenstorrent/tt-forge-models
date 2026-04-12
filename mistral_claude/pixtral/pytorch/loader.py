# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)


class ModelLoader(ForgeModel):
    """Pixtral-12B multimodal (vision + language) model loader with tensor parallelism support.

    The vision tower uses standard separate Q/K/V/O projections and gate/up/down MLP.
    The language model (Mistral-Nemo-12B backbone) also uses standard separate projections.
    Both towers are sharded independently.
    """

    _VARIANTS = {}
    DEFAULT_VARIANT = None

    _PRETRAINED_MODEL_NAME = "mistralai/Pixtral-12B-2409"
    _MAX_LENGTH = 128

    sample_text = "Describe the image."

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None) -> ModelInfo:
        return ModelInfo(
            model="Pixtral",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._PRETRAINED_MODEL_NAME)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            self._PRETRAINED_MODEL_NAME, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        inputs = self.processor(
            text=self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._MAX_LENGTH,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._PRETRAINED_MODEL_NAME)
        return self.config

    def get_mesh_config(self, num_devices: int):
        if self.config is None:
            self.load_config()
        # For multimodal models the language model heads are the binding constraint;
        # vision tower heads (16) always divide evenly when language heads do.
        text_cfg = getattr(self.config, "text_config", self.config)
        heads = text_cfg.num_attention_heads
        if num_devices == 32:
            mesh_shape = (4, 8) if heads % 8 == 0 else (8, 4)
        elif heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif num_devices % 2 == 0 and heads % (num_devices // 2) == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot split {heads} language-model heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """Return per-tensor shard specs for tensor parallelism.

        Both towers are sharded:
        - Vision tower (mistral/pixtral/pytorch):
            model.model.vision_tower.transformer.layers[i]
              .attention.{q,k,v,o}_proj.weight
              .feed_forward.{gate,up,down}_proj.weight
        - Language model (Mistral backbone):
            model.model.language_model.layers[i]
              .self_attn.{q,k,v,o}_proj.weight
              .mlp.{gate,up,down}_proj.weight

        Args:
            model: Loaded model instance whose weights are to be sharded.
            strategy: "fsdp" shards across both mesh axes; "megatron" shards
                      on the "model" axis only (other axis replicated).
            batch_axis: Non-model mesh axis name for FSDP specs. Pass "data"
                        when input sharding uses a "data"-named mesh axis.

        Returns:
            Dict mapping weight tensors to shard spec tuples.
        """
        shard_specs = {}

        if strategy == "fsdp":
            self._shard_vision_fsdp(model, shard_specs, batch_axis)
            self._shard_language_fsdp(model, shard_specs, batch_axis)
        elif strategy == "megatron":
            self._shard_vision_megatron(model, shard_specs)
            self._shard_language_megatron(model, shard_specs)
        else:
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")

        return shard_specs

    # ------------------------------------------------------------------
    # Vision tower helpers
    # ------------------------------------------------------------------

    def _shard_vision_fsdp(self, model, shard_specs, batch_axis):
        for layer in model.model.vision_tower.transformer.layers:
            shard_specs[layer.attention.q_proj.weight] = ("model", batch_axis)
            shard_specs[layer.attention.k_proj.weight] = ("model", batch_axis)
            shard_specs[layer.attention.v_proj.weight] = ("model", batch_axis)
            shard_specs[layer.attention.o_proj.weight] = (batch_axis, "model")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", batch_axis)
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", batch_axis)
            shard_specs[layer.feed_forward.down_proj.weight] = (batch_axis, "model")

    def _shard_vision_megatron(self, model, shard_specs):
        for layer in model.model.vision_tower.transformer.layers:
            shard_specs[layer.attention.q_proj.weight] = ("model", None)
            shard_specs[layer.attention.k_proj.weight] = ("model", None)
            shard_specs[layer.attention.v_proj.weight] = ("model", None)
            shard_specs[layer.attention.o_proj.weight] = (None, "model")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", None)
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", None)
            shard_specs[layer.feed_forward.down_proj.weight] = (None, "model")

    # ------------------------------------------------------------------
    # Language model helpers
    # ------------------------------------------------------------------

    def _shard_language_fsdp(self, model, shard_specs, batch_axis):
        lm = model.model.language_model
        shard_specs[lm.embed_tokens.weight] = (None, batch_axis)
        shard_specs[lm.norm.weight] = (batch_axis,)
        shard_specs[model.lm_head.weight] = ("model", batch_axis)
        for layer in lm.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.up_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")
            shard_specs[layer.input_layernorm.weight] = (batch_axis,)
            shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

    def _shard_language_megatron(self, model, shard_specs):
        lm = model.model.language_model
        shard_specs[lm.embed_tokens.weight] = (None, None)
        shard_specs[lm.norm.weight] = (None,)
        shard_specs[model.lm_head.weight] = ("model", None)
        for layer in lm.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")
            shard_specs[layer.input_layernorm.weight] = (None,)
            shard_specs[layer.post_attention_layernorm.weight] = (None,)
