# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 Maverick image+text model loader (AutoProcessor + AutoModelForImageTextToText).
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
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


class ModelVariant(StrEnum):
    """Available Llama 4 Maverick model variants."""

    LLAMA_4_MAVERICK_17B_128E_INSTRUCT = "Maverick-17B-128E-Instruct"
    LLAMA_4_SCOUT_17B_16E = "Llama-4-Scout-17B-16E"


class ModelLoader(ForgeModel):
    """Llama 4 Maverick multimodal loader (image + text)."""

    _VARIANTS = {
        ModelVariant.LLAMA_4_MAVERICK_17B_128E_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            max_length=256,
        ),
        ModelVariant.LLAMA_4_SCOUT_17B_16E: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-4-Scout-17B-16E",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_4_MAVERICK_17B_128E_INSTRUCT

    sample_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model metadata for discovery and reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama-4",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load AutoProcessor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Llama 4 Maverick with AutoModelForImageTextToText."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )

        # Workaround: Llama4VisionRotaryEmbedding.freqs_ci is computed in
        # __init__ from config params and assigned as a plain tensor attribute
        # (not a checkpoint weight). With low_cpu_mem_usage=True (default),
        # __init__ runs on the meta device so freqs_ci stays meta and
        # .to(device) fails at inference. Re-initialize after loading using
        # model.vision_model.config which is always a real CPU object.
        from transformers.models.llama4.modeling_llama4 import (
            Llama4VisionRotaryEmbedding,
        )

        vision_config = model.vision_model.config
        for module in model.modules():
            if isinstance(module, Llama4VisionRotaryEmbedding):
                Llama4VisionRotaryEmbedding.__init__(module, vision_config)

        model.eval()
        self.config = model.config
        self.model = model
        print("model", model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build sample multimodal inputs via processor.apply_chat_template (matches HF recipe)."""
        if self.processor is None:
            self._load_processor()

        inputs = self.processor.apply_chat_template(
            self.sample_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = (
                    inputs[key].repeat_interleave(batch_size, dim=0).contiguous()
                )
        return inputs

    def get_mesh_config(self, num_devices: int):
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (2, num_devices // 2)

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard specs for Llama4ForConditionalGeneration.
        Mirrors patterns from llama/causal_lm (text), pixtral (vision tower), and
        gpt_oss (MoE expert tensors). Vision blocks use Linear with bias; text
        attention uses bias=False; MoE layers use Llama4TextExperts Parameters.
        """
        shard_specs = {}

        vm = model.vision_model

        shard_specs[vm.patch_embedding.linear.weight] = ("model", "batch")

        for layer in vm.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.bias] = ("batch",)

            shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
            shard_specs[layer.mlp.fc2.weight] = ("batch", "model")
            shard_specs[layer.mlp.fc1.bias] = ("model",)
            shard_specs[layer.mlp.fc2.bias] = ("batch",)

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.input_layernorm.bias] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.bias] = ("batch",)

        shard_specs[vm.layernorm_pre.weight] = ("batch",)
        shard_specs[vm.layernorm_pre.bias] = ("batch",)
        shard_specs[vm.layernorm_post.weight] = ("batch",)
        shard_specs[vm.layernorm_post.bias] = ("batch",)

        adapter_mlp = vm.vision_adapter.mlp
        shard_specs[adapter_mlp.fc1.weight] = ("model", "batch")
        shard_specs[adapter_mlp.fc2.weight] = ("batch", "model")

        shard_specs[model.multi_modal_projector.linear_1.weight] = ("model", "batch")

        lm = model.language_model
        text = lm.model
        shard_specs[lm.lm_head.weight] = ("model", "batch")

        for layer in text.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            ff = layer.feed_forward
            if hasattr(ff, "router"):
                shard_specs[ff.router.weight] = (None, "batch")
                shard_specs[ff.experts.gate_up_proj] = ("model", "batch", None)
                shard_specs[ff.experts.down_proj] = ("model", None, "batch")
                se = ff.shared_expert
                shard_specs[se.gate_proj.weight] = ("model", "batch")
                shard_specs[se.up_proj.weight] = ("model", "batch")
                shard_specs[se.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[ff.gate_proj.weight] = ("model", "batch")
                shard_specs[ff.up_proj.weight] = ("model", "batch")
                shard_specs[ff.down_proj.weight] = ("batch", "model")

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs
