# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Omni model loader implementation for multimodal text generation.

Qwen/Qwen3-Omni-30B-A3B-Instruct is an any-to-any MoE model. Only the thinker
(``Qwen3OmniMoeThinkerForConditionalGeneration``) is loaded: the top-level
class is generation-only and also initializes the talker and waveform decoder,
neither of which is part of the text-logit forward the model runner compiles.

The default ``30B-A3B-Instruct`` variant brings up the text-only path
(input_ids + attention_mask only). The ``30B-A3B-Instruct-image`` variant
reuses the same checkpoint and the same verified shard spec, and routes
``load_inputs`` to the image+text path.
"""

import types
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


def _deepstack_process_tt(self, hidden_states, visual_pos_masks, visual_embeds):
    """Inject DeepStack features without boolean ``index_put_``.

    Equivalent to the upstream ``hidden_states[visual_pos_masks, :] +=
    visual_embeds``: the mask broadcasts over the hidden dimension and
    ``masked_scatter`` fills in row-major order, which is the order the
    boolean gather it replaces would produce.
    """
    mask = visual_pos_masks.to(hidden_states.device).unsqueeze(-1)
    embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    addend = torch.zeros_like(hidden_states).masked_scatter(mask, embeds)
    return hidden_states + addend


class ModelVariant(StrEnum):
    """Available Qwen3-Omni variants.

    ``30B-A3B-Instruct`` is the text-only path; ``30B-A3B-Instruct-image``
    drives the unified multimodal (vision) path on the same checkpoint.
    """

    QWEN3_OMNI_30B_A3B_INSTRUCT = "30B-A3B-Instruct"
    QWEN3_OMNI_30B_A3B_INSTRUCT_IMAGE = "30B-A3B-Instruct-image"


class ModelLoader(ForgeModel):
    """Load the Qwen3-Omni thinker for multimodal text generation."""

    MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    _VARIANTS = {
        ModelVariant.QWEN3_OMNI_30B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name=MODEL_NAME,
            max_length=256,
        ),
        ModelVariant.QWEN3_OMNI_30B_A3B_INSTRUCT_IMAGE: LLMModelConfig(
            pretrained_model_name=MODEL_NAME,
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_OMNI_30B_A3B_INSTRUCT

    sample_text = "Give a short introduction to large language models."
    sample_image_text = "Describe this image in one short sentence."
    sample_image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
    )
    MIN_PIXELS = 56 * 56
    # The checkpoint default is 12,845,056 pixels and produced a 25.3 GB
    # activation allocation on QB2. About 100K pixels keeps the real vision
    # path while limiting it to roughly 100 merged visual tokens per frame.
    MAX_PIXELS = 128 * 28 * 28

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3-Omni",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the multimodal processor for the text and image+text paths.

        The pixel caps are applied here rather than per call so both paths
        share one processor instance.

        Returns:
            The loaded processor instance (``Qwen3OmniMoeProcessor``).
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load only the thinker, the forward-capable text-logit submodel.

        The top-level Qwen3-Omni class is generation-only and also initializes
        the talker and waveform decoder. Those components are not part of the
        image-to-text forward compiled by the model runner.

        Args:
            dtype_override: Optional torch dtype to load weights in. The model
                ships in bfloat16; when not provided, transformers uses the
                checkpoint's native dtype.

        Returns:
            torch.nn.Module: The Qwen3-Omni thinker instance, with the
            DeepStack injection replaced by its traceable equivalent.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor()

        outer_config = AutoConfig.from_pretrained(pretrained_model_name)
        thinker_config = outer_config.thinker_config
        # The text model's forward fills use_cache from its own config via
        # @merge_with_config_defaults, so clearing it here is what keeps a
        # DynamicCache out of the traced forward.
        thinker_config.text_config.use_cache = False
        if self.num_layers is not None:
            thinker_config.text_config.num_hidden_layers = self.num_layers

        model_kwargs = {"config": thinker_config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()
        model.model._deepstack_process = types.MethodType(
            _deepstack_process_tt, model.model
        )

        self.config = model.config
        self.model = model
        return model

    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP.

        The shard spec in ``load_shard_spec`` shards weights only on the
        ``model`` axis, so the mesh is 1D over the available devices. Brought
        up and measured on 4 devices.
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel specs for the thinker text and vision towers.

        Vision blocks are column-parallel on the fused qkv and the first MLP
        projection, row-parallel on the output projections; the patch mergers
        follow the same pattern. ``visual.merger_list`` holds the DeepStack
        mergers (``visual.deepstack_merger_list`` is a property alias for it).
        The text stack follows the same Megatron mapping as the other Qwen3 MoE
        loaders. Per-layer norms, the router and the audio tower are left
        replicated.
        """
        shard_specs = {}

        visual = model.visual
        for block in visual.blocks:
            shard_specs[block.attn.qkv.weight] = ("model", "batch")
            shard_specs[block.attn.qkv.bias] = ("model",)
            shard_specs[block.attn.proj.weight] = ("batch", "model")
            shard_specs[block.mlp.linear_fc1.weight] = ("model", "batch")
            shard_specs[block.mlp.linear_fc1.bias] = ("model",)
            shard_specs[block.mlp.linear_fc2.weight] = ("batch", "model")

        for merger in [visual.merger, *visual.merger_list]:
            shard_specs[merger.mlp[0].weight] = ("model", "batch")
            shard_specs[merger.mlp[0].bias] = ("model",)
            shard_specs[merger.mlp[2].weight] = ("batch", "model")

        for layer in model.model.layers:
            # Routed expert tensors are added by get_tt_moe_shard_specs when
            # inject_custom_moe is enabled in the runner configuration.
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("batch", "model")
            shard_specs[attn.k_proj.weight] = ("batch", "model")
            shard_specs[attn.v_proj.weight] = ("batch", "model")
            shard_specs[attn.o_proj.weight] = ("model", "batch")

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        include_image: bool = False,
        image_url: Optional[str] = None,
    ):
        """Build the text or image prefill selected by the variant.

        Defaults to the text-only path (input_ids + attention_mask). Set
        ``include_image`` to ``True``, or use the ``-image`` variant, to get
        the unified multimodal (vision) path: the processor turns the rendered
        chat template plus a PIL image into ``input_ids`` (with the image token
        span), ``attention_mask``, ``pixel_values`` merged patches and
        ``image_grid_thw``, with the visual token count following
        ``MAX_PIXELS``. Only ``pixel_values`` is cast to ``dtype_override``;
        the id/mask tensors stay integer.

        Returns:
            dict: {"input_ids", "attention_mask"} for the text path, or
            {input_ids, attention_mask, pixel_values, image_grid_thw,
            position_ids} for the image+text path.
        """
        if batch_size not in (None, 1):
            raise ValueError("Qwen3-Omni multimodal bring-up supports batch_size=1")
        if self.processor is None:
            self._load_processor()

        # Variant-driven dispatch: the runner calls load_inputs with only
        # dtype_override/batch_size, so the -image variant selects its
        # modality here (an explicit include_image kwarg still wins).
        if not include_image:
            include_image = (
                self._variant == ModelVariant.QWEN3_OMNI_30B_A3B_INSTRUCT_IMAGE
            )

        if include_image:
            assert self.model is not None, (
                "load_model must run before load_inputs: the MRoPE positions "
                "are precomputed with the model's get_rope_index"
            )

            image_path = get_file(image_url or self.sample_image_url)
            image = Image.open(image_path).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url or self.sample_image_url},
                        {"type": "text", "text": prompt or self.sample_image_text},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            if dtype_override is not None and "pixel_values" in inputs:
                inputs["pixel_values"] = cast_input_to_type(
                    inputs["pixel_values"], dtype_override
                )

            # Upstream builds the multimodal MRoPE positions inside forward
            # using torch.Tensor(t_index) (get_llm_pos_ids_for_vision), the
            # legacy constructor that always allocates on CPU; the h/w indexes
            # next to it are XLA tensors, so the torch.stack of the three fails
            # to compile. Compute the static prefill positions here, with the
            # arguments forward would use, while every processor output is
            # still on CPU; forward only recomputes them when position_ids is
            # None.
            position_ids, _ = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs["attention_mask"],
            )
            inputs["position_ids"] = position_ids
            return inputs

        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt or self.sample_text}],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return self.processor.tokenizer([text], padding=True, return_tensors="pt")

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        ).thinker_config
        return self.config
