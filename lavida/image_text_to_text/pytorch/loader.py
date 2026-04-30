# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA model loader implementation for image-text-to-text tasks.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


class ModelVariant(StrEnum):
    """Available LaViDa-LLaDA model variants for image-text-to-text tasks."""

    LAVIDA_LLADA_V1_0_INSTRUCT = "lavida_llada_v1_0_instruct"


class ModelLoader(ForgeModel):
    """LaViDa-LLaDA model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT: LLMModelConfig(
            pretrained_model_name="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT

    sample_image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LaViDa-LLaDA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LaViDa-LLaDA model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Get remote model classes for patching.
        LLaDAModelLM = get_class_from_dynamic_module(
            "modeling_lavida.LLaDAModelLM",
            pretrained_model_name,
        )
        SigLipVisionConfig = get_class_from_dynamic_module(
            "modeling_lavida.SigLipVisionConfig",
            pretrained_model_name,
        )

        # Fix 1: LLaDAModelLM.tie_weights only accepts `self` but transformers 5.x calls
        # tie_weights(missing_keys=..., recompute_mapping=False) internally during from_pretrained.
        def _tie_weights_compat(self, missing_keys=None, recompute_mapping=True):
            if self.config.weight_tying:
                self.model.transformer.ff_out = self.model.transformer.wte

        LLaDAModelLM.tie_weights = _tie_weights_compat

        # Fix 1b: LlavaLladaForMaskedDiffusion.forward calls super().forward() with
        # position_ids, prompt_len, and num_items_in_batch kwargs, but LLaDAModelLM.forward
        # does not accept those parameters (they are silently unused for this architecture).
        _orig_lm_forward = LLaDAModelLM.forward

        def _lm_forward_compat(self, input_ids=None, inputs_embeds=None,
                               attention_mask=None, attention_bias=None,
                               past_key_values=None, labels=None, use_cache=None,
                               output_attentions=None, output_hidden_states=None,
                               return_dict=None, cache_position=None, **kwargs):
            return _orig_lm_forward(
                self, input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, attention_bias=attention_bias,
                past_key_values=past_key_values, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
                cache_position=cache_position,
            )

        LLaDAModelLM.forward = _lm_forward_compat

        SigLipVisionTower = get_class_from_dynamic_module(
            "modeling_lavida.SigLipVisionTower",
            pretrained_model_name,
        )

        # Fix 2: SigLipVisionConfig.from_pretrained calls cls._set_token_in_kwargs() which
        # was removed in transformers 5.x.  Add a no-op shim so vision tower loading works.
        if not hasattr(SigLipVisionConfig, "_set_token_in_kwargs"):

            @classmethod
            def _set_token_in_kwargs_noop(cls, kwargs, token=None):
                pass

            SigLipVisionConfig._set_token_in_kwargs = _set_token_in_kwargs_noop

        # Fix 3: In transformers 5.x, from_pretrained always runs model __init__ inside a
        # torch.device("meta") context.  SigLipVisionTower.__init__ calls
        # SigLipVisionModel.from_pretrained() when delay_load=False, which is forbidden
        # inside the meta context (check_and_set_device_map raises RuntimeError).
        # Set delay_load=True so that the vision tower is not loaded during __init__.
        # After from_pretrained completes, load the vision tower explicitly.
        # Also clear mm_vision_tower from mm_tunable_parts: SigLipVisionTower.__init__
        # calls load_model() regardless of delay_load when mm_tunable_parts contains
        # "mm_vision_tower", bypassing the delay_load guard.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        config.delay_load = True
        # Fix 4: LLaDAConfig.__init__ sets use_cache=False via kwargs but the HF JSON
        # omits it; transformers 5.x no longer falls back to None for missing attrs.
        if not hasattr(config, "use_cache"):
            config.use_cache = False
        if getattr(config, "mm_tunable_parts", None):
            parts = [
                p for p in config.mm_tunable_parts.split(",")
                if p.strip() != "mm_vision_tower"
            ]
            config.mm_tunable_parts = ",".join(parts)

        # Fix 3b: SigLipVisionModel.from_pretrained runs inside the standard
        # get_init_context which includes torch.device("meta").  The non-persistent
        # position_ids buffer (register_buffer persistent=False) is not in the
        # checkpoint, so it is never materialized from meta and ends up as garbage.
        # Patch load_model to re-initialize position_ids after from_pretrained.
        SigLipVisionEmbeddings = get_class_from_dynamic_module(
            "modeling_lavida.SigLipVisionEmbeddings",
            pretrained_model_name,
        )
        _orig_vt_load_model = SigLipVisionTower.load_model

        def _patched_vt_load_model(self, device_map=None):
            _orig_vt_load_model(self, device_map=device_map)
            # Re-initialize non-persistent position_ids; it remains as a meta/garbage
            # tensor because it is not in the vision-tower checkpoint.
            emb = self.vision_tower.vision_model.embeddings
            if isinstance(emb, SigLipVisionEmbeddings):
                num_pos = emb.num_positions
                emb.register_buffer(
                    "position_ids",
                    torch.arange(num_pos).expand((1, -1)),
                    persistent=False,
                )

        SigLipVisionTower.load_model = _patched_vt_load_model

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Load the vision tower now that we are outside the meta-device context.
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            vision_tower.load_model()

        SigLipVisionTower.load_model = _orig_vt_load_model

        model.resize_token_embeddings(len(self.tokenizer))
        model.eval()

        # Fix 8: LLaDATransformer._activation_checkpoint_fn is unconditionally set to
        # partial(checkpoint, use_reentrant=False). PyTorch 2.7's use_reentrant=False
        # path calls _get_device_module("xla") → getattr(torch, "xla") which raises
        # AttributeError because torch.xla is not accessible as a torch attribute.
        # Gradient checkpointing is not needed for inference; replace with a plain call.
        def _plain_checkpoint(fn, *args, **kwargs):
            return fn(*args)

        for m in model.modules():
            if getattr(m, "_activation_checkpoint_fn", None) is not None:
                m._activation_checkpoint_fn = _plain_checkpoint

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LaViDa-LLaDA model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {
                "role": "user",
                "content": f"<image>\n{self.sample_text}",
            }
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tokenizer(
            pre, return_tensors="pt", add_special_tokens=False
        ).input_ids
        post_ids = self.tokenizer(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.model.get_vision_tower()
        # SigLipImageProcessor has no __call__; delegate to preprocess() directly.
        pixel_values = vision_tower.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"]

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        # LLaDA is a masked diffusion model. The forward() unconditionally indexes
        # into `labels` before the `if labels is not None` guard, so we must always
        # provide a labels tensor.  Append mask_id response tokens so the model can
        # exercise its full diffusion forward path; prompt positions get -100 (ignored).
        MASK_ID = 126336
        response_len = 32
        response_ids = torch.full((1, response_len), MASK_ID, dtype=input_ids.dtype)
        prompt_labels = torch.full_like(input_ids, -100)
        response_labels = response_ids.clone()
        labels = torch.cat([prompt_labels, response_labels], dim=1)
        input_ids = torch.cat([input_ids, response_ids], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, response_len, dtype=attention_mask.dtype)],
            dim=1,
        )

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            labels = labels.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
            "labels": labels,
        }
