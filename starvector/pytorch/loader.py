# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
StarVector model loader implementation for image-to-SVG generation.
"""

from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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


def _patch_starcoder_init():
    """Patch starvector v1 internals to avoid downloading the gated bigcode/starcoderbase-1b model.

    StarCoderModel (v1): uses bigcode/starcoder2-3b tokenizer (same 49152-token vocab, ungated)
    and initializes GPTBigCode from config instead of from pretrained weights.  The
    starvector from_pretrained call that follows will override all weights anyway.

    StarVectorStarCoder (v1): wraps the processor assignment in a try/except because the
    starvector HF repo has no processor_config.json; the processor is recovered later
    from the image encoder when load_inputs is called.
    """
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        GPTBigCodeConfig,
        GPTBigCodeForCausalLM,
        utils,
    )
    import starvector.model.llm.starcoder as sc_module
    import starvector.model.models.starvector_v1 as sv1_module

    def _patched_starcoder_init(self, config, **kwargs):
        nn.Module.__init__(self)

        # Use ungated starcoder2 tokenizer (identical 49152-token vocabulary)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder2-3b", use_fast=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        for special_token in ["<svg-start>", "<image-start>", "<caption-start>"]:
            self.tokenizer.add_tokens([special_token])
        self.svg_start_token = "<svg-start>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]

        self.max_length = config.max_length

        # Build GPTBigCode config from starvector config params (no gated download needed).
        # Set vocab_size to the post-resize size so resize_token_embeddings is not needed
        # (calling it on meta tensors would fail; weights are overridden by from_pretrained).
        vocab_size = len(self.tokenizer)
        model_config = GPTBigCodeConfig(
            vocab_size=vocab_size,
            n_positions=getattr(config, "max_length", 8192),
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.hidden_size * 4,
            multi_query=getattr(config, "multi_query", True),
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        model_config.torch_dtype = getattr(config, "torch_dtype", "float16")
        if not utils.is_flash_attn_2_available():
            config.use_flash_attn = False

        self.transformer = GPTBigCodeForCausalLM(model_config)
        self.prompt = "<svg"

    sc_module.StarCoderModel.__init__ = _patched_starcoder_init

    original_sv1_init = sv1_module.StarVectorStarCoder.__init__

    def _patched_sv1_init(self, config, **kwargs):
        # Call StarVectorBase.__init__ (sets up image encoder, adapter, etc.)
        sv1_module.StarVectorStarCoder.__bases__[0].__init__(self, config, **kwargs)
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                config._name_or_path, trust_remote_code=True
            )
        except Exception:
            # Processor will be recovered from image_encoder.processor in load_inputs
            self.processor = getattr(
                getattr(self, "image_encoder", None), "processor", None
            )

    sv1_module.StarVectorStarCoder.__init__ = _patched_sv1_init

    # starvector_arch.py (written for transformers 4.x) never calls post_init(), so
    # all_tied_weights_keys is missing in transformers 5.x.  Guard the method that needs it.
    from transformers import PreTrainedModel

    original_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched_adjust(self, missing_and_mismatched):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original_adjust(self, missing_and_mismatched)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust


def _patch_image_encoder_init():
    """Patch ImageEncoder to exit the meta device context when loading siglip from_pretrained.

    transformers 5.x sets torch.default_device('meta') during AutoModelForCausalLM.from_pretrained
    for memory efficiency.  ImageEncoder's siglip branch calls AutoModel.from_pretrained inside
    that context, which raises RuntimeError.  We temporarily reset to CPU, download the visual
    encoder, then restore the original device.
    """
    import starvector.model.image_encoder.image_encoder as ie_module

    original_ie_init = ie_module.ImageEncoder.__init__

    def _patched_ie_init(self, config, **kwargs):
        import torch

        if "siglip" not in config.image_encoder_type:
            original_ie_init(self, config, **kwargs)
            return

        # Exit meta device context so from_pretrained can allocate real tensors
        _orig_device = torch.get_default_device()
        if _orig_device.type == "meta":
            torch.set_default_device("cpu")
            try:
                original_ie_init(self, config, **kwargs)
            finally:
                torch.set_default_device("meta")
        else:
            original_ie_init(self, config, **kwargs)

    ie_module.ImageEncoder.__init__ = _patched_ie_init


def _patch_starcoder2_init():
    """Patch starvector v2 internals to avoid downloading bigcode/starcoder2-7b with flash attention.

    StarCoderModel (v2): downloads starcoder2-7b config and builds the model from config instead
    of from pretrained weights.  The starvector from_pretrained call overrides all weights anyway.
    Flash attention is disabled since flash-attn is not installed in this environment.
    """
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
    import starvector.model.llm.starcoder2 as sc2_module

    def _patched_starcoder2_init(self, config, **kwargs):
        nn.Module.__init__(self)

        # Downloads tokenizer only (small, fast); starcoder2-7b is ungated
        self.init_tokenizer(config.starcoder_model_name)
        self.max_length = config.max_length

        # Download just config (tiny JSON) from the base model; no weight download
        model_config = AutoConfig.from_pretrained(
            config.starcoder_model_name, trust_remote_code=True
        )
        model_config.use_cache = getattr(config, "use_cache", True)
        model_config.use_bfloat16 = True
        # Disable flash attention (not installed)
        if hasattr(model_config, "attn_implementation"):
            model_config.attn_implementation = None
        # Pre-set vocab_size to avoid resize_token_embeddings on the fresh model
        model_config.vocab_size = len(self.tokenizer)

        # Build model from config (no weight download); starvector weights override later
        model = AutoModelForCausalLM.from_config(model_config)
        self.transformer = model

        self.prompt = "<svg"

        # transformer_layer_cls is only needed for FSDP training; safe to skip for inference
        try:
            from starvector.train.util import get_module_class_from_name

            self.transformer_layer_cls = get_module_class_from_name(
                self, kwargs.get("transformer_layer_cls", "Starcoder2DecoderLayer")
            )
        except Exception:
            self.transformer_layer_cls = None

    sc2_module.StarCoderModel.__init__ = _patched_starcoder2_init


class _StarVectorWrapper(nn.Module):
    """Wraps StarVectorForCausalLM to accept raw pixel_values.

    StarVectorForCausalLM.forward() expects a batch dict with 'image', 'svg', etc.
    for training.  This wrapper encodes the image through the model's own image
    encoder and runs a single forward through the underlying LLM so the test
    harness can call model(pixel_values) directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        sv = self.model.model  # StarVectorBase subclass (v1 or v2)

        # Encode image through vision encoder + projection adapter
        vision_embeds = sv.image_encoder(pixel_values.to(sv.model_precision))
        vision_embeds = sv.image_projection(vision_embeds)  # [B, Q, hidden]

        # Build a minimal token embedding for the SVG start token
        svg_start_id = sv.svg_transformer.svg_start_token_id
        input_ids = torch.tensor(
            [[svg_start_id]], dtype=torch.long, device=pixel_values.device
        )
        token_embeds = sv._get_embeddings(input_ids)  # [1, 1, hidden]

        # Concatenate vision features with the start token and run LLM forward
        inputs_embeds = torch.cat([vision_embeds, token_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=pixel_values.device
        )

        return sv.svg_transformer.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )


class ModelVariant(StrEnum):
    """Available StarVector model variants."""

    STARVECTOR_1B = "1B"
    STARVECTOR_8B = "8B"


class ModelLoader(ForgeModel):
    """StarVector model loader implementation for image-to-SVG generation tasks."""

    _VARIANTS = {
        ModelVariant.STARVECTOR_1B: ModelConfig(
            pretrained_model_name="starvector/starvector-1b-im2svg",
        ),
        ModelVariant.STARVECTOR_8B: ModelConfig(
            pretrained_model_name="starvector/starvector-8b-im2svg",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STARVECTOR_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="StarVector",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        _patch_starcoder_init()
        _patch_starcoder2_init()
        _patch_image_encoder_init()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.processor = getattr(model.model, "processor", None)

        return _StarVectorWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            pretrained_model_name = self._variant_config.pretrained_model_name
            try:
                self.processor = AutoProcessor.from_pretrained(
                    pretrained_model_name, trust_remote_code=True
                )
            except Exception:
                from starvector.data.util import ImageTrainProcessor

                self.processor = ImageTrainProcessor(size=224)

        # Use a synthetic image to avoid importing datasets (local spacy/ dir shadows the
        # real spacy package and breaks datasets fingerprinting).
        image = Image.new("RGB", (384, 384), color=(128, 128, 128))

        result = self.processor(image)
        if isinstance(result, dict) or hasattr(result, "data"):
            pixel_values = result["pixel_values"]
            # SimpleStarVectorProcessor returns [C, H, W] for single images; add batch dim
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
        else:
            # ImageTrainProcessor returns a raw tensor
            pixel_values = result.unsqueeze(0) if result.dim() == 3 else result

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
