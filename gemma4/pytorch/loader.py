# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 model loader implementation for causal language modeling.

google/gemma-4-12B is a Gemma4UnifiedForConditionalGeneration (any-to-any:
text + vision + audio). The default ``12B`` variant brings up the text-only
causal-LM path (input_ids + attention_mask only). The ``12B-image`` /
``12B-audio`` / ``12B-video`` variants drive the unified multimodal paths:
they reuse the same checkpoint and the verified text shard spec, and route
``load_inputs`` to the matching modality (image / audio / video + text). The
vision/audio embedder weights are left replicated (out of the shard map) for
this bring-up.

``26B-A4B-it`` is the MoE family (128 experts, top-8) with no audio encoder,
hence image/video variants only.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file

EXACT_SOFTMAX_ATTN = "tt_exact_softmax"


def _exact_softmax_attention(
    module, query, key, value, attention_mask, scaling, **kwargs
):
    """fp32 attention with softmax written as exp * reciprocal(sum).

    The stock exp / sum fuses into ttnn.softmax, which tt-mlir leaves at
    math_approx_mode=True (rows sum to up to 1.039). This form is not fused
    and is exact on TT. Remove once tt-mlir disables the approximation.
    """
    from transformers.models.gemma4.modeling_gemma4 import repeat_kv

    # fp32 like the stock SDPA decomposition; bf16 scores at |logit| ~30 cost
    # far more than the softmax fix gains.
    key = repeat_kv(key, module.num_key_value_groups).float()
    value = repeat_kv(value, module.num_key_value_groups).float()
    scores = torch.matmul(query.float(), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    exp = torch.exp(scores - torch.amax(scores, dim=-1, keepdim=True))
    probs = exp * torch.reciprocal(torch.sum(exp, dim=-1, keepdim=True))
    out = torch.matmul(probs, value).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def _use_exact_softmax_attention(model):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(EXACT_SOFTMAX_ATTN, _exact_softmax_attention)
    AttentionMaskInterface.register(EXACT_SOFTMAX_ATTN, eager_mask)
    model.set_attn_implementation(EXACT_SOFTMAX_ATTN)


def _use_fp32_vision_path(model):
    """Keep image features near fp32 so bf16 noise does not flip MoE routing.

    The vision standardize step roughly doubles incoming error; in bf16 the
    image tokens reach the language model ~9% off fp32. The merge casts the
    features back to the text dtype.
    """
    vision = model.model.vision_tower
    vision.float()
    model.model.embed_vision.float()

    def to32(t):
        return t.float() if torch.is_tensor(t) and t.is_floating_point() else t

    def upcast(module, args, kwargs):
        return tuple(map(to32, args)), {k: to32(v) for k, v in kwargs.items()}

    vision.register_forward_pre_hook(upcast, with_kwargs=True)


class ModelVariant(StrEnum):
    """Available Gemma4 model variants.

    ``12B`` is the text-only causal-LM path; the ``12B-{image,audio,video}``
    variants drive the unified multimodal paths on the same checkpoint.
    """

    GEMMA_4_12B = "12B"
    GEMMA_4_12B_IMAGE = "12B-image"
    GEMMA_4_12B_AUDIO = "12B-audio"
    GEMMA_4_12B_VIDEO = "12B-video"
    GEMMA_4_26B_A4B_IT = "26B-A4B-it"
    GEMMA_4_26B_A4B_IT_IMAGE = "26B-A4B-it-image"
    GEMMA_4_26B_A4B_IT_VIDEO = "26B-A4B-it-video"


class ModelLoader(ForgeModel):
    """Gemma4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_12B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-12B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_12B_IMAGE: LLMModelConfig(
            pretrained_model_name="google/gemma-4-12B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_12B_AUDIO: LLMModelConfig(
            pretrained_model_name="google/gemma-4-12B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_12B_VIDEO: LLMModelConfig(
            pretrained_model_name="google/gemma-4-12B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_26B_A4B_IT_IMAGE: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_26B_A4B_IT_VIDEO: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_12B

    _MOE_VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B_IT,
        ModelVariant.GEMMA_4_26B_A4B_IT_IMAGE,
        ModelVariant.GEMMA_4_26B_A4B_IT_VIDEO,
    }

    # Maps the multimodal variants to the modality their load_inputs drives and
    # the task reported in model_info. The text variant is absent (text path).
    _MODALITY_BY_VARIANT = {
        ModelVariant.GEMMA_4_12B_IMAGE: "image",
        ModelVariant.GEMMA_4_12B_AUDIO: "audio",
        ModelVariant.GEMMA_4_12B_VIDEO: "video",
        ModelVariant.GEMMA_4_26B_A4B_IT_IMAGE: "image",
        ModelVariant.GEMMA_4_26B_A4B_IT_VIDEO: "video",
    }
    _TASK_BY_VARIANT = {
        ModelVariant.GEMMA_4_12B_IMAGE: ModelTask.MM_IMAGE_TTT,
        ModelVariant.GEMMA_4_12B_AUDIO: ModelTask.MM_AUDIO_TTT,
        ModelVariant.GEMMA_4_12B_VIDEO: ModelTask.MM_VIDEO_TTT,
        ModelVariant.GEMMA_4_26B_A4B_IT_IMAGE: ModelTask.MM_IMAGE_TTT,
        ModelVariant.GEMMA_4_26B_A4B_IT_VIDEO: ModelTask.MM_VIDEO_TTT,
    }
    # The runner passes no num_frames; the full 32-frame clip is activation-bound.
    VIDEO_NUM_FRAMES = 4
    # Checkpoint default image budget (soft tokens, not pixels).
    IMAGE_MAX_SOFT_TOKENS = 280
    _IMAGE_MAX_SOFT_TOKENS_BY_VARIANT = {
        ModelVariant.GEMMA_4_26B_A4B_IT_IMAGE: 140,
    }

    sample_text = "What is your favorite city?"
    # Used by the optional image+text path of ``load_inputs`` (see ``include_image``).
    sample_image_text = "Describe the image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
    # Used by the optional audio+text path of ``load_inputs`` (see ``include_audio``).
    sample_audio_text = "Transcribe the audio."
    # Pre-saved 16 kHz mono waveform hosted on the tt-forge-models file server
    # (reused from the Whisper loader); avoids needing an audio-decode backend.
    sample_audio_file = "test_files/pytorch/whisper/1272-128104-0000.pt"
    # Used by the optional video+text path of ``load_inputs`` (see ``include_video``).
    sample_video_text = "Describe the video."
    sample_video_url = (
        "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/"
        "ForBiggerBlazes.mp4"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.processor = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        group = ModelGroup.GENERALITY
        # Multimodal variants report their modality task; text uses causal_lm.
        task = cls._TASK_BY_VARIANT.get(variant, ModelTask.NLP_CAUSAL_LM)

        return ModelInfo(
            model="Gemma 4",
            variant=variant,
            group=group,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current causal_lm variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _load_processor(self):
        """Load the multimodal processor for the image+text path.

        Returns:
            The loaded processor instance (``Gemma4UnifiedProcessor``).
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma4 causal_lm model instance (text-only path).

        Args:
            dtype_override: Optional torch dtype to load weights in. The model
                ships in bfloat16; when not provided, transformers uses the
                checkpoint's native dtype.

        Returns:
            torch.nn.Module: The Gemma4 unified model instance, driven through
            its causal language modeling (text) path.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer()
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Gemma4UnifiedForConditionalGeneration.__init__ does not accept
        # ``use_cache`` as a kwarg (unlike most causal-LM heads), so set it on
        # the config — on the text decoder sub-config when present.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False
        if hasattr(config, "text_config"):
            config.text_config.use_cache = False
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        # MoE routing amplifies small numeric errors into different experts.
        if self._variant in self._MOE_VARIANTS:
            _use_exact_softmax_attention(model)
            _use_fp32_vision_path(model)
        self.model = model
        self.config = model.config
        return model

    def unpack_forward_output(self, fwd_output):
        """Extract the logits tensor from the Gemma4 unified model output.

        Gemma4UnifiedForConditionalGeneration returns a custom
        ``Gemma4UnifiedCausalLMOutputWithPast`` that is not registered in the
        generic handler table, so unpack it explicitly here.
        """
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """1D Megatron TP on the ``model`` axis.

        Only query heads need to divide the axis: KV stays replicated (see
        ``load_shard_spec``).
        """
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        n_heads = text_cfg.num_attention_heads
        assert (
            n_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column/row split of attention and MLP; experts on the expert dim.

        KV stays replicated: the global layers fuse K and V into too few heads
        to split. Norms, embeddings, lm_head, router and vision stay replicated.
        """
        shard_specs = {}
        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            attn = layer.self_attn
            is_moe = hasattr(layer, "experts")
            if not is_moe or getattr(attn, "v_proj", None) is not None:
                shard_specs[attn.q_proj.weight] = ("model", None)
                shard_specs[attn.o_proj.weight] = (None, "model")

            if is_moe:
                shard_specs[layer.experts.gate_up_proj] = ("model", None, None)
                shard_specs[layer.experts.down_proj] = ("model", None, None)
        return shard_specs

    def _chat_template(self):
        """The processor's chat template, or None on a base checkpoint."""
        return getattr(self.processor, "chat_template", None) or getattr(
            getattr(self.processor, "tokenizer", None), "chat_template", None
        )

    def _apply_template_if_instruct(self, modality_token: str, text: str) -> str:
        """Instruct checkpoints need the turn structure to answer on-distribution."""
        if self._chat_template() is None:
            return f"{modality_token}{text}"

        return self.processor.apply_chat_template(
            [{"role": "user", "content": f"{modality_token}{text}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def load_image_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Image+text inputs; only ``pixel_values`` takes ``dtype_override``.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids, pixel_values,
                   image_position_ids}.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        image_token = getattr(self.processor, "image_token", "<|image|>")
        input_text = self._apply_template_if_instruct(
            image_token, prompt or self.sample_image_text
        )

        inputs = self.processor(
            text=input_text,
            images=image,
            max_soft_tokens=self._IMAGE_MAX_SOFT_TOKENS_BY_VARIANT.get(
                self._variant, self.IMAGE_MAX_SOFT_TOKENS
            ),
            return_tensors="pt",
        )
        inputs = dict(inputs)
        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )
        return inputs

    def load_audio_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        audio_array=None,
    ):
        """Load audio+text inputs for the Gemma4 unified multimodal (audio) path.

        Gemma4 is an any-to-any model; this drives its audio+text path. The
        ``Gemma4UnifiedProcessor`` turns ``"<|audio|>" + text`` plus a mono
        waveform into ``input_ids`` (with the audio token span),
        ``attention_mask``, ``mm_token_type_ids`` (0=text/1=audio),
        ``input_features`` ``(1, frames, 640)`` log-mel features, and
        ``input_features_mask`` ``(1, frames)``. google/gemma-4-12B is a base
        (non-instruct) checkpoint with no chat template, so the prompt is the
        plain ``audio_token`` + text (no ``apply_chat_template``).

        The waveform defaults to a pre-saved 16 kHz mono sample hosted on the
        tt-forge-models file server (reused from the Whisper loader), so no
        audio-decode backend (soundfile/librosa) is required. The feature
        extractor expects 16 kHz; pass ``audio_array`` to override.

        Only ``input_features`` is cast to ``dtype_override``; the id/mask
        tensors stay integer/bool.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids, input_features,
                   input_features_mask} for the audio+text path.
        """
        if self.processor is None:
            self._load_processor()

        if audio_array is None:
            sample_path = get_file(self.sample_audio_file)
            sample = torch.load(sample_path, weights_only=False)
            audio_array = sample["audio"]["array"]

        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio_token = getattr(self.processor, "audio_token", "<|audio|>")
        input_text = f"{audio_token}{prompt or self.sample_audio_text}"

        inputs = self.processor(
            text=input_text,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = dict(inputs)
        if dtype_override is not None and "input_features" in inputs:
            inputs["input_features"] = cast_input_to_type(
                inputs["input_features"], dtype_override
            )
        return inputs

    def load_video_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        num_frames: int = 32,
        image_url: Optional[str] = None,
    ):
        """Video+text inputs; only ``pixel_values_videos`` takes ``dtype_override``.

        Instruct checkpoints sample a real clip, since a repeated still draws a
        refusal. Base checkpoints have no template for a video URL, so they keep
        the replicated still their PCC baselines were measured on.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids,
                   pixel_values_videos, video_position_ids}.
        """
        if self.processor is None:
            self._load_processor()

        text = prompt or self.sample_video_text
        if self._chat_template() is not None:
            inputs = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": self.sample_video_url},
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                processor_kwargs={"num_frames": num_frames},
            )
        else:
            image_file = get_file(image_url or self.sample_image_url)
            frame = np.array(Image.open(image_file).convert("RGB"))
            frames = np.stack([frame] * num_frames)  # (T, H, W, C)
            video_token = getattr(self.processor, "video_token", "<|video|>")
            inputs = self.processor(
                text=f"{video_token}{text}",
                videos=frames,
                do_sample_frames=False,
                return_tensors="pt",
            )
        inputs = dict(inputs)
        if dtype_override is not None and "pixel_values_videos" in inputs:
            inputs["pixel_values_videos"] = cast_input_to_type(
                inputs["pixel_values_videos"], dtype_override
            )
        return inputs

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
        include_image: bool = False,
        image_url: Optional[str] = None,
        include_audio: bool = False,
        include_video: bool = False,
        num_frames: int = 32,
    ):
        """Load and return sample inputs for the Gemma4 model.

        Defaults to the text-only path. Set one of ``include_image`` /
        ``include_audio`` / ``include_video`` to ``True`` to get inputs for the
        corresponding unified multimodal path (delegates to
        :meth:`load_image_inputs` / :meth:`load_audio_inputs` /
        :meth:`load_video_inputs`). The modality flags are mutually exclusive;
        ``include_image`` takes precedence, then ``include_audio``, then
        ``include_video``.

        Returns a dict (not a list) so the harness passes the tensors as
        keyword arguments. Gemma4UnifiedForConditionalGeneration.forward has
        ``pixel_values`` as its second positional parameter, so a positional
        ``[input_ids, attention_mask]`` would bind the attention mask to
        ``pixel_values`` and wrongly trigger the vision tower. Keying by name
        keeps ``pixel_values`` None and stays on the text-only path.

        Returns:
            dict: {"input_ids", "attention_mask"} for the text path, or the
            multimodal tensor dict for the selected modality.
        """
        # Variant-driven dispatch: the runner calls load_inputs with only
        # dtype_override/batch_size, so the 12B-{image,audio,video} variants
        # select their modality here (explicit include_* kwargs still win).
        if not (include_image or include_audio or include_video):
            modality = self._MODALITY_BY_VARIANT.get(self._variant)
            if modality == "image":
                include_image = True
            elif modality == "audio":
                include_audio = True
            elif modality == "video":
                include_video = True
                num_frames = self.VIDEO_NUM_FRAMES

        if include_image:
            return self.load_image_inputs(
                dtype_override=dtype_override, prompt=prompt, image_url=image_url
            )
        if include_audio:
            return self.load_audio_inputs(dtype_override=dtype_override, prompt=prompt)
        if include_video:
            return self.load_video_inputs(
                dtype_override=dtype_override,
                prompt=prompt,
                num_frames=num_frames,
                image_url=image_url,
            )
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()
        # google/gemma-4-12B is a base (non-instruct) checkpoint and ships
        # without a chat template, so fall back to plain prompt text when no
        # template is available.
        if getattr(self.tokenizer, "chat_template", None):
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt or self.sample_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            input_text = prompt or self.sample_text
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return {"input_ids": input_ids, "attention_mask": attn_mask}
