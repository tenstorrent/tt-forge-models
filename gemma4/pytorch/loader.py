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
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


def _gemma4_dense_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Static, graph-traceable replacement for ``Gemma4TextExperts.forward``.

    The stock forward is data-dependent: it builds an expert-hit mask, calls
    ``.nonzero()`` and then loops in Python over the hit experts, dynamically
    gathering the tokens routed to each. That control flow cannot be traced
    into a static graph — on the ``tt`` backend it segfaults the compiler
    (CPU-fallback partitioning of the dynamic gather / ``index_add_``).

    This rewrite computes every expert densely and combines them with a
    one-hot routing weight matrix — no ``nonzero``, no Python loop, no
    dynamic gather. It is numerically identical to the sparse form (verified
    PCC 1.0 on CPU): top-k selection is expressed as zeros in ``weights_full``
    for the non-selected experts, so the extra experts contribute nothing.
    Dense costs ``num_experts / top_k`` (16x here) more expert FLOPs, which is
    tractable for prefill and is the standard MoE-on-static-shapes trade.

    Shapes: hidden_states ``[T, H]``, top_k_index/top_k_weights ``[T, K]``,
    gate_up_proj ``[E, 2I, H]``, down_proj ``[E, H, I]``. Sharding the expert
    weights on their leading (expert) dim makes this expert-parallel: each
    chip owns a slice of ``E`` and the final sum over ``E`` becomes a
    cross-device reduce (see ``load_shard_spec``).
    """
    E = self.num_experts
    # Dense routing weights: weights_full[t, e] = sum_k [top_k_index[t,k]==e] * w.
    # Built with a broadcast-compare (arange == index) rather than one_hot: on
    # the SPMD/tensor-parallel path one_hot lowers to a stablehlo.scatter whose
    # arguments cannot be sharding-annotated ("GSPMD presharded argument missing
    # @Sharding custom call"), while the compare lowers to broadcast + eq that
    # shards cleanly.
    arange_e = torch.arange(E, device=top_k_index.device)
    oh = (top_k_index.unsqueeze(-1) == arange_e).to(top_k_weights.dtype)  # [T,K,E]
    weights_full = (oh * top_k_weights.unsqueeze(-1)).sum(dim=1)  # [T, E]

    # Batched (per-expert) matmuls, expert dim as the batch dim, rather than a
    # single 4D einsum. einsum lowers to a broadcast-multiply-reduce whose
    # [T, E, H, I] intermediate (tens of GB) is materialized in full under the
    # Shardy manual-computation path — an explicit bmm keeps it a contracting
    # matmul that composes with the intermediate-dim (column->row) sharding.
    h = hidden_states.unsqueeze(0)  # [1, T, H]
    gate_up = torch.matmul(h, self.gate_up_proj.transpose(-1, -2))  # [E, T, 2I]
    gate, up = gate_up.chunk(2, dim=-1)  # [E, T, I]
    inter = self.act_fn(gate) * up  # [E, T, I]
    down = torch.matmul(inter, self.down_proj.transpose(-1, -2))  # [E, T, H]
    w = weights_full.transpose(0, 1).unsqueeze(-1)  # [E, T, 1]
    out = (down * w).sum(dim=0)  # [T, H]
    return out.to(hidden_states.dtype)

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


class ModelVariant(StrEnum):
    """Available Gemma4 model variants.

    ``12B`` is the text-only causal-LM path; the ``12B-{image,audio,video}``
    variants drive the unified multimodal paths on the same checkpoint.
    """

    GEMMA_4_12B = "12B"
    GEMMA_4_12B_IMAGE = "12B-image"
    GEMMA_4_12B_AUDIO = "12B-audio"
    GEMMA_4_12B_VIDEO = "12B-video"
    # Sparse-MoE VLM checkpoint (Gemma4ForConditionalGeneration). This variant
    # brings up the text MoE decoder prefill path only, tensor-parallel.
    GEMMA_4_26B_A4B_IT = "26B-A4B-it"


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
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_12B

    # Maps the multimodal variants to the modality their load_inputs drives and
    # the task reported in model_info. The text variant is absent (text path).
    _MODALITY_BY_VARIANT = {
        ModelVariant.GEMMA_4_12B_IMAGE: "image",
        ModelVariant.GEMMA_4_12B_AUDIO: "audio",
        ModelVariant.GEMMA_4_12B_VIDEO: "video",
    }
    _TASK_BY_VARIANT = {
        ModelVariant.GEMMA_4_12B_IMAGE: ModelTask.MM_IMAGE_TTT,
        ModelVariant.GEMMA_4_12B_AUDIO: ModelTask.MM_AUDIO_TTT,
        ModelVariant.GEMMA_4_12B_VIDEO: ModelTask.MM_VIDEO_TTT,
    }
    # Frames in the static video clip when the video variant runs through the
    # runner (which calls load_inputs without num_frames). 4 frames (~256 video
    # tokens) is the verified, activation-tractable footprint; the full 32-frame
    # clip (2048 tokens) is activation-bound. See load_video_inputs.
    VIDEO_NUM_FRAMES = 4

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
        # The 26B-A4B-it checkpoint is a Gemma4ForConditionalGeneration (a
        # vision+text VLM), which is registered under the image-text-to-text
        # auto class rather than causal-LM. The 12B unified variants keep the
        # existing causal-LM auto class.
        if self._variant == ModelVariant.GEMMA_4_26B_A4B_IT:
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
            self._patch_dense_experts(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    @staticmethod
    def _patch_dense_experts(model):
        """Swap the data-dependent MoE expert dispatch for a static dense form.

        google/gemma-4-26B-A4B-it's ``Gemma4TextExperts.forward`` routes tokens
        to experts with ``.nonzero()`` + a Python loop + a dynamic gather, which
        segfaults the ``tt`` compiler (it cannot trace data-dependent control
        flow) and, single-chip, falls the ``nonzero`` back to CPU.
        ``_gemma4_dense_experts_forward`` is a numerically-identical static
        replacement (verified PCC 1.0 on CPU) so the prefill graph is fully
        static and, with expert-parallel sharding, splits the 45.7 GB of expert
        weights across the mesh.

        The patch is applied to the **class**, not the module instance:
        ``torch.compile``/dynamo traces ``type(module).forward`` and ignores an
        instance-level ``forward`` attribute, so an instance override would
        silently leave the original ``nonzero`` path in the compiled graph.
        """
        import transformers.models.gemma4.modeling_gemma4 as g4

        g4.Gemma4TextExperts.forward = _gemma4_dense_experts_forward

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
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP.

        The Gemma4 unified text decoder is a standard causal-LM stack, so it
        uses Megatron 1D tensor parallelism: weights are sharded only on the
        ``model`` axis and the non-sharded tensor dimension is replicated
        (``None`` in the shard specs rather than a second ``batch`` shard axis).
        Query heads (and the MLP) are sharded on the model axis, so
        ``num_attention_heads`` must be divisible by it. KV projections are
        left replicated (see ``load_shard_spec``):
        the 8 global layers carry a single global KV head (``attention_k_eq_v``
        with ``num_global_key_value_heads == 1``) that cannot be split across
        the mesh, so no KV-head divisibility constraint is imposed here.
        """
        mesh_shape = (1, num_devices)
        text_cfg = getattr(self.config, "text_config", self.config)
        n_heads = text_cfg.num_attention_heads
        assert (
            n_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map for the Gemma4 unified text decoder.

        Column-parallel (shard out_features on the model axis) for q_proj and
        the MLP gate/up projections; row-parallel for o_proj and down_proj.

        KV projections are intentionally **replicated** (omitted from the map):
        Gemma4's global ``full_attention`` layers use ``attention_k_eq_v`` so
        ``v_proj is None`` (value reuses key) and carry only a single global KV
        head — a single head cannot be sharded across the model axis, and
        mixing sharded/replicated KV per layer-type is fragile on a first
        compile. Replicating all KV is the standard GQA-TP fallback and keeps
        every query head correctly grouped on each chip. ``k_proj``/``v_proj``
        are therefore skipped (and guarded for absence/None).

        Per-projection RMSNorms (q_norm/k_norm/v_norm), layernorms, embeddings
        and lm_head are left replicated. Vision/audio towers are unused on the
        text-only path and are replicated.
        """
        shard_specs = {}
        text_cfg = getattr(self.config, "text_config", self.config)
        layer_types = getattr(text_cfg, "layer_types", None)
        for i, layer in enumerate(model.model.language_model.layers):
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            attn = layer.self_attn
            # Shard the query/output projections (Megatron column->row) only on
            # the local sliding-window layers. The global ``full_attention``
            # layers carry a single global KV head (``attention_k_eq_v``): with
            # the query heads sharded on the model axis, broadcasting that lone
            # replicated KV head across the sharded query heads forces an
            # ``sdy.all_slice`` on the model axis that is already bound by the
            # tensor-parallel manual_computation ("axis already bound"), so those
            # layers fail to compile. They are only 5 of 30 layers and their
            # attention weights are tiny next to the experts, so leave them
            # replicated. (Sliding layers have 8 KV heads and shard cleanly.)
            layer_type = layer_types[i] if layer_types else "sliding_attention"
            if layer_type != "full_attention":
                shard_specs[attn.q_proj.weight] = ("model", None)
                shard_specs[attn.o_proj.weight] = (None, "model")
            # k_proj/v_proj replicated (skipped). On Gemma4 global layers
            # v_proj is None and k_proj holds a single unsharddable KV head.

            # Expert-parallel sharding for the sparse-MoE (26B-A4B-it) experts.
            # The 128 experts hold 45.7 GB that cannot be replicated on a 32 GB
            # chip. Shard both 3D expert tensors on their leading (expert) dim,
            # which is the batch dim of the dense expert forward's per-expert
            # matmuls: each chip owns E/num_devices experts, computes their
            # contribution, and the final sum over the expert dim becomes a
            # cross-device all-reduce. This is preferred over sharding the fused
            # gate_up 2I dim (which would split the gate/up halves across chips
            # and break the ``chunk(2)`` + gate*up). Requires
            # num_experts % model-axis == 0 (128 % 4 == 0).
            if hasattr(layer, "experts"):
                experts = layer.experts
                shard_specs[experts.gate_up_proj] = ("model", None, None)
                shard_specs[experts.down_proj] = ("model", None, None)
        return shard_specs

    def _build_prefill_mask_dict(self, seq_len, dtype):
        """Precompute the additive causal-mask dict for the text prefill path.

        Gemma4's forward builds ``full_attention`` / ``sliding_attention`` masks
        via ``create_causal_mask`` / ``create_sliding_window_causal_mask``, which
        lower to a ``stablehlo.scatter`` (from an ``aten::nonzero``) that the
        tt-mlir SPMD pipeline cannot sharding-annotate (``AnnotateLocalShapesPass``
        → "GSPMD presharded argument missing @Sharding custom call"), so
        tensor-parallel compilation fails. The forward accepts a pre-built dict
        for ``attention_mask`` (skipping construction entirely), so we compute
        the masks on the host here — an additive ``[1, 1, S, S]`` tensor, ``0``
        where attention is allowed and ``finfo.min`` where masked. For prefill
        with no padding and ``S <= sliding_window`` both mask types are plain
        causal. The scatter therefore never enters the traced graph.
        """
        mask_dtype = dtype if dtype is not None else torch.float32
        cfg = getattr(self, "config", None)
        if cfg is None:
            cfg = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        sliding_window = getattr(
            getattr(cfg, "text_config", cfg), "sliding_window", None
        )
        q = torch.arange(seq_len).view(seq_len, 1)
        k = torch.arange(seq_len).view(1, seq_len)
        allowed_full = k <= q
        if sliding_window:
            allowed_sliding = allowed_full & (k > q - sliding_window)
        else:
            allowed_sliding = allowed_full
        minval = torch.finfo(mask_dtype).min

        def additive(allowed):
            m = torch.where(
                allowed,
                torch.zeros((), dtype=mask_dtype),
                torch.full((), minval, dtype=mask_dtype),
            )
            return m.view(1, 1, seq_len, seq_len)

        return {
            "full_attention": additive(allowed_full),
            "sliding_attention": additive(allowed_sliding),
        }

    def load_image_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load image+text inputs for the Gemma4 unified multimodal (vision) path.

        Gemma4 is an any-to-any model; this drives its image+text path. The
        ``Gemma4UnifiedProcessor`` turns ``"<|image|>" + text`` plus a PIL image
        into the five tensors the forward needs: ``input_ids`` (with the image
        token span), ``attention_mask``, ``mm_token_type_ids`` (0=text/1=image),
        ``pixel_values`` ``(1, 280, 6912)`` merged patches, and
        ``image_position_ids`` ``(1, 280, 2)``. google/gemma-4-12B is a base
        (non-instruct) checkpoint with no chat template, so the prompt is the
        plain ``image_token`` + text (no ``apply_chat_template``).

        Only ``pixel_values`` is cast to ``dtype_override``; the id/mask tensors
        stay integer.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids, pixel_values,
                   image_position_ids} for the image+text path.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        image_token = getattr(self.processor, "image_token", "<|image|>")
        input_text = f"{image_token}{prompt or self.sample_image_text}"

        inputs = self.processor(text=input_text, images=image, return_tensors="pt")
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
        """Load video+text inputs for the Gemma4 unified multimodal (video) path.

        Gemma4 is an any-to-any model; this drives its image-sequence (video)
        path. The ``Gemma4UnifiedProcessor`` turns ``"<|video|>" + text`` plus a
        stack of frames into ``input_ids`` (with the video token span),
        ``attention_mask``, ``mm_token_type_ids`` (0=text/1=video),
        ``pixel_values_videos`` ``(1, T, patches, 6912)`` merged patches, and
        ``video_position_ids`` ``(1, T, patches, 2)``. google/gemma-4-12B is a
        base (non-instruct) checkpoint with no chat template, so the prompt is
        the plain ``video_token`` + text (no ``apply_chat_template``).

        No video-decode backend (av/decord) is available, so frames are built
        by replicating the sample image into ``num_frames`` pre-sampled frames
        (a static clip). ``do_sample_frames=False`` is passed so the processor
        consumes exactly the frames given (rather than its default 32-frame
        resampler), which lets ``num_frames`` control the video token count
        directly: each frame contributes 64 video tokens, so the on-device
        sequence is ~ ``64 * num_frames``. The full 32-frame clip emits 2048
        video tokens (a ~2.3k sequence) and is activation-bound on a single
        chip; lower ``num_frames`` (e.g. 4) for a tractable first bring-up.

        Only ``pixel_values_videos`` is cast to ``dtype_override``; the
        id/mask tensors stay integer.

        Returns:
            dict: {input_ids, attention_mask, mm_token_type_ids,
                   pixel_values_videos, video_position_ids} for the video path.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(image_url or self.sample_image_url)
        frame = np.array(Image.open(image_file).convert("RGB"))
        frames = np.stack([frame] * num_frames)  # (T, H, W, C)

        video_token = getattr(self.processor, "video_token", "<|video|>")
        input_text = f"{video_token}{prompt or self.sample_video_text}"

        inputs = self.processor(
            text=input_text,
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

        # The 26B-A4B-it prefill path is brought up tensor-parallel. Pass a
        # precomputed additive causal-mask dict so the model skips its own
        # scatter-producing mask construction, which the SPMD pipeline cannot
        # compile (see _build_prefill_mask_dict). Other variants keep the plain
        # 2D attention_mask and let the model build its masks.
        if self._variant == ModelVariant.GEMMA_4_26B_A4B_IT:
            mask_dict = self._build_prefill_mask_dict(
                input_ids.shape[1], dtype_override
            )
            return {"input_ids": input_ids, "attention_mask": mask_dict}
        return {"input_ids": input_ids, "attention_mask": attn_mask}
