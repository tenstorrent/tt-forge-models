# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Muse-Glimmer 30B multimodal (image + text → text) model loader.

``meta-models/Muse-Glimmer-30B`` is MuseGlimmerForConditionalGeneration.
Native support requires transformers>=5.15 (pinned in requirements.txt).
"""

from typing import Optional

import torch
from PIL import Image

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
from ....tools.utils import cast_input_to_type, get_file

# NOTE: ``transformers`` is intentionally NOT imported at module top level.
# This model pins transformers==5.15.0 (see requirements.txt). The test runner
# installs that pin at test time and purges transformers from sys.modules. A
# top-level import would bind Auto* classes to whatever transformers was loaded
# during pytest collection. Auto* classes are imported lazily in the methods
# that use them.


class ModelVariant(StrEnum):
    """Available Muse-Glimmer multimodal model variants."""

    MUSE_GLIMMER_30B = "Muse-Glimmer-30B"


class ModelLoader(ForgeModel):
    """Muse-Glimmer 30B loader for image-conditioned generation."""

    _VARIANTS = {
        ModelVariant.MUSE_GLIMMER_30B: LLMModelConfig(
            pretrained_model_name="meta-models/Muse-Glimmer-30B",
            # Must fit max_image_tokens + prompt; keep short for DRAM.
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSE_GLIMMER_30B

    sample_text = "Describe this image."
    sample_image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/p-blog/candy.JPG"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.model = None
        # CPU copy of ``image_grid_thw`` (set by ``load_inputs``). The vision
        # tower derives all of its shapes from it; see ``_patch_vision_tower``.
        self._grid_cpu = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Muse-Glimmer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _load_processor(self):
        """Load Muse-Glimmer multimodal processor (image + tokenizer)."""
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return MuseGlimmerForConditionalGeneration.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: MuseGlimmerForConditionalGeneration in eval mode.
        """
        from transformers import AutoModelForMultimodalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.eval()
        self._patch_vision_tower(model)
        self.config = model.config
        self.model = model
        print(f"Model loaded: {model}")
        return model

    def _patch_vision_tower(self, model):
        """Make the vision tower's shape logic static / host-side.

        The HF vision path derives every shape (attention segment lengths,
        window index, rope positions, pixel-shuffle permutation, per-image
        split sizes) from ``image_grid_thw`` via ``.tolist()`` / ``int()`` /
        ``repeat_interleave``. The test harness moves ``image_grid_thw`` to the
        XLA device, so each of those becomes a device->host sync inside the
        compiled region. The first execution is numerically right, but every
        subsequent execution of the same compiled graphs returns corrupted
        image features (Muse-Glimmer PCC 0.99 on run 0, ~0.06 on runs >= 1),
        which is what the runner sees after its warmup/perf iterations.

        All grid-derived values are computed once, eagerly, from the CPU copy of
        the grid captured in ``load_inputs`` (see ``_build_vision_static``) and
        only static python ints / constant index buffers reach the compiled
        graph. The math is unchanged.
        """
        import types

        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.muse_glimmer import modeling_muse_glimmer as M

        inner = model.model
        vt = inner.vision_tower
        if getattr(vt, "_tt_static_patch", False):
            return

        def attn_forward(attn, hidden_states, cu_seqlens, position_embeddings=None, **kw):
            # ``cu_seqlens`` is a python list of segment lengths here.
            seq_length = hidden_states.shape[0]
            q = attn.q_proj(hidden_states).reshape(1, seq_length, -1, attn.head_dim)
            k = attn.k_proj(hidden_states).reshape(1, seq_length, -1, attn.head_dim)
            v = attn.v_proj(hidden_states).reshape(1, seq_length, -1, attn.head_dim)
            cos, sin = position_embeddings
            q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
            q, k, v = (t.transpose(2, 1) for t in (q, k, v))
            interface = M.ALL_ATTENTION_FUNCTIONS.get_interface(
                attn.config._attn_implementation, M.eager_attention_forward
            )
            splits = [torch.split(t, cu_seqlens, dim=2) for t in (q, k, v)]
            outs = [
                interface(
                    attn,
                    qs,
                    ks,
                    vs,
                    attention_mask=None,
                    scaling=attn.scaling,
                    dropout=0.0,
                    is_causal=False,
                )[0]
                for qs, ks, vs in zip(*splits)
            ]
            out = torch.cat(outs, dim=1).reshape(seq_length, -1).contiguous()
            return attn.proj(out)

        for layer in vt.layers:
            layer.attn.forward = types.MethodType(attn_forward, layer.attn)

        def vision_forward(vt_, pixel_values, grid_thw, **kwargs):
            st = vt_._tt_static
            cfg = vt_.config
            factor = cfg.merge_size

            hidden = vt_.ln_pre(vt_.patch_embedder(pixel_values, grid_thw))
            hidden = hidden[vt_.tt_window_index, :]
            pos_emb = vt_.rotary_emb(hidden, vt_.tt_position_ids)

            lengths = {
                "full_attention": st["full_lengths"],
                "window_attention": st["window_lengths"],
            }
            for i, block in enumerate(vt_.layers):
                hidden = block(
                    hidden,
                    position_embeddings=pos_emb,
                    cu_seqlens=lengths[cfg.layer_types[i]],
                )

            hidden = hidden[vt_.tt_reverse_indices, :]
            hidden = vt_.ln_post(hidden)

            # Static pixel shuffle (same math as ``pixel_shuffle`` in the HF model).
            dim = hidden.shape[-1]
            out = []
            for i, (offset, n_tokens, n_out) in enumerate(st["shuffle"]):
                chunk = hidden[offset : offset + n_tokens]
                down = chunk[getattr(vt_, f"tt_ds_perm_{i}")]
                down = down.view(n_out, factor * factor, dim)
                out.append(down.permute(0, 2, 1).contiguous().view(n_out, dim * factor * factor))
            return BaseModelOutputWithPooling(last_hidden_state=torch.cat(out, dim=0))

        vt.forward = types.MethodType(vision_forward, vt)

        def get_image_features(m, pixel_values, image_grid_thw, **kwargs):
            out = m.vision_tower(pixel_values=pixel_values, grid_thw=image_grid_thw)
            feats = m.vision_adapter(out.last_hidden_state)
            feats = m.vision_projection(feats)
            feats = m.perception_emb_norm(feats)
            out.pooler_output = torch.split(feats, m.vision_tower._tt_static["split_sizes"])
            return out

        inner.get_image_features = types.MethodType(get_image_features, inner)
        vt._tt_static_patch = True
        self._vision_tower = vt
        self._build_vision_static()

    def _build_vision_static(self):
        """Precompute grid-derived constants (eagerly, on CPU) for the patched vision tower."""
        vt = getattr(self, "_vision_tower", None)
        grid = self._grid_cpu
        if vt is None or grid is None:
            return
        from transformers.vision_utils import (
            get_vision_position_ids,
            get_vision_window_index,
        )

        cfg = vt.config
        factor = cfg.merge_size

        full_lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).tolist()
        window_index, cu_window = get_vision_window_index(
            grid,
            spatial_merge_size=1,
            window_size=cfg.pos_emb_height * cfg.patch_size,
            patch_size=cfg.patch_size,
        )
        window_lengths = (cu_window[1:] - cu_window[:-1]).tolist()
        position_ids = get_vision_position_ids(grid, spatial_merge_size=1)
        position_ids = (position_ids.flip(-1) + 1)[None, window_index, :]

        shuffle = []
        offset = 0
        for i, (t, h, w) in enumerate(grid.tolist()):
            t, h, w = int(t), int(h), int(w)
            n_tokens = t * h * w
            n_out_per_frame = (h // factor) * (w // factor)
            perm = torch.arange(h * w)
            perm = perm.view(h // factor, factor, w // factor, factor)
            perm = perm.permute(0, 2, 1, 3).reshape(-1)
            if t > 1:
                frame_offsets = (torch.arange(t) * h * w).view(t, 1)
                perm = (perm.unsqueeze(0) + frame_offsets).reshape(-1)
            vt.register_buffer(f"tt_ds_perm_{i}", perm, persistent=False)
            shuffle.append((offset, n_tokens, t * n_out_per_frame))
            offset += n_tokens

        vt.register_buffer("tt_window_index", window_index, persistent=False)
        vt.register_buffer("tt_reverse_indices", torch.argsort(window_index), persistent=False)
        vt.register_buffer("tt_position_ids", position_ids, persistent=False)
        vt._tt_static = {
            "full_lengths": full_lengths,
            "window_lengths": window_lengths,
            "shuffle": shuffle,
            "split_sizes": (grid.prod(-1) // factor**2).tolist(),
        }

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Build image + text inputs via MuseGlimmerProcessor.

        Args:
            dtype_override: Optional dtype for floating-point image tensors.
            batch_size: Only ``batch_size=1`` is supported (image grids are
                image-specific and do not tile cleanly).
            prompt: Optional text prompt; defaults to ``sample_text``.
            image_url: Optional image URL/path; defaults to ``sample_image_url``.

        Returns:
            dict: ``input_ids``, ``attention_mask``, ``pixel_values``,
            ``image_grid_thw`` (and related processor keys).
        """
        if batch_size != 1:
            raise ValueError(
                "Muse-Glimmer multimodal bring-up only supports batch_size=1 "
                f"(got {batch_size})"
            )

        if self.processor is None:
            self._load_processor()

        # Resolve to a local file so offline/CI hosts still work; pass a PIL
        # image in the chat content (HF Muse demo accepts path/URL/PIL).
        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")
        # Extra bound on pixel area before the processor's smart_resize; keeps
        # vision matmuls off the critical DRAM path on 8-chip meshes.
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        text = prompt or self.sample_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if "image_grid_thw" in inputs:
            self._grid_cpu = inputs["image_grid_thw"].detach().clone().cpu()
            self._build_vision_static()

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def unpack_forward_output(self, fwd_output):
        """Extract logits from MuseGlimmerCausalLMOutputWithPast."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        mesh_shape = (1, num_devices)
        if attn_heads % mesh_shape[1] != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {mesh_shape[1]}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for the Muse-Glimmer text decoder.

        Same ``qwen_3_5``-style map as the causal_lm loader. Vision tower /
        adapter / projection stay replicated (out of the map) for bring-up.
        """
        shard_specs = {}

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

            sa = layer.self_attn
            # k_proj / v_proj replicated: GQA num_key_value_heads=2 cannot split
            # evenly across an 8-wide model axis.
            shard_specs[sa.q_proj.weight] = ("batch", "model")
            shard_specs[sa.gate_proj.weight] = ("batch", "model")
            shard_specs[sa.o_proj.weight] = ("model", "batch")

        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
