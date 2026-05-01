# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 8B GGUF model loader implementation for image to text.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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


def _register_qwen3vl_gguf_support():
    """Register qwen3vl in transformers GGUF architecture tables.

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF loading
    support for the qwen3vl architecture.  Registers the config field mapping and
    tokenizer converter so that load_gguf_checkpoint can parse the metadata.
    Tensor loading is handled separately in load_model() to bypass the
    model_to_load keyword argument incompatibility in other loaders' patches.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3_vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


_register_qwen3vl_gguf_support()

_TEXT_CONFIG_KEYS = [
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "vocab_size",
    "tie_word_embeddings",
]


def _build_qwen3vl_gguf_tensor_mapping(n_layers):
    """Return a {gguf_name: hf_param_name} dict for Qwen3VL text backbone."""
    m = {
        "token_embd.weight": "model.language_model.embed_tokens.weight",
        "output_norm.weight": "model.language_model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    for i in range(n_layers):
        g = f"blk.{i}."
        h = f"model.language_model.layers.{i}."
        m.update(
            {
                f"{g}attn_q.weight": f"{h}self_attn.q_proj.weight",
                f"{g}attn_k.weight": f"{h}self_attn.k_proj.weight",
                f"{g}attn_v.weight": f"{h}self_attn.v_proj.weight",
                f"{g}attn_output.weight": f"{h}self_attn.o_proj.weight",
                f"{g}attn_q_norm.weight": f"{h}self_attn.q_norm.weight",
                f"{g}attn_k_norm.weight": f"{h}self_attn.k_norm.weight",
                f"{g}ffn_gate.weight": f"{h}mlp.gate_proj.weight",
                f"{g}ffn_up.weight": f"{h}mlp.up_proj.weight",
                f"{g}ffn_down.weight": f"{h}mlp.down_proj.weight",
                f"{g}attn_norm.weight": f"{h}input_layernorm.weight",
                f"{g}ffn_norm.weight": f"{h}post_attention_layernorm.weight",
            }
        )
    return m


def _patch_qwen3vl_for_tt_device(model=None):
    """Patch Qwen3 VL methods that call .tolist() on device tensors.

    TT device does not support eager tensor reads (.tolist() triggers a sync
    that fails with INTERNAL: Error code: 13). Move metadata tensors to CPU
    before the .tolist() calls while keeping all vision/language computation
    on TT device.

    fast_pos_embed_interpolate is reimplemented fully on CPU using a pre-captured
    CPU copy of pos_embed.weight (captured before the model moves to TT device).
    The result is transferred back via xm.send_cpu_data_to_device to avoid
    forcing a premature graph sync.

    model: Qwen3VLForConditionalGeneration on CPU, used to capture
           pos_embed.weight before the model is moved to TT device.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        return

    orig_rot_pos = modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_vl.Qwen3VLModel.get_rope_index
    orig_get_image = modeling_qwen3_vl.Qwen3VLModel.get_image_features

    _pos_embed_weight_cpu = None
    if model is not None:
        try:
            _pos_embed_weight_cpu = (
                model.model.visual.pos_embed.weight.detach().clone().cpu()
            )
        except AttributeError:
            pass

    @torch.compiler.disable
    def _patched_fast_pos(self, grid_thw):
        cpu_weight = _pos_embed_weight_cpu
        if cpu_weight is None:
            cpu_weight = self.pos_embed.weight.detach().cpu()

        grid_thw_list = grid_thw.cpu().tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        weight_tensor = torch.tensor(weight_list, dtype=cpu_weight.dtype)
        pos_embeds = (
            torch.nn.functional.embedding(idx_tensor, cpu_weight)
            * weight_tensor[:, :, None]
        )
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)

        orig_device = self.pos_embed.weight.device
        if str(orig_device) != "cpu":
            import torch_xla.core.xla_model as xm
            [patch_pos_embeds] = xm.send_cpu_data_to_device(
                [patch_pos_embeds], orig_device
            )
        return patch_pos_embeds

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu())

    def _patched_get_rope(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = orig_get_rope(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    def _patched_get_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return orig_get_image(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            **kwargs,
        )

    modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_vl.Qwen3VLModel.get_rope_index = _patched_get_rope
    modeling_qwen3_vl.Qwen3VLModel.get_image_features = _patched_get_image


class ModelVariant(StrEnum):
    """Available MediX R1 8B GGUF model variants for image to text."""

    MEDIX_R1_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-8B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: "MediX-R1-8B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_8B_Q4_K_M

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    # Pixel limits to cap patch count and stay within hardware memory budget
    _min_pixels = 56 * 56
    _max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Install a wide-sig load_gguf_checkpoint wrapper that handles the
        # model_to_load kwarg that other loaders' narrow-sig patches ignore.
        # For config loading (return_tensors=False): delegate to whatever is
        # currently installed (safe for narrow-sig chain).
        # For tensor loading (return_tensors=True): load directly via GGUFReader,
        # bypassing the narrow-sig chain entirely.
        prev_load = gguf_utils.load_gguf_checkpoint

        def _qwen3vl_load_gguf(*args, **kw):
            model_to_load = kw.pop("model_to_load", None)
            return_tensors = kw.get("return_tensors", False)
            if len(args) > 1:
                return_tensors = args[1]

            # Config pass: call through the existing chain (narrow-sig compatible)
            config_kw = dict(kw)
            config_kw["return_tensors"] = False
            config_args = list(args)
            if len(config_args) > 1:
                config_args[1] = False
            result = prev_load(*config_args, **config_kw)

            # Translate qwen3vl flat config → nested Qwen3VLConfig structure
            if result.get("config", {}).get("model_type") == "qwen3vl":
                config = result["config"]
                text_config = {}
                for k in _TEXT_CONFIG_KEYS:
                    if k in config:
                        text_config[k] = config.pop(k)
                config["text_config"] = text_config
                config["model_type"] = "qwen3_vl"
                # The GGUF ships no vision_config; the default out_hidden_size
                # (3584) does not match this model's text hidden_size (4096).
                # The merger must project to text hidden_size, so fix it here.
                config.setdefault("vision_config", {})["out_hidden_size"] = (
                    text_config.get("hidden_size", 4096)
                )

            if return_tensors and model_to_load is not None:
                # Tensor pass: load directly from GGUF, bypassing narrow-sig chain
                gguf_path = args[0] if args else kw.get("gguf_checkpoint_path")
                result["tensors"] = _load_qwen3vl_tensors(gguf_path, model_to_load)

            return result

        gguf_utils.load_gguf_checkpoint = _qwen3vl_load_gguf
        for mod in (config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = _qwen3vl_load_gguf

        try:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs["gguf_file"] = gguf_file
            model_kwargs |= kwargs

            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
            )
            self.processor.image_processor.min_pixels = self._min_pixels
            self.processor.image_processor.max_pixels = self._max_pixels

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            gguf_utils.load_gguf_checkpoint = prev_load
            for mod in (config_utils, modeling_utils):
                if hasattr(mod, "load_gguf_checkpoint"):
                    mod.load_gguf_checkpoint = prev_load

        model.eval()
        # Patch .tolist() calls AFTER loading so pos_embed.weight is captured
        # while the model is still on CPU.
        _patch_qwen3vl_for_tt_device(model=model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs


def _load_qwen3vl_tensors(gguf_path, model_to_load):
    """Load GGUF text-backbone tensors directly, bypassing the patching chain.

    Uses a hard-coded qwen3vl → HF parameter name mapping so we are not
    affected by the get_gguf_hf_weights_map multi-submodule traversal issue
    (visual encoder submodules would otherwise claim 'output_norm' first).
    """
    import numpy as np
    from gguf import GGUFReader, dequantize
    from tqdm.auto import tqdm

    n_layers = model_to_load.config.text_config.num_hidden_layers
    tensor_key_mapping = _build_qwen3vl_gguf_tensor_mapping(n_layers)

    reader = GGUFReader(gguf_path)
    state_dict = {}
    for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
        name = tensor.name
        if name not in tensor_key_mapping:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        hf_name = tensor_key_mapping[name]
        state_dict[hf_name] = torch.from_numpy(np.copy(weights))

    return state_dict
