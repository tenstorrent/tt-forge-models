# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 8B GGUF model loader implementation for image to text.
"""

import torch
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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


def _find_real_load_gguf():
    """Walk the monkey-patch chain to find the real transformers load_gguf_checkpoint.

    Several loaders install broken wrappers with fixed signatures
    (gguf_path, return_tensors=False) that drop the model_to_load kwarg added
    in transformers 5.2.0. Some wrappers chain via __globals__['_orig_load_gguf_checkpoint'];
    others use a closure variable (e.g. orig_load). We BFS both paths to find the
    first function whose signature explicitly accepts model_to_load.
    """
    import inspect
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    seen_ids: set = set()
    queue = [_gguf_utils.load_gguf_checkpoint]

    while queue:
        fn = queue.pop(0)
        if fn is None or not callable(fn):
            continue
        fn_id = id(fn)
        if fn_id in seen_ids:
            continue
        seen_ids.add(fn_id)

        try:
            sig = inspect.signature(fn)
            if "model_to_load" in sig.parameters:
                return fn
        except (ValueError, TypeError):
            pass

        # Path 1: module-global _orig_load_gguf_checkpoint (most loaders)
        next_g = getattr(fn, "__globals__", {}).get("_orig_load_gguf_checkpoint")
        if next_g is not None and callable(next_g):
            queue.append(next_g)

        # Path 2: closure cells whose names suggest an "orig" or "load" function
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is not None and closure is not None:
            for varname, cell in zip(code.co_freevars, closure):
                if "orig" in varname or "load" in varname:
                    try:
                        val = cell.cell_contents
                        if callable(val):
                            queue.append(val)
                    except ValueError:
                        pass

    return _gguf_utils.load_gguf_checkpoint


def _register_qwen3vl_gguf_architecture():
    """Register qwen3vl GGUF architecture in transformers.

    The MBZUAI MediX-R1-8B GGUF stores general.architecture = "qwen3vl" (no
    underscore). transformers does not have qwen3vl in
    GGUF_SUPPORTED_ARCHITECTURES, so load_gguf_checkpoint raises ValueError.
    Also, get_gguf_hf_weights_map uses hf_model.config.model_type = "qwen3_vl"
    (with underscore), which gguf-py cannot find in MODEL_ARCH_NAMES since it
    expects "qwen3vl".

    Fix:
    1. Add qwen3vl to GGUF_CONFIG_MAPPING (and GGUF_SUPPORTED_ARCHITECTURES).
    2. Patch get_gguf_hf_weights_map to remap "qwen3_vl" -> "qwen3vl".
    3. Install a top-level load_gguf_checkpoint wrapper that properly forwards
       model_to_load (bypassing broken fixed-signature wrappers from other
       loaders that dropped this kwarg added in transformers 5.2.0).
    """
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    if "qwen3vl" in GGUF_CONFIG_MAPPING:
        return

    GGUF_CONFIG_MAPPING["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # Fix (2): patch get_gguf_hf_weights_map
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None and hasattr(hf_model, "config"):
                cfg = hf_model.config
                num_layers = getattr(
                    getattr(cfg, "text_config", cfg), "num_hidden_layers", None
                )
        return _orig_get_map(
            hf_model, processor, model_type=model_type, num_layers=num_layers, qual_name=qual_name
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # Fix (3): install a properly-signed top wrapper that routes model_to_load
    # to the real transformers function while keeping config-loading compat.
    _real_fn = _find_real_load_gguf()
    _broken_chain = _gguf_utils.load_gguf_checkpoint

    def _compat_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
        if kwargs:
            # Weight-loading path (model_to_load present): call real function
            return _real_fn(gguf_path, return_tensors=return_tensors, **kwargs)
        return _broken_chain(gguf_path, return_tensors=return_tensors)

    _gguf_utils.load_gguf_checkpoint = _compat_load_gguf_checkpoint


def _patch_qwen3vl_for_tt_device(model=None):
    """Patch Qwen3 VL methods that call .tolist() on device tensors.

    The test runner moves all input tensors to TT device, but the VisionModel
    and get_rope_index methods call .tolist() on grid_thw / input_ids tensors for
    Python control flow. TT device does not support eager tensor reads.

    fast_pos_embed_interpolate additionally calls torch.tensor(python_list,
    device=self.pos_embed.weight.device).  When the model is on TT device,
    creating a TT-device tensor from a Python list fails during XLA graph
    extraction. We pre-capture pos_embed.weight as a CPU tensor before the
    model is moved to TT (by passing model at call time).

    model: Qwen3VLForConditionalGeneration on CPU, used to capture
           pos_embed.weight before the model is moved to TT device.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        return

    orig_fast_pos = modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate
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

    # Base model provides the processor and correct config
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    # Standard pixel limits for Qwen VL models
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

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
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels

        # Register qwen3vl GGUF architecture before loading
        _register_qwen3vl_gguf_architecture()

        # Load config from the base model so nested text_config/vision_config
        # are correct for Qwen3VLForConditionalGeneration.
        config = Qwen3VLConfig.from_pretrained(self.BASE_MODEL)

        # ignore_mismatched_sizes: the qwen3vl GGUF only contains LM backbone
        # weights. The visual merger norm spuriously claims the 'output_norm' GGUF
        # key, producing a size mismatch (4096 vs 1152). Allowing the mismatch
        # re-inits that visual parameter from the model default; PCC is unaffected.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
        model.eval()

        # Patch .tolist() and torch.tensor(list, device=TT) calls after loading
        # so pos_embed.weight can be captured while the model is still on CPU.
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
