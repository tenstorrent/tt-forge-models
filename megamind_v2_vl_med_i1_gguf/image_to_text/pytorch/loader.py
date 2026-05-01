# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Megamind v2 VL med i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from typing import Optional

import numpy as np
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
)


def _patch_qwen3vl_for_tt_device(model=None):
    """Patch Qwen3 VL methods that call .tolist() on TT-device tensors.

    TT device does not support eager tensor readback — .tolist() triggers a
    PJRT sync that fails with Error code: 13. We move the metadata tensors
    (grid_thw, input_ids, attention_mask) to CPU for the control-flow
    computations while keeping the main vision/LM computation on device.

    fast_pos_embed_interpolate is reimplemented entirely on CPU using a
    pre-captured copy of pos_embed.weight (passed via 'model' before the
    model is moved to TT device). The CPU result is transferred back via
    xm.send_cpu_data_to_device to avoid a premature graph sync.
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
            [patch_pos_embeds] = xm.send_cpu_data_to_device([patch_pos_embeds], orig_device)
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


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture and its config key mapping.

    The GGUF file declares architecture 'qwen3vl' but transformers 5.x does
    not list it in GGUF_SUPPORTED_ARCHITECTURES.  The text-decoder layout is
    identical to qwen3, so we reuse its config-key mapping.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        if "qwen3" in mapping:
            mapping.setdefault("qwen3vl", mapping["qwen3"])


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle Qwen3VL composite config.

    Two issues need fixing for Qwen3VL GGUF loading:

    1. Model-type normalisation: the gguf-py MODEL_ARCH_NAMES dict uses
       'qwen3vl' (no underscore) but the transformers model_type is 'qwen3_vl'.

    2. Visual sub-module key stealing: the qwen3vl name_map also matches
       vision encoder sub-modules (e.g. merger.norm) against text-decoder GGUF
       names (e.g. output_norm).  When the vision encoder is traversed before
       the language model, it claims keys like output_norm.weight and the text
       decoder's norm ends up unmapped.  Fix by stripping visual entries from
       the map and re-adding the language-model mappings.
    """
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"

    if num_layers is None:
        cfg = hf_model.config
        num_layers = getattr(cfg, "num_hidden_layers", None)
        if num_layers is None:
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                num_layers = getattr(text_cfg, "num_hidden_layers", None)

    # Top-level Qwen3VL call: restrict mapping to the text-decoder sub-tree.
    if (
        model_type == "qwen3vl"
        and qual_name == ""
        and hasattr(getattr(hf_model, "config", None), "text_config")
    ):
        # Build the full map first; vision sub-modules may steal text-decoder
        # GGUF tensor names because the name_map is architecture-agnostic.
        full_map = _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )
        # Remove any visual entries — the GGUF contains no vision weights.
        clean_map = {k: v for k, v in full_map.items() if "visual" not in v}

        # Re-add language-model entries whose keys were stolen by visual modules.
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
            lm_map = _orig_get_gguf_hf_weights_map(
                hf_model.model.language_model,
                processor,
                model_type,
                num_layers,
                "model.language_model.",
            )
            for k, v in lm_map.items():
                if k not in clean_map:
                    clean_map[k] = v

        return clean_map

    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


# Install arch support and weight-map patch at import time.
_patch_qwen3vl_gguf_support()
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


def _load_tensors_from_gguf(gguf_path, model_to_load):
    """Load and dequantize GGUF tensors into model_to_load's parameter space.

    Many other GGUF-loader modules patch transformers.load_gguf_checkpoint
    with a narrow signature that does not accept the model_to_load kwarg
    added in transformers 5.2.0.  This function bypasses that patched chain
    by implementing the tensor-loading pass directly.
    """
    from gguf import GGUFReader, dequantize
    from transformers.modeling_gguf_pytorch_utils import TensorProcessor
    from tqdm import tqdm

    _patch_qwen3vl_gguf_support()

    reader = GGUFReader(gguf_path)
    processor = TensorProcessor(config={})
    tensor_key_mapping = _patched_get_gguf_hf_weights_map(model_to_load, processor)

    tensors = {}
    for tensor in tqdm(reader.tensors, desc="Loading GGUF tensors..."):
        name = tensor.name
        if name not in tensor_key_mapping:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        tensors[tensor_key_mapping[name]] = torch.from_numpy(np.copy(weights))

    return tensors


class ModelVariant(StrEnum):
    """Available Megamind v2 VL med i1 GGUF model variants for image to text."""

    MEGAMIND_V2_VL_MED_I1_GGUF = "v2_vl_med_i1_gguf"


class ModelLoader(ForgeModel):
    """Megamind v2 VL med i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEGAMIND_V2_VL_MED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Megamind-v2-VL-med-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEGAMIND_V2_VL_MED_I1_GGUF

    GGUF_FILE = "Megamind-v2-VL-med.i1-Q4_K_M.gguf"

    # The GGUF repo ships only text-decoder weights; use the base model for
    # the full architecture config and the processor.
    BASE_MODEL = "digitranslab/Megamind-v2-VL-med"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Megamind v2 VL med i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import huggingface_hub

        pretrained_model_name = self._variant_config.pretrained_model_name

        # GGUF repos do not ship a processor; use the base model.
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        self.processor.image_processor.min_pixels = 56 * 56
        self.processor.image_processor.max_pixels = 13 * 28 * 1280

        # The GGUF repo has no config.json.  Load the full VL architecture
        # config (36 text-decoder layers, correct dims, vision encoder) from
        # the base model.
        config = AutoConfig.from_pretrained(self.BASE_MODEL)

        # Instantiate model from the base config.  The text-decoder weights
        # are provided by the GGUF; vision encoder weights are randomly
        # initialised (the GGUF only contains the text decoder).
        model = Qwen3VLForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Resolve the local GGUF path (downloads if not cached).
        gguf_path = huggingface_hub.hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )

        # Load text-decoder weights from GGUF directly, bypassing the
        # patched load_gguf_checkpoint chain (which uses a narrow signature
        # that does not support the model_to_load kwarg).
        state_dict = _load_tensors_from_gguf(gguf_path, model)
        model.load_state_dict(state_dict, strict=False)

        # Patch .tolist() calls AFTER loading while model is still on CPU so
        # pos_embed.weight can be captured as a CPU tensor.
        _patch_qwen3vl_for_tt_device(model=model)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
