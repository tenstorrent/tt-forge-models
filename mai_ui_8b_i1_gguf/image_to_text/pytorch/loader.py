# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAI-UI-8B i1 GGUF model loader implementation for image to text.
"""
import importlib.metadata

import torch
from transformers import (
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


def _patch_transformers_qwen3vl_gguf():
    """Register qwen3vl as a supported GGUF architecture in transformers.

    Transformers 5.x supports qwen3 but not qwen3vl. We add the config
    mapping (identical to qwen3) and remap model_type to qwen3_vl so
    AutoConfig resolves to Qwen3VLConfig.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen3"].copy()

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            text_hidden = config.get("hidden_size")
            if text_hidden is not None:
                config.setdefault("vision_config", {})
                config["vision_config"]["out_hidden_size"] = text_hidden
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available MAI-UI-8B i1 GGUF model variants for image to text."""

    MAI_UI_8B_I1_GGUF = "mai_ui_8b_i1_gguf"


class ModelLoader(ForgeModel):
    """MAI-UI-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MAI_UI_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/MAI-UI-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAI_UI_8B_I1_GGUF

    GGUF_FILE = "MAI-UI-8B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MAI-UI-8B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_gguf_version_detection():
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Tongyi-MAI/MAI-UI-8B")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self._patch_vision_repeat(model)

        return model

    @staticmethod
    def _patch_vision_repeat(model):
        """Patch vision model to avoid repeat(1, 1) which lowers to an
        empty StableHLO Concatenate and crashes the TT compiler."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel,
        )

        orig_method = Qwen3VLVisionModel.fast_pos_embed_interpolate

        def _patched_fast_pos_embed_interpolate(self, grid_thw):
            grid_thw_list = grid_thw.tolist()
            grid_ts = [row[0] for row in grid_thw_list]
            grid_hs = [row[1] for row in grid_thw_list]
            grid_ws = [row[2] for row in grid_thw_list]
            device = self.pos_embed.weight.device

            idx_list = [[] for _ in range(4)]
            weight_list = [[] for _ in range(4)]

            for t, h, w in grid_thw_list:
                h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h))
                w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w))

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

            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            weight_tensor = torch.tensor(
                weight_list, dtype=self.pos_embed.weight.dtype, device=device
            )
            pos_embeds = (
                self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
            )
            patch_pos_embeds = (
                pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
            )

            patch_pos_embeds = patch_pos_embeds.split(
                [int(h * w) for h, w in zip(grid_hs, grid_ws)]
            )

            patch_pos_embeds_permute = []
            merge_size = self.config.spatial_merge_size
            for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
                t, h, w = int(t), int(h), int(w)
                if t > 1:
                    pos_embed = pos_embed.repeat(t, 1)
                pos_embed = (
                    pos_embed.view(
                        t,
                        h // merge_size,
                        merge_size,
                        w // merge_size,
                        merge_size,
                        -1,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .flatten(0, 4)
                )
                patch_pos_embeds_permute.append(pos_embed)
            patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
            return patch_pos_embeds

        Qwen3VLVisionModel.fast_pos_embed_interpolate = (
            _patched_fast_pos_embed_interpolate
        )

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
