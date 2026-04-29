# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Reason1 model loader implementation for vision-language reasoning tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional


from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


def _patch_qwen2_5_vl_for_tt_device(model):
    """Patch Qwen2.5 VL methods for TT device compatibility.

    Three classes of issues are fixed:

    1. .tolist() on TT tensors: get_rope_index calls .tolist() on input_ids /
       image_grid_thw. TT device does not support eager tensor reads, so those
       tensors are moved to CPU before the call and the results moved back.

    2. Cross-device gather in the vision transformer: rot_pos_emb builds pos_ids
       on CPU (via grid_thw.tolist()) and then gathers from rotary_pos_emb_full
       which lives on TT. Similarly get_window_index returns a CPU window_index
       that is used to gather TT hidden_states. TT does not support indexing a
       device tensor with CPU indices. Fix: replace the visual transformer forward
       with a reimplementation that computes rot_pos_emb and window_index entirely
       on CPU, then explicitly moves them to TT device before gathering. The CPU
       inv_freq copy is stored at load time (before .to(device)) as a module
       attribute to avoid reading TT device parameters inside the compiled region.

    3. torch.repeat_interleave tile-padding: the visual forward uses
       repeat_interleave to build cu_seqlens. On TT device the output VALUE is
       tile-padded (e.g. 2204→2208), corrupting split_with_sizes in the vision
       attention. Fix: compute cu_seqlens via pure Python arithmetic on CPU grid_thw
       values and move the result to device.
    """
    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
        from transformers.modeling_outputs import BaseModelOutputWithPooling
    except ImportError:
        return

    # Store CPU copy of inv_freq on the visual module at load time (model is on CPU
    # here). This attribute is a plain tensor (not a Parameter/buffer) so .to(device)
    # will NOT move it to TT — it stays CPU throughout the compiled forward.
    vis = model.model.visual
    vis._inv_freq_cpu = vis.rotary_pos_emb.inv_freq.detach().cpu().float()
    vis._inv_freq_orig_dtype = vis.rotary_pos_emb.inv_freq.dtype

    orig_get_rope = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index
    orig_get_image = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features
    orig_get_window = modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_window_index

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

    def _patched_vis_fwd(self, hidden_states, grid_thw, **kwargs):
        # Discard return_dict; we always return BaseModelOutputWithPooling directly.
        kwargs.pop("return_dict", None)

        device = hidden_states.device
        grid_thw_cpu = grid_thw.cpu() if grid_thw.device.type != "cpu" else grid_thw

        # patch_embed runs on TT device with pixel_values on TT
        hidden_states = self.patch_embed(hidden_states)

        # --- rotary positional embeddings (entirely on CPU to avoid cross-device gather) ---
        # Use _inv_freq_cpu stored at load time to avoid reading a TT parameter here.
        pos_ids = []
        for t, h, w in grid_thw_cpu.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw_cpu[:, 1:].max().item())
        inv_freq = self._inv_freq_cpu  # CPU tensor, never moved to TT
        seq = torch.arange(max_grid_size, dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        rotary_pos_emb = freqs[pos_ids].flatten(1)
        # move result to TT device with the model's original dtype
        rotary_pos_emb = rotary_pos_emb.to(dtype=self._inv_freq_orig_dtype, device=device)

        # --- window index (computed on CPU, index moved to device for gather) ---
        window_index_cpu, cu_window_seqlens_list = orig_get_window(self, grid_thw_cpu)
        window_index = window_index_cpu.to(device)

        # Deduplicate in Python (avoids torch.unique_consecutive on TT device which
        # is unsupported). The list comes from orig_get_window which builds it via
        # cumsum so it is already sorted; we just need to remove consecutive dupes.
        unique_cu_window = []
        prev = None
        for v in cu_window_seqlens_list:
            if v != prev:
                unique_cu_window.append(v)
                prev = v
        cu_window_seqlens = torch.tensor(unique_cu_window, device=device, dtype=torch.int32)

        # --- cu_seqlens via pure Python to avoid repeat_interleave tile-padding ---
        cumsum = 0
        cu_seqlens_vals = [0]
        for t, h, w in grid_thw_cpu.tolist():
            for _ in range(int(t)):
                cumsum += int(h) * int(w)
                cu_seqlens_vals.append(cumsum)
        cu_seqlens = torch.tensor(cu_seqlens_vals, dtype=torch.int32, device=device)

        # --- standard vision transformer body ---
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = (
                cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            )
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        # argsort on CPU (window_index_cpu) to avoid TT argsort overhead
        reverse_indices = torch.argsort(window_index_cpu).to(device)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index = _patched_get_rope
    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features = _patched_get_image
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = _patched_vis_fwd


class ModelVariant(StrEnum):
    """Available Cosmos Reason1 model variants for vision-language reasoning tasks."""

    COSMOS_REASON1_7B = "7B"


class ModelLoader(ForgeModel):
    """Cosmos Reason1 model loader implementation for vision-language reasoning tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.COSMOS_REASON1_7B: LLMModelConfig(
            pretrained_model_name="nvidia/Cosmos-Reason1-7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.COSMOS_REASON1_7B

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Cosmos-Reason1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with vision parameters
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cosmos Reason1 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Cosmos Reason1 model instance for vision-language reasoning tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # transformers 5.x no longer accepts use_cache in __init__; set via config
        model.config.text_config.use_cache = False
        # Apply TT device patches after loading (model is on CPU here so inv_freq
        # can be safely captured as a CPU tensor without device-to-host transfer)
        _patch_qwen2_5_vl_for_tt_device(model)
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Cosmos Reason1 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(self.messages)

        # Process all inputs together
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Convert pixel_values to specified dtype if provided
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
