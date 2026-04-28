# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXagent model loader implementation for chest X-ray vision-language tasks.
"""

import os
import re
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _apply_transformers5_compat() -> None:
    """Stub is_tf_available removed from transformers 5.x.

    transformers 5.x removed TensorFlow support and dropped is_tf_available
    from transformers.utils and transformers.utils.import_utils.  The
    CheXagent tokenizer's remote code imports it and uses it in a
    TYPE_CHECKING guard.  Adding a False-returning stub here allows:
      - check_imports() to skip the guarded `import tensorflow` block, and
      - the `from transformers.utils import is_tf_available` line to succeed.
    """
    import transformers.utils as tu
    import transformers.utils.import_utils as tiu

    if not hasattr(tiu, "is_tf_available"):
        tiu.is_tf_available = lambda: False
        tu.is_tf_available = lambda: False


_ALL_SOURCE_FILES = (
    "modeling_chexagent.py",
    "modeling_visual.py",
    "configuration_chexagent.py",
    "tokenization_chexagent.py",
)


def _patch_model_files(model_name: str) -> None:
    """Apply all transformers 5.x compatibility patches to CheXagent source files.

    Patches applied:
    - modeling_chexagent.py / modeling_visual.py: remove strict transformers==4.40.0
      assertions (purely developer warnings, no functional effect).
    - configuration_chexagent.py: add pad_token_id=None (removed from PretrainedConfig
      defaults in 5.x) and reset auto-populated rope_scaling default dict (model code
      uses the old 4.40 'type'/'factor' format, not the new 'rope_type' format).
    - modeling_visual.py: escape the outer torch.device("meta") context before calling
      the nested AutoModel.from_pretrained for the vision encoder.
    """
    from huggingface_hub import hf_hub_download

    for filename in _ALL_SOURCE_FILES:
        hub_path = Path(hf_hub_download(model_name, filename))
        real_path = Path(os.path.realpath(hub_path))
        _patch_file(real_path, filename)
        _patch_modules_cache_copy(model_name, filename)


def _patch_file(path: Path, filename: str) -> None:
    """Apply all patches appropriate for the given filename."""
    content = path.read_text()
    content = _fix_version_assert(content)
    if filename == "configuration_chexagent.py":
        content = _fix_config_pad_token_id(content)
    if filename == "modeling_visual.py":
        content = _fix_nested_from_pretrained(content)
        content = _fix_visual_forward(content)
    if filename == "tokenization_chexagent.py":
        content = _fix_decode_signature(content)
    if filename == "modeling_chexagent.py":
        content = _fix_pixel_values_input(content)
    path.write_text(content)


def _fix_pixel_values_input(content: str) -> str:
    """Add pixel_values parameter to CheXagentModel and CheXagentForCausalLM forward().

    Under torch.compile, transforms.ToTensor() internally creates a uint8 numpy
    array (graph break) whose resulting uint8 tensor is then passed as a compiled
    subgraph argument.  bridge.extract_compiled_graph segfaults when it tries to
    compile permute/cast ops on a uint8 XLA tensor.

    Fix: accept a pre-processed float pixel_values tensor so the model can call
    self.visual.forward(pixel_values) directly, skipping URL decode + image download.
    """
    if "i_v, a_v, b_v = i.item(), a.item(), b.item()" in content:
        return content

    # 1. Add pixel_values param to CheXagentModel.forward() signature and fix image injection
    old_model_sig = (
        "            output_hidden_states: Optional[bool] = None,\n"
        "            return_dict: Optional[bool] = None,\n"
        "    ) -> Union[Tuple, BaseModelOutputWithPast]:\n"
        "        # IMAGE: encode images\n"
        "        if past_key_values is None and torch.any(input_ids == self.tokenizer.img_start_id):"
    )
    new_model_sig = (
        "            output_hidden_states: Optional[bool] = None,\n"
        "            return_dict: Optional[bool] = None,\n"
        "            pixel_values: Optional[torch.FloatTensor] = None,\n"
        "    ) -> Union[Tuple, BaseModelOutputWithPast]:\n"
        "        # IMAGE: encode images\n"
        "        if pixel_values is not None and past_key_values is None:\n"
        "            bos_pos = torch.where(input_ids == self.tokenizer.img_start_id)\n"
        "            eos_pos = torch.where(input_ids == self.tokenizer.img_end_id)\n"
        "            assert (bos_pos[0] == eos_pos[0]).all()\n"
        "            img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)\n"
        "            images = self.visual.forward(pixel_values.to(dtype=next(self.visual.parameters()).dtype))\n"
        "            fake_images = None\n"
        "        elif past_key_values is None and torch.any(input_ids == self.tokenizer.img_start_id):"
    )
    content = content.replace(old_model_sig, new_model_sig, 1)

    # 1b. Replace in-place slice assignment with torch.cat to avoid XLA view issues.
    # On XLA devices, `hidden_states[i][a+1:b] = images[idx]` uses two-level indexing:
    # `hidden_states[i]` with a 0-D tensor index returns a gather result (not a view),
    # so the subsequent slice-assign does not propagate back to `hidden_states`.
    # Using torch.cat rebuilds the tensor without relying on view semantics.
    old_inject = (
        "        # IMAGE: embed positions\n"
        "        hidden_states = inputs_embeds.clone()\n"
        "        if fake_images is not None:\n"
        "            hidden_states = hidden_states + images.mean() * 0\n"
        "        elif images is not None:\n"
        "            for idx, (i, a, b) in enumerate(img_pos):\n"
        "                hidden_states[i][a + 1: b] = images[idx]"
    )
    new_inject = (
        "        # IMAGE: embed positions\n"
        "        hidden_states = inputs_embeds.clone()\n"
        "        if fake_images is not None:\n"
        "            hidden_states = hidden_states + images.mean() * 0\n"
        "        elif images is not None:\n"
        "            for idx, (i, a, b) in enumerate(img_pos):\n"
        "                i_v, a_v, b_v = i.item(), a.item(), b.item()\n"
        "                row = torch.cat([\n"
        "                    hidden_states[i_v:i_v+1, :a_v+1, :],\n"
        "                    images[idx:idx+1],\n"
        "                    hidden_states[i_v:i_v+1, b_v:, :],\n"
        "                ], dim=1)\n"
        "                hidden_states = torch.cat(\n"
        "                    [hidden_states[:i_v], row, hidden_states[i_v+1:]], dim=0\n"
        "                )"
    )
    content = content.replace(old_inject, new_inject, 1)

    # 2. Add pixel_values param to CheXagentForCausalLM.forward() signature
    old_causal_sig = (
        "            output_hidden_states: Optional[bool] = None,\n"
        "            return_dict: Optional[bool] = None,\n"
        "    ) -> Union[Tuple, CausalLMOutputWithPast]:\n"
    )
    new_causal_sig = (
        "            output_hidden_states: Optional[bool] = None,\n"
        "            return_dict: Optional[bool] = None,\n"
        "            pixel_values: Optional[torch.FloatTensor] = None,\n"
        "    ) -> Union[Tuple, CausalLMOutputWithPast]:\n"
    )
    content = content.replace(old_causal_sig, new_causal_sig, 1)

    # 3. Pass pixel_values through in CheXagentForCausalLM.forward() -> self.model(...)
    old_model_call = (
        "        outputs = self.model(\n"
        "            input_ids=input_ids,\n"
        "            attention_mask=attention_mask,\n"
        "            position_ids=position_ids,\n"
        "            past_key_values=past_key_values,\n"
        "            inputs_embeds=inputs_embeds,\n"
        "            use_cache=use_cache,\n"
        "            output_attentions=output_attentions,\n"
        "            output_hidden_states=output_hidden_states,\n"
        "            return_dict=return_dict,\n"
        "        )\n"
    )
    new_model_call = (
        "        outputs = self.model(\n"
        "            input_ids=input_ids,\n"
        "            attention_mask=attention_mask,\n"
        "            position_ids=position_ids,\n"
        "            past_key_values=past_key_values,\n"
        "            inputs_embeds=inputs_embeds,\n"
        "            use_cache=use_cache,\n"
        "            output_attentions=output_attentions,\n"
        "            output_hidden_states=output_hidden_states,\n"
        "            return_dict=return_dict,\n"
        "            pixel_values=pixel_values,\n"
        "        )\n"
    )
    content = content.replace(old_model_call, new_model_call, 1)

    return content


def _fix_decode_signature(content: str) -> str:
    """Fix _decode() call to pass spaces_between_special_tokens as keyword arg.

    transformers 5.x removed spaces_between_special_tokens from the _decode
    positional parameters.  The old code passes it as the 4th positional arg
    which raises TypeError.  Pass it as a keyword argument instead.
    """
    old = (
        "        return super()._decode(\n"
        "            token_ids, skip_special_tokens, clean_up_tokenization_spaces, spaces_between_special_tokens, **kwargs\n"
        "        )"
    )
    new = (
        "        return super()._decode(\n"
        "            token_ids, skip_special_tokens, clean_up_tokenization_spaces,\n"
        "            spaces_between_special_tokens=spaces_between_special_tokens, **kwargs\n"
        "        )"
    )
    return content.replace(old, new)


def _fix_version_assert(content: str) -> str:
    return re.sub(
        r'^assert transformers\.__version__ == "[^"]+",.*\n',
        "",
        content,
        flags=re.MULTILINE,
    )


def _fix_config_pad_token_id(content: str) -> str:
    """Add pad_token_id=None to CheXagentConfig and reset rope_scaling default dict."""
    if "pad_token_id=None" in content:
        return content
    content = content.replace(
        "bos_token_id=1,\n            eos_token_id=2,\n            **kwargs,",
        "bos_token_id=1,\n            eos_token_id=2,\n            pad_token_id=None,\n            **kwargs,",
    )
    old_super = (
        "        super().__init__(\n"
        "            bos_token_id=bos_token_id,\n"
        "            eos_token_id=eos_token_id,\n"
        "            tie_word_embeddings=tie_word_embeddings,\n"
        "            **kwargs,\n        )"
    )
    new_super = (
        "        super().__init__(\n"
        "            bos_token_id=bos_token_id,\n"
        "            eos_token_id=eos_token_id,\n"
        "            pad_token_id=pad_token_id,\n"
        "            tie_word_embeddings=tie_word_embeddings,\n"
        "            **kwargs,\n        )\n"
        "        if isinstance(self.rope_scaling, dict) and self.rope_scaling.get('rope_type') == 'default':\n"
        "            self.rope_scaling = None"
    )
    return content.replace(old_super, new_super)


def _fix_visual_forward(content: str) -> str:
    """Replace hidden_states[-1] call to work with transformers 5.x.

    In transformers 4.40, SiglipEncoder collected all intermediate hidden_states
    when output_hidden_states=True, so .hidden_states[-1] returned the encoder's
    final pre-post_layernorm output.  In transformers 5.x the encoder no longer
    collects them; the return is always hidden_states=None.

    In both versions, encoder.last_hidden_state is the same pre-post_layernorm
    tensor.  Call embedding and encoder directly to replicate the 4.40 behaviour.
    """
    old = (
        "    def forward(self, x: torch.Tensor):\n"
        "        # get feature\n"
        "        x = self.model(x, output_hidden_states=True).hidden_states[-1]\n"
    )
    new = (
        "    def forward(self, x: torch.Tensor):\n"
        "        # get feature\n"
        "        x = self.model.encoder(inputs_embeds=self.model.embeddings(x, interpolate_pos_encoding=False)).last_hidden_state\n"
    )
    return content.replace(old, new)


def _fix_nested_from_pretrained(content: str) -> str:
    """Escape the meta device context before nested from_pretrained calls.

    transformers 5.x wraps model __init__ in torch.device("meta") to defer
    memory allocation.  CLIPModel.__init__ calls AutoModel.from_pretrained()
    which tries to push a second DeviceContext, conflicting with the outer one.
    Temporarily remove the outer DeviceContext from the TorchFunctionMode stack
    so the inner from_pretrained can push its own.  Push the outer DeviceContext
    back on TOP afterward so torch.device.__exit__ (a C-level simple pop) finds it.

    After loading, re-create position_ids: the SiglipVisionEmbeddings registers it
    as persistent=False so it is absent from the checkpoint and created in the inner
    meta context.  When the outer loader materialises remaining meta tensors the
    buffer gets uninitialized (garbage) data, causing IndexError in embedding lookup.
    """
    if "register_buffer('position_ids'" in content:
        return content
    # Replace the original unpatched code, old broken variants, or the previous
    # correct-order variant that lacked the position_ids fix.
    old_unpatched = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "        self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor"
    )
    old_saved_device = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        import torch as _torch\n"
        "        _saved_device = _torch.get_default_device()\n"
        "        _torch.set_default_device(\"cpu\")\n"
        "        try:\n"
        "            self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "            self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor\n"
        "        finally:\n"
        "            _torch.set_default_device(_saved_device)"
    )
    old_saved_device_with_comment = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        # transformers 5.x enters torch.device(\"meta\") context during from_pretrained\n"
        "        # to avoid allocating memory; nested from_pretrained calls are blocked in that\n"
        "        # context. Temporarily reset to CPU so the sub-model can be loaded.\n"
        "        import torch as _torch\n"
        "        _saved_device = _torch.get_default_device()\n"
        "        _torch.set_default_device(\"cpu\")\n"
        "        try:\n"
        "            self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "            self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor\n"
        "        finally:\n"
        "            _torch.set_default_device(_saved_device)"
    )
    old_stack_wrong_order = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        import torch.utils._device as _tud\n"
        "        from torch._C import _len_torch_function_stack as _lts\n"
        "        from torch.overrides import _pop_mode as _pm, _push_mode as _pushm\n"
        "        _stack = [_pm() for _ in range(_lts())]\n"
        "        _dc = next((_m for _m in _stack if isinstance(_m, _tud.DeviceContext)), None)\n"
        "        _others = [_m for _m in _stack if _m is not _dc]\n"
        "        for _m in reversed(_others): _pushm(_m)\n"
        "        try:\n"
        "            self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "            self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor\n"
        "        finally:\n"
        "            if _dc is not None:\n"
        "                _curr = [_pm() for _ in range(_lts())]\n"
        "                _pushm(_dc)\n"
        "                for _m in reversed(_curr): _pushm(_m)"
    )
    old_stack_correct_order = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        import torch.utils._device as _tud\n"
        "        from torch._C import _len_torch_function_stack as _lts\n"
        "        from torch.overrides import _pop_mode as _pm, _push_mode as _pushm\n"
        "        _stack = [_pm() for _ in range(_lts())]\n"
        "        _dc = next((_m for _m in _stack if isinstance(_m, _tud.DeviceContext)), None)\n"
        "        _others = [_m for _m in _stack if _m is not _dc]\n"
        "        for _m in reversed(_others): _pushm(_m)\n"
        "        try:\n"
        "            self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "            self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor\n"
        "        finally:\n"
        "            if _dc is not None: _pushm(_dc)"
    )
    new = (
        "        super().__init__()\n"
        "        # load model and processor\n"
        "        import torch as _t\n"
        "        import torch.utils._device as _tud\n"
        "        from torch._C import _len_torch_function_stack as _lts\n"
        "        from torch.overrides import _pop_mode as _pm, _push_mode as _pushm\n"
        "        _stack = [_pm() for _ in range(_lts())]\n"
        "        _dc = next((_m for _m in _stack if isinstance(_m, _tud.DeviceContext)), None)\n"
        "        _others = [_m for _m in _stack if _m is not _dc]\n"
        "        for _m in reversed(_others): _pushm(_m)\n"
        "        try:\n"
        "            self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model\n"
        "            _emb = self.model.embeddings\n"
        "            _emb.register_buffer('position_ids', _t.arange(_emb.position_embedding.weight.shape[0]).expand((1, -1)), persistent=False)\n"
        "            self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor\n"
        "        finally:\n"
        "            if _dc is not None: _pushm(_dc)"
    )
    for old in (old_stack_correct_order, old_stack_wrong_order, old_saved_device_with_comment, old_saved_device, old_unpatched):
        if old in content:
            return content.replace(old, new)
    return content


def _patch_modules_cache_copy(model_name: str, filename: str) -> None:
    """Patch a file in the transformers_modules cache, if it is already there."""
    from huggingface_hub import hf_hub_download

    try:
        hub_path = Path(hf_hub_download(model_name, "config.json"))
        snapshot_hash = hub_path.parent.name
    except Exception:
        return

    org, model_id = model_name.split("/", 1)
    encoded_model = model_id.replace("-", "_hyphen_")
    modules_file = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
        / org
        / encoded_model
        / snapshot_hash
        / filename
    )
    if modules_file.exists():
        _patch_file(modules_file, filename)


class ModelVariant(StrEnum):
    """Available CheXagent model variants."""

    CHEXAGENT_2_3B = "chexagent_2_3b"


class ModelLoader(ForgeModel):
    """CheXagent model loader implementation for chest X-ray vision-language tasks."""

    _VARIANTS = {
        ModelVariant.CHEXAGENT_2_3B: ModelConfig(
            pretrained_model_name="StanfordAIMI/CheXagent-2-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEXAGENT_2_3B

    sample_image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    )
    sample_text = "Describe the findings in this chest X-ray."
    sample_system_prompt = "You are a helpful assistant."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._visual_image_transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CheXagent",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        _apply_transformers5_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        _apply_transformers5_compat()
        _patch_model_files(pretrained_model_name)

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # transformers._move_missing_keys_from_meta_to_device unconditionally
        # overwrites ALL persistent=False buffers with empty (garbage) tensors.
        # SiglipVisionEmbeddings.position_ids is persistent=False and absent from
        # the checkpoint, so it ends up with garbage data.  Re-create it here,
        # after from_pretrained is fully done, to get the correct sequential values.
        import torch as _torch
        try:
            _emb = model.model.visual.model.embeddings
            _n = _emb.position_embedding.weight.shape[0]
            _emb.register_buffer(
                "position_ids",
                _torch.arange(_n).expand((1, -1)),
                persistent=False,
            )
        except AttributeError:
            pass

        # The inner AutoModel.from_pretrained for SigLIP loads weights as float32.
        # CLIPModel also creates pos_embed via torch.from_numpy() which may bypass
        # the meta context, leaving it as a real float32 tensor.  The outer
        # from_pretrained uses copy_() for any non-meta tensors, preserving float32.
        # Cast the entire visual module to match the checkpoint dtype (bfloat16 when
        # dtype_override=bfloat16) to prevent mismatches in forward_resampler.
        try:
            _visual = model.model.visual
            _target_dtype = _visual.attn_pool[0].weight.dtype
            _visual.to(_target_dtype)
        except AttributeError:
            pass

        model.eval()

        try:
            self._visual_image_transform = model.model.visual.image_transform
        except AttributeError:
            pass

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        import io

        import requests
        import torch
        from PIL import Image

        if self.tokenizer is None:
            self._load_tokenizer()

        query = self.tokenizer.from_list_format(
            [
                {"image": self.sample_image_url},
                {"text": self.sample_text},
            ]
        )

        conv = [
            {"from": "system", "value": self.sample_system_prompt},
            {"from": "human", "value": query},
        ]

        result = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers 5.x returns BatchEncoding; older versions return a raw tensor
        if hasattr(result, "input_ids"):
            input_ids = result.input_ids
        else:
            input_ids = result

        # Pre-process the image outside of torch.compile to avoid a segfault:
        # transforms.ToTensor() uses np.array(pil, uint8) internally, which is a
        # graph break under torch.compile.  The resulting uint8 tensor passed into
        # the compiled subgraph causes bridge.extract_compiled_graph to segfault
        # when compiling permute/cast ops on a uint8 XLA tensor.
        # By providing pixel_values as a pre-processed float tensor, the model can
        # call self.visual.forward(pixel_values) directly, bypassing URL decode
        # and image download inside the compiled graph.
        pixel_values = None
        if self._visual_image_transform is not None:
            try:
                response = requests.get(self.sample_image_url, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                pixel_values = self._visual_image_transform(pil_image).unsqueeze(0)
                if dtype_override is not None:
                    pixel_values = pixel_values.to(dtype=dtype_override)
            except Exception:
                pixel_values = None

        inputs = {"input_ids": input_ids}
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        return inputs
