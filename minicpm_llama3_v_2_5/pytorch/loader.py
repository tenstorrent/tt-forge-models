# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2.5 model loader implementation for multimodal visual question answering.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Optional

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
from ...tools.utils import get_file


class _MiniCPMVWrapper(torch.nn.Module):
    """Wraps MiniCPMV to accept **kwargs instead of a single data dict.

    MiniCPMV.forward(self, data, **kwargs) expects a dict as its first
    positional argument. The test framework spreads inputs as keyword
    arguments, so this wrapper repacks them into the expected dict form.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._wrapped = model

    def forward(self, **data):
        return self._wrapped(data)


class ModelVariant(StrEnum):
    """Available MiniCPM-Llama3-V-2.5 model variants."""

    MINICPM_LLAMA3_V_2_5 = "MiniCPM-Llama3-V-2.5"


class ModelLoader(ForgeModel):
    """MiniCPM-Llama3-V-2.5 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.MINICPM_LLAMA3_V_2_5: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-Llama3-V-2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_LLAMA3_V_2_5

    sample_text = "What is in the image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize MiniCPM-Llama3-V-2.5 model loader."""
        super().__init__(variant)
        self.tokenizer = None
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-Llama3-V-2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    @staticmethod
    def _patch_resampler(model_name: str) -> None:
        """Patch the cached resampler.py for known issues in all published revisions.

        1. Missing ``List`` import from typing (causes NameError on PyTorch ≥ 2.9).
        2. ``reshape((tgt_h * tgt_w, -1))`` fails when tgt_h=0 during Dynamo fake-tensor
           tracing because ''-1'' is ambiguous for a 0-element tensor.  Replace with
           ``reshape((-1, self.pos_embed.shape[-1]))`` which resolves to (0, D) safely.
        3. Decorate ``Resampler.forward`` with ``@torch._dynamo.disable`` so that Dynamo
           does not trace into the resampler during compilation.  The resampler uses
           data-dependent shapes throughout (per-tile slicing, ``torch.max(patch_len)``
           as a tensor dimension) that are fundamentally incompatible with static tracing.
           With the decorator, the entire ``forward`` runs eagerly and receives real
           tensor values, avoiding all shape-mismatch errors.
        """
        try:
            from transformers.dynamic_module_utils import get_cached_module_file

            resampler_path = Path(get_cached_module_file(model_name, "resampler.py"))
            content = resampler_path.read_text()
            changed = False
            if (
                "from typing import Optional, Tuple" in content
                and "from typing import List" not in content
            ):
                content = content.replace(
                    "from typing import Optional, Tuple",
                    "from typing import List, Optional, Tuple",
                )
                changed = True
            if ".reshape((tgt_h * tgt_w, -1))" in content:
                content = content.replace(
                    ".reshape((tgt_h * tgt_w, -1))",
                    ".reshape((-1, self.pos_embed.shape[-1]))",
                )
                changed = True
            if (
                "    def forward(self, x, tgt_sizes=None):" in content
                and "@torch._dynamo.disable" not in content
            ):
                content = content.replace(
                    "    def forward(self, x, tgt_sizes=None):",
                    "    @torch._dynamo.disable\n    def forward(self, x, tgt_sizes=None):",
                )
                changed = True
            if changed:
                resampler_path.write_text(content)
                # Invalidate any cached import of this module so the patched version
                # is picked up on the next import via AutoModel.from_pretrained.
                import sys

                for key in list(sys.modules.keys()):
                    if "resampler" in key and "transformers_modules" in key:
                        del sys.modules[key]
        except Exception:
            pass

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2.5 model instance."""
        model_name = self._variant_config.pretrained_model_name
        self._patch_resampler(model_name)
        model = AutoModel.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        # Disable batch vision input: the batch path uses torch.max(tgt_sizes)
        # as a tensor dimension which is a data-dependent shape that TorchDynamo
        # cannot trace. The per-tile path avoids patch_attention_mask entirely.
        model.config.batch_vision_input = False

        # Wrap so the test framework can call model(**inputs_dict) and forward
        # receives the dict as the required `data` positional argument.
        return _MiniCPMVWrapper(model)

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self._processor

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MiniCPM-Llama3-V-2.5."""
        if self._processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        # Build messages in the format expected by the processor (mirrors chat() logic)
        msgs = [{"role": "user", "content": [image, self.sample_text]}]
        images = []
        for msg in msgs:
            content = msg["content"]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self._processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(prompt, images, return_tensors="pt")

        # Ensure int64 for embedding layer compatibility
        inputs["input_ids"] = inputs["input_ids"].long()

        # forward() requires position_ids (not returned by processor)
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Return as a plain dict. The test framework spreads this as **kwargs to the
        # wrapped model, which repacks it into the `data` dict that MiniCPMV.forward
        # expects.
        return dict(inputs)

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, str):
            return outputs

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
