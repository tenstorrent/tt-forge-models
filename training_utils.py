import torch
from typing import Any, Callable, Dict

_ATTR_REGISTRY: Dict[str, str] = {}
_FUNC_REGISTRY: Dict[str, Callable[[Any], torch.Tensor]] = {}

def _register_attr(cls_name: str, attr: str) -> None:
    _ATTR_REGISTRY[cls_name] = attr

def _register_func(cls_name: str, fn: Callable[[Any], torch.Tensor]) -> None:
    _FUNC_REGISTRY[cls_name] = fn

def unpack_output_training(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list, dict)):
        raise ValueError(
            "Tuples/lists/dicts are not handled. "
            "Add a model-specific wrapper or explicit registry rule."
        )

    if isinstance(output, torch.Tensor):
        return output

    cls_name = output.__class__.__name__

    if cls_name in _ATTR_REGISTRY:
        v = getattr(output, _ATTR_REGISTRY[cls_name], None)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(f"{cls_name}.{_ATTR_REGISTRY[cls_name]} is not a torch.Tensor")

    if cls_name in _FUNC_REGISTRY:
        v = _FUNC_REGISTRY[cls_name](output)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(f"Custom func for {cls_name} did not return a torch.Tensor")

    raise ValueError(
        f"unpack_output_training: no handler for {cls_name}. "
        f"Use register_attr('{cls_name}', '<attr>') or register_func('{cls_name}', fn)."
    )

# Example defaults
_register_attr("BaseModelOutputWithPast", "last_hidden_state")
_register_attr("BaseModelOutputWithPastAndCrossAttentions", "last_hidden_state")
_register_attr("CausalLMOutputWithCrossAttentions", "logits")
_register_attr("CausalLMOutputWithPast", "logits")
_register_attr("CLIPOutput", "logits_per_text")
_register_attr("DepthEstimatorOutput", "predicted_depth")
_register_attr("DPRReaderOutput", "end_logits")
_register_attr("ImageClassifierOutput", "logits")
_register_attr("ImageClassifierOutputWithNoAttention", "logits")
_register_attr("LlavaCausalLMOutputWithPast", "logits")
_register_attr("MambaCausalLMOutput", "logits")
_register_attr("MaskedLMOutput", "logits")
_register_attr("MgpstrModelOutput", "logits")
_register_attr("PerceiverClassifierOutput", "logits")
_register_attr("PerceiverMaskedLMOutput", "logits")
_register_attr("SegFormerImageClassifierOutput", "logits")
_register_attr("Seq2SeqLMOutput", "logits")
_register_attr("Seq2SeqSequenceClassifierOutput", "logits")
_register_attr("SequenceClassifierOutput", "logits")
_register_attr("SequenceClassifierOutputWithPast", "logits")
_register_attr("TokenClassifierOutput", "logits")
_register_attr("UNet2DConditionOutput", "sample")
