# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ModelOutputWithPast,
    token_type_ids_mask_function,
)


# Original forward:
# https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/gemma3/modeling_gemma3.py#L836-L977
def patched_forward(
    self,
    input_ids=None,
    pixel_values=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    token_type_ids=None,
    cache_position=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **lm_kwargs,
):

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and self.config.image_token_id >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )

        # Decomposed masked_scatter to avoid introduction of dynamic shapes.
        # https://github.com/tenstorrent/tt-xla/issues/3977
        mask_flat = special_image_mask.reshape(-1)
        data_flat = inputs_embeds.reshape(-1)
        source_flat = image_features.reshape(-1)
        # Convert bool mask to int for cumsum
        mask_i = mask_flat.long()
        # Expand source to same size as data:
        # cumsum counts Trues seen so far -> becomes index into source
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        # Gather source values for all positions (dummy values at False positions)
        gathered = source_flat[source_idx]
        # Pick: True -> source value, False -> keep original
        result_flat = torch.where(mask_flat, gathered, data_flat)
        inputs_embeds = result_flat.view_as(inputs_embeds)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config.get_text_config(),
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        is_prefill = (
            not use_cache
            or past_key_values is None
            or not past_key_values.is_initialized
            or pixel_values is not None
        )
        if token_type_ids is not None and is_prefill:
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = (
                is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            )
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(
                is_image,
                image_group_ids,
                torch.full_like(token_type_ids, -1, device=is_image.device),
            )
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device),
                image_group_ids,
                self.config.mm_tokens_per_image,
            )

        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    outputs = self.language_model(
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **lm_kwargs,
    )

    return Gemma3ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values if use_cache else None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )
