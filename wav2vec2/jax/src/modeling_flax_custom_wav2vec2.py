# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

""" Custom Flax Wav2Vec2 model """

from transformers.models.wav2vec2.modeling_flax_wav2vec2 import (
    FlaxWav2Vec2Model,
    FlaxWav2Vec2Attention,
    FlaxWav2Vec2BaseModelOutput,
    FlaxWav2Vec2PreTrainedModel,
    FlaxWav2Vec2FeatureProjection,
    FlaxWav2Vec2FeedForward,
    FlaxWav2Vec2PositionalConvEmbedding,
    FlaxWav2Vec2LayerNormConvLayer,
    FlaxWav2Vec2StableLayerNormEncoder,
    FlaxWav2Vec2Adapter,
    Wav2Vec2Config,
)

from typing import Tuple, Optional, Union
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN


class FlaxWav2Vec2NoLayerNormConvLayer(nn.Module):
    """Custom convolutional layer without layer normalization."""

    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = (
            self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        )
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],
            kernel_size=(self.config.conv_kernel[self.layer_id],),
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.feat_extract_norm == "layer":
            self.layers = [
                FlaxWav2Vec2LayerNormConvLayer(
                    self.config, layer_id=i, name=str(i), dtype=self.dtype
                )
                for i in range(self.config.num_feat_extract_layers)
            ]
        elif self.config.feat_extract_norm == "group":
            self.layers = [
                FlaxWav2Vec2LayerNormConvLayer(
                    self.config, layer_id=0, name=str(0), dtype=self.dtype
                )
            ] + [
                FlaxWav2Vec2NoLayerNormConvLayer(
                    self.config, layer_id=i + 1, name=str(i + 1), dtype=self.dtype
                )
                for i in range(self.config.num_feat_extract_layers - 1)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group',"
                " 'layer']"
            )

    def __call__(self, hidden_states):
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class FlaxWav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, input_values, freeze_feature_encoder=False):
        hidden_states = input_values[:, :, None]
        hidden_states = self.conv_layers(hidden_states)
        if freeze_feature_encoder:
            hidden_states = jax.lax.stop_gradient(hidden_states)
        return hidden_states


class FlaxWav2Vec2EncoderStableLayerNormPostAttention(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.attention = FlaxWav2Vec2Attention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
    ):
        attn_residual = hidden_states

        if jnp.all(hidden_states == 0):
            print(f"Warning: All zero input to encoder layer")

        # Self attention
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        if jnp.all(hidden_states == 0):
            print(f"Warning: All zero output from encoder layer")

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxWav2Vec2LayerNormPostAttentionCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxWav2Vec2EncoderStableLayerNormPostAttention(
                self.config, name=str(i), dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if jnp.all(hidden_states == 0):
            print(f"Warning: All zero input to encoder")

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if jnp.all(hidden_states == 0):
                print(f"Warning: All zero output from layer {i}")

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxWav2Vec2LayerNormPostAttention(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pos_conv_embed = FlaxWav2Vec2PositionalConvEmbedding(
            self.config, dtype=self.dtype
        )
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # Use the fixed layer collection
        self.layers = FlaxWav2Vec2LayerNormPostAttentionCollection(
            self.config, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        if jnp.all(hidden_states == 0):
            print(f"Warning: All zero input to LayerNormPostAttention")

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states = jnp.where(
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape),
                hidden_states,
                0,
            )

        position_embeddings = self.pos_conv_embed(hidden_states)

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        if jnp.all(hidden_states == 0):
            print(f"Warning: All zero after preprocessing in LayerNormPostAttention")

        outputs = self.layers(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (
                outputs[2:] if output_hidden_states else outputs[1:]
            )
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


# Inherit from the original model and only override the encoder part
class FlaxWav2Vec2CustomModule(FlaxWav2Vec2Model.module_class):
    """Custom module that inherits most functionality but uses custom encoder."""

    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(
            self.config, dtype=self.dtype
        )
        self.feature_projection = FlaxWav2Vec2FeatureProjection(
            self.config, dtype=self.dtype
        )
        self.masked_spec_embed = self.param(
            "masked_spec_embed",
            jax.nn.initializers.uniform(),
            (self.config.hidden_size,),
        )

        if self.config.do_stable_layer_norm:
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(
                self.config, dtype=self.dtype
            )
        else:
            self.encoder = FlaxWav2Vec2LayerNormPostAttention(
                self.config, dtype=self.dtype
            )

        self.adapter = (
            FlaxWav2Vec2Adapter(self.config, dtype=self.dtype)
            if self.config.add_adapter
            else None
        )

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        extract_features = self.feature_extractor(
            input_values, freeze_feature_encoder=freeze_feature_encoder
        )

        # make sure that no loss is computed on padded inputs
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(
            extract_features, deterministic=deterministic
        )
        if (
            mask_time_indices is not None
        ):  # apply SpecAugment along time axis with given indices
            hidden_states = jnp.where(
                jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape),
                jnp.broadcast_to(
                    self.masked_spec_embed[None, None, :], hidden_states.shape
                ),
                hidden_states,
            )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return FlaxWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, 1, self.config.adapter_stride
                )

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(
            non_padded_lengths, add_adapter=add_adapter
        )

        batch_size = attention_mask.shape[0]

        attention_mask = jnp.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype
        )
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        attention_mask = attention_mask.at[
            jnp.arange(attention_mask.shape[0]), output_lengths - 1
        ].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype(
            "bool"
        )
        return attention_mask


class FlaxWav2Vec2CustomModel(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2CustomModule


class FlaxWav2Vec2ForCTCCustomModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.wav2vec2 = FlaxWav2Vec2CustomModule(self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.final_dropout)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        freeze_feature_encoder: bool = False,
        return_dict=None,
        **kwargs,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxWav2Vec2ForCTCCustom(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2ForCTCCustomModule
