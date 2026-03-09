#!/usr/bin/env python3
from __future__ import annotations

import types
import warnings

import torch
from einops import rearrange
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from fla.layers.rwkv7 import (
    RWKV7Attention,
    chunk_rwkv7,
    fused_addcmul_rwkv7,
    fused_k_rwkv7,
    fused_mul_recurrent_rwkv7,
    gate_output_correction,
    l2_norm,
    token_shift,
)


def inject_future_seed(model, layer_start: int = 1, alpha_init: float = -2.0):
    decoder = getattr(model, "model", model)
    if getattr(decoder, "_future_seed_patched", False):
        decoder._future_seed_layer_start = int(layer_start)
        return model

    for layer in decoder.layers:
        attn = getattr(layer, "attn", None)
        if not isinstance(attn, RWKV7Attention):
            continue
        if not hasattr(attn, "future_seed_alpha"):
            attn.register_parameter(
                "future_seed_alpha",
                torch.nn.Parameter(torch.full((1, attn.num_heads, 1, 1), float(alpha_init))),
            )
        attn._future_seed_original_forward = attn.forward

        def patched_attn_forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            v_first=None,
            cu_seqlens=None,
            future_seed_state=None,
            **kwargs,
        ):
            if past_key_values is not None or use_cache:
                return self._future_seed_original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    cu_seqlens=cu_seqlens,
                    **kwargs,
                )

            batch_size, seq_len, _ = hidden_states.shape
            if attention_mask is not None:
                assert len(attention_mask.shape) == 2
                am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)
                hidden_states = hidden_states.mul(am)
            else:
                am = None

            conv_cache = None
            recurrent_state = future_seed_state

            delta, conv_state = token_shift(hidden_states, cu_seqlens, output_cache=True, cache=conv_cache)
            xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
                hidden_states, delta, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g,
            )

            r = self.r_proj(xr)
            w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
            k = self.k_proj(xk)
            v = self.v_proj(xv)

            if self.layer_idx == 0:
                v_first = v
            else:
                v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
            a = self.a_lora(xa).sigmoid()
            g = self.g_lora(xg)

            if self.fuse_norm:
                kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
            else:
                kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
            k = fused_k_rwkv7(k, a, self.k_a)

            if am is not None:
                v = v * am

            r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

            if self.training or seq_len >= 64:
                o, recurrent_state = chunk_rwkv7(
                    r=r,
                    w=w,
                    k=k,
                    v=v,
                    a=-kk,
                    b=kk * a,
                    scale=1.,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, recurrent_state = fused_mul_recurrent_rwkv7(
                    r=r,
                    w=w,
                    k=k,
                    v=v,
                    kk=kk,
                    a=a,
                    scale=1.,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )

            if self.fuse_norm:
                o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
            else:
                o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

            o = gate_output_correction(o, r, k, self.r_k, v, g)
            o = self.o_proj(o)
            self._future_seed_last_state = recurrent_state
            return o, None, None, v_first

        attn.forward = types.MethodType(patched_attn_forward, attn)

    decoder._future_seed_original_forward = decoder.forward

    def patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cu_seqlens=None,
        **kwargs,
    ):
        if output_attentions:
            warnings.warn("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, 'output_attentions', False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, 'output_hidden_states', False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("must specify input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        v_first = torch.zeros_like(hidden_states)
        seed_state = None
        layer_start = int(getattr(self, "_future_seed_layer_start", 1))

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            future_seed_state = None
            attn = getattr(layer, 'attn', None)
            if idx >= layer_start and seed_state is not None and isinstance(attn, RWKV7Attention):
                gate = torch.sigmoid(attn.future_seed_alpha).to(seed_state.dtype)
                future_seed_state = seed_state * gate
            hidden_states, _, _, v_first = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                v_first=v_first,
                cu_seqlens=cu_seqlens,
                future_seed_state=future_seed_state,
                **kwargs,
            )
            if isinstance(attn, RWKV7Attention):
                seed_state = getattr(attn, "_future_seed_last_state", None)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(i for i in [hidden_states, None, all_hidden_states, None] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    decoder.forward = types.MethodType(patched_model_forward, decoder)
    decoder._future_seed_patched = True
    decoder._future_seed_layer_start = int(layer_start)
    return model


def mark_future_seed_trainable(model):
    for name, param in model.named_parameters():
        if "future_seed_alpha" in name:
            param.requires_grad = True
