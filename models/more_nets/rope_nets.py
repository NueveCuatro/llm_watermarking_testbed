import torch
from torch import nn
from typing import Optional, Callable, Union
from transformers.cache_utils import Cache, EncoderDecoderCache, DynamicCache
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
"""
This aims to overwrite the forward passes for GPT2Attention and GPT2Model, in order to add rotary positional embdding
"""

def _rotate_half(x):
        # [B,H,T,Drot] -> split even/odd and rotate
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin, rotary_dim):
    # q,k: [B,H,T,D]; cos,sin: [T, 1, rotary_dim] (broadcastable)
    # apply on first rotary_dim dims, leave the tail untouched
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    # broadcast [B,H,T,rotary_dim] with [T,1,rotary_dim]
    q_rot = (q1 * cos) + (_rotate_half(q1) * sin)
    k_rot = (k1 * cos) + (_rotate_half(k1) * sin)
    return torch.cat([q_rot, q2], dim=-1), torch.cat([k_rot, k2], dim=-1)

def build_rope_cache(max_len, rotary_dim, base, device, dtype, scale=None):
    # angles: theta = base^(−2i/rotary_dim), i = 0..rotary_dim/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
    t = torch.arange(max_len, device=device, dtype=dtype)  # positions
    freqs = torch.einsum('t,f->tf', t, inv_freq)           # [T, D/2]
    if scale is not None:
        freqs = freqs / scale
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)#.unsqueeze(1)  # [T,1,D]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)#.unsqueeze(1)  # [T,1,D]
    return cos, sin

def build_position_ids_prime_cumulative_from_posbase(pos_base, wm_applied, key_vec):
        """
        pos_base:   [B, T] long/int positions for the tokens processed in THIS attention call
                (e.g., [past_len .. past_len+T-1] or cache_position)
        wm_applied: [B] bool/int (1 => watermark ON for that sample)
        key_vec:    [K] int displacements per segment (delta_s)

        Returns:
            pos_prime: [B, T] Long
        """
        assert pos_base.dim() == 2, f"pos_base must be [B,T], got {pos_base.shape}"
        B, T = pos_base.shape
        device = pos_base.device

        # tensors + device
        if not torch.is_tensor(wm_applied):
            wm_applied = torch.tensor(wm_applied, device=device)
        else:
            wm_applied = wm_applied.to(device)

        if not torch.is_tensor(key_vec):
            key_vec = torch.tensor(key_vec, device=device)
        else:
            key_vec = key_vec.to(device)

        key_vec = key_vec.long()
        K = key_vec.numel()
        assert K >= 1, "key_vec must have at least 1 element"

        # segment id based on token index inside the chunk: 0..T-1
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
        seg_id = (t_idx * K) // T                                         # [B,T] in [0..K-1]

        # prefix sums then shift-by-one: seg0 -> 0, seg1 -> k0, seg2 -> k0+k1, ...
        prefix = torch.cumsum(key_vec, dim=0)                             # [K]
        prefix_shift = torch.cat(
            [torch.zeros(1, device=device, dtype=prefix.dtype), prefix[:-1]],
            dim=0
        )                                                                 # [K]

        delta = prefix_shift[seg_id]                                      # [B,T]

        # apply watermark per sample
        wm = wm_applied.long().unsqueeze(1)                               # [B,1]
        delta = delta * wm                                                # [B,T]

        pos_prime = pos_base.long() + delta                               # [B,T]
        return pos_prime.long()

class GPT2RopeAdapter:
    """Patches HF GPT-2 to use RoPE (rotary) instead of absolute wpe."""
    def supports(self, hfmodel) -> bool:
        return getattr(hfmodel.config, "model_type", "") == "gpt2"

    def add_rope(self, hfmodel, *, theta=10000.0, rotary_dim: Optional[int]=None,
                 scale: Optional[float]=None, cache_max_len: int=4096):
        """
        Modifies the model in-place:
        - bypasses wpe addition,
        - injects RoPE into GPT2Attention (Q,K),
        - installs per-layer cos/sin caches.
        """
        transformer = hfmodel.transformer
        cfg = hfmodel.config
        n_head = cfg.n_head
        head_dim = cfg.n_embd // n_head
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        rotary_dim = min(rotary_dim, head_dim)

        # 1) Disable adding absolute position embeddings (wpe)
        self._bypass_wpe(transformer)

        # 2) Patch attention in every block to rotate Q,K
        for layer_idx, block in enumerate(transformer.h):
            self._patch_attention(block.attn, layer_idx, theta, rotary_dim, scale, cache_max_len, n_head, head_dim)

        # 3) record in config so you can reconstruct at test time
        setattr(cfg, "use_rope_watermark", True)
        setattr(cfg, "rope_theta", float(theta))
        setattr(cfg, "rope_dim", int(rotary_dim))
        if scale is not None:
            setattr(cfg, "rope_scale", float(scale))

    # ---------- internals ----------

    def _bypass_wpe(self, transformer):
        """Monkey-patch GPT2Model.forward to ignore wpe(position_ids)."""
        # orig_forward = transformer.__class__.forward
        
        def forward_no_wpe(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[tuple[tuple[torch.Tensor]], Cache]] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None, 
            return_dict: Optional[bool] = None,
            **kwargs,
        ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
            r"""
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
                `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
                `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
                sequence tokens in the vocabulary.

                If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
                `input_ids`.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            """
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    print.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

            # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
            return_legacy_cache = False
            if use_cache:
                if past_key_values is None:
                    return_legacy_cache = True
                    past_key_values = DynamicCache()
                elif not isinstance(past_key_values, Cache):
                    return_legacy_cache = True
                    print.warning_once(
                        "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                        "You should pass an instance of `Cache` instead, e.g. "
                        "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                    )
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds #+ position_embeds.to(inputs_embeds.device)
            #no positional embedding because we are using Rope

            # Attention mask.
            # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
            if attention_mask is not None and attention_mask.ndim < 4:
                attention_mask = attention_mask.view(batch_size, -1)
                
                causal_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
            if self.config.add_cross_attention and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                if _use_sdpa:
                    encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                        mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                    )
                elif not self._attn_implementation == "flash_attention_2":
                    encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)

            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds

            hidden_states = self.drop(hidden_states)

            output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
            all_hidden_states = () if output_hidden_states else None
            for i, block in enumerate(self.h):
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                outputs = block(
                    hidden_states,
                    past_key_values if not (self.gradient_checkpointing and self.training) else None,
                    cache_position,
                    causal_mask,
                    head_mask[i],
                    encoder_hidden_states,  # as a positional argument for gradient checkpointing
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )

                hidden_states = outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

            hidden_states = self.ln_f(hidden_states)

            hidden_states = hidden_states.view(output_shape)
            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_values = past_key_values if use_cache else None
            if return_legacy_cache:
                past_key_values = (
                    past_key_values.self_attention_cache.to_legacy_cache()
                    if self.config.add_cross_attention
                    else past_key_values.to_legacy_cache()
                )
            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                    if v is not None
                )

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )

        # Bind the method
        transformer.forward = forward_no_wpe.__get__(transformer, transformer.__class__)

    # def _patch_attention(self,
    #                      attn_mod : nn.Module,
    #                      layer_idx : int,
    #                      theta : int,
    #                      rotary_dim : Optional[int],
    #                      scale : Optional[float],
    #                      cache_max_len : Optional[int],
    #                      n_head : int,
    #                      head_dim : int) -> None:
    #     """
    #     Replace GPT2Attention.forward with a version that:
    #     - computes Q,K,V as usual,
    #     - applies RoPE to Q and K on the first 'rotary_dim' dims,
    #     - handles past_key_value (KV cache) correctly.
    #     """
    #     orig_forward = attn_mod.forward

    #     # buffers per layer (lazy init on first forward)
    #     attn_mod.register_buffer("rope_cos", None, persistent=False)
    #     attn_mod.register_buffer("rope_sin", None, persistent=False)
    #     attn_mod._rope_meta = dict(theta=theta, rotary_dim=rotary_dim, scale=scale,
    #                                cache_max_len=cache_max_len, n_head=n_head, head_dim=head_dim)

    #     def forward_with_rope(
    #         self,
    #         hidden_states: Optional[tuple[torch.FloatTensor]],
    #         # cos_sin : Optional[tuple[torch.FloatTensor]],
    #         past_key_value: Optional[Cache] = None,
    #         cache_position: Optional[torch.LongTensor] = None,
    #         attention_mask: Optional[torch.FloatTensor] = None,
    #         head_mask: Optional[torch.FloatTensor] = None,
    #         encoder_hidden_states: Optional[torch.Tensor] = None,
    #         encoder_attention_mask: Optional[torch.FloatTensor] = None,
    #         output_attentions: Optional[bool] = False,
    #         **kwargs,
    #     ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
    #         is_cross_attention = encoder_hidden_states is not None
    #         if is_cross_attention:
    #             if not hasattr(self, "q_attn"):
    #                 raise ValueError(
    #                     "If class is used as cross attention, the weights `q_attn` have to be defined. "
    #                     "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
    #                 )

    #             query_states = self.q_attn(hidden_states)
    #             key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
    #             attention_mask = encoder_attention_mask
    #         else: #c_attn size [B, T, 3*d] so the split is on the second dim
    #             query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2) #split size == d
    #             # current q,k,v shape : [B, T, d]


    #         shape_q = (*query_states.shape[:-1], -1, self.head_dim) #(B, T, -1, d//num_heads) here -1 will be h (and num_heads==h)
    #         shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    #         query_states = query_states.view(shape_q).transpose(1, 2) #[B, h, T, d//h]
    #         key_states = key_states.view(shape_kv).transpose(1, 2)
    #         value_states = value_states.view(shape_kv).transpose(1, 2)

            
    #         T = query_states.shape[-2]

    #         #Compute the cos_sin if their are not already cached 
    #         if (self.rope_cos is None) or (self.rope_cos.size(0) < (T + (past_key_value[0].size(-2) if past_key_value else 0))):
    #             max_len = max(self._rope_meta["cache_max_len"], T + (past_key_value[0].size(-2) if past_key_value else 0) if past_key_value else T)
    #             cos, sin = build_rope_cache(
    #                 max_len=max_len,
    #                 rotary_dim=self._rope_meta["rotary_dim"],
    #                 base=self._rope_meta["theta"],
    #                 device=hidden_states.device,
    #                 dtype=hidden_states.dtype,
    #                 scale=self._rope_meta["scale"],
    #             )
    #             self.rope_cos = cos  # [max_len,1,rot_dim]
    #             self.rope_sin = sin

    #         #check for past_key_values and if so, truncate the cos_sin
    #         past_len = past_key_value[0].size(-2)       # cache length along seq axis
    #         # T_cur is current query length
    #         pos_slice = slice(past_len, past_len + T)
    #         cos = self.rope_cos[pos_slice]
    #         sin = self.rope_sin[pos_slice]

    #         #apply the rotary embeding to query_states and key_states
    #         query_states, key_states = apply_rope(query_states, key_states, cos, sin, self._rope_meta["rotary_dim"])

    #         if past_key_value is not None:
    #             if isinstance(past_key_value, EncoderDecoderCache):
    #                 if is_cross_attention:
    #                     past_key_value = past_key_value.cross_attention_cache
    #                 else:
    #                     past_key_value = past_key_value.self_attention_cache
    #             cache_kwargs = {"cache_position": cache_position}
    #             key_states, value_states = past_key_value.update(
    #                 key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
    #             )

    #         is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

    #         using_eager = self.config._attn_implementation == "eager"
    #         attention_interface: Callable = eager_attention_forward
    #         if self.config._attn_implementation != "eager":
    #             if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
    #                 using_eager = True
    #                 print.warning_once(
    #                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
    #                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
    #                 )
    #             else:
    #                 # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
    #                 # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
    #                 # not necessarily to eager (if mentioned options are provided).
    #                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    #         if using_eager and self.reorder_and_upcast_attn:
    #             attn_output, attn_weights = self._upcast_and_reordered_attn(
    #                 query_states, key_states, value_states, attention_mask, head_mask
    #             )
    #         else:
    #             attn_output, attn_weights = attention_interface(
    #                 self,
    #                 query_states,
    #                 key_states,
    #                 value_states,
    #                 attention_mask,
    #                 head_mask=head_mask,
    #                 dropout=self.attn_dropout.p if self.training else 0.0,
    #                 is_causal=is_causal,
    #                 **kwargs,
    #             )

    #         attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    #         attn_output = self.c_proj(attn_output)
    #         attn_output = self.resid_dropout(attn_output)

    #         return attn_output, attn_weights
        
    #     # bind the new forward
    #     attn_mod.forward = forward_with_rope.__get__(attn_mod, attn_mod.__class__)

    def _patch_attention(self,
                        attn_mod: nn.Module,
                        layer_idx: int,
                        theta: int,
                        rotary_dim: Optional[int],
                        scale: Optional[float],
                        cache_max_len: Optional[int],
                        n_head: int,
                        head_dim: int) -> None:

        orig_forward = attn_mod.forward  # keep if you ever want to fall back

        attn_mod.register_buffer("rope_cos", None, persistent=False)
        attn_mod.register_buffer("rope_sin", None, persistent=False)
        attn_mod._rope_meta = dict(
            theta=theta,
            rotary_dim=rotary_dim,
            scale=scale,
            cache_max_len=cache_max_len,
            n_head=n_head,
            head_dim=head_dim,
        )
        attn_mod.layer_idx = layer_idx  # needed for Cache.update

        def forward_with_rope(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            **kwargs,
        ):
            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention:
                if not hasattr(self, "q_attn"):
                    raise ValueError(
                        "If class is used as cross attention, the weights `q_attn` have to be defined. "
                        "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                    )
                query_states = self.q_attn(hidden_states)
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                attention_mask_ = encoder_attention_mask
            else:
                # self-attention
                query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
                attention_mask_ = attention_mask

            # [B,T,d] -> [B,H,T,Dh]
            B, T_full, D = query_states.size()
            query_states = query_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            key_states   = key_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            # [B,H,T,Dh]
            T = query_states.shape[-2]

            # RoPE (self-attention only)
            if (not is_cross_attention) and (self._rope_meta["rotary_dim"] is not None):
                # compute past_len safely
                if past_key_value is None:
                    past_len = 0
                elif isinstance(past_key_value, EncoderDecoderCache):
                    pkv = past_key_value.self_attention_cache
                    past_len = pkv.get_usable_length(self.layer_idx)
                elif isinstance(past_key_value, Cache):
                    past_len = past_key_value.get_usable_length(self.layer_idx)
                else:
                    # tuple (k, v)
                    past_len = past_key_value[0].size(-2)

                needed_len = past_len + T
                rd = self._rope_meta["rotary_dim"]

                if (self.rope_cos is None) or (self.rope_cos.size(0) < needed_len):
                    max_len = max(self._rope_meta["cache_max_len"], needed_len)
                    cos, sin = build_rope_cache(
                        max_len=max_len,
                        rotary_dim=rd,
                        base=self._rope_meta["theta"],
                        device=query_states.device,
                        dtype=query_states.dtype,
                        scale=self._rope_meta["scale"],
                    )
                    self.rope_cos = cos   # [L,1,rd]
                    self.rope_sin = sin

                pos_slice = slice(past_len, past_len + T)
                cos = self.rope_cos[pos_slice].unsqueeze(0).unsqueeze(0)  # [1,1,T,rd]
                sin = self.rope_sin[pos_slice].unsqueeze(0).unsqueeze(0)

                query_states, key_states = apply_rope(
                    query_states, key_states, cos, sin, rd
                )

            # KV cache update
            if past_key_value is not None:
                if isinstance(past_key_value, EncoderDecoderCache):
                    if is_cross_attention:
                        pkv = past_key_value.cross_attention_cache
                    else:
                        pkv = past_key_value.self_attention_cache
                else:
                    pkv = past_key_value

                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = pkv.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
                )

            # attention core (HF helpers)
            is_causal = attention_mask_ is None and query_states.shape[-2] > 1 and not is_cross_attention

            using_eager = self.config._attn_implementation == "eager"
            attention_interface: Callable = eager_attention_forward

            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                    using_eager = True
                    print.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                        'Falling back to eager attention. Use `attn_implementation="eager"` to silence this.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            if using_eager and self.reorder_and_upcast_attn:
                attn_output, attn_weights = self._upcast_and_reordered_attn(
                    query_states, key_states, value_states, attention_mask_, head_mask
                )
            else:
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask_,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )

            # [B,H,T,Dh] -> [B,T,d]
            attn_output = attn_output.reshape(B, T_full, -1).contiguous()
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights

        attn_mod.forward = forward_with_rope.__get__(attn_mod, attn_mod.__class__)


class GPT2RopeAdaptaterWithWatermarkLabels(GPT2RopeAdapter):
    """
    This adaptater, will do the same as GPT2RopeAdaptater (adding Rope to gpt2) but will also overwright
    GPT2LMHeadModel.forward() to pass the wm label through to the attention level
    """
    def supports(self, hfmodel) -> bool:
        return getattr(hfmodel.config, "model_type", "") == "gpt2"
    
    @staticmethod
    def set_rope_wm_context(hf_model, wm_applied, key_vec):
        """
        Store watermark metadata on the HF model object.

        hf_model: the actual HuggingFace model instance (e.g. GPT2LMHeadModel)
        wm_applied: Bool/Int tensor [B] (1 = watermark on for this sample)
        key_vec: Long/Int tensor [K] (displacement per segment)
        """
        # Make sure they are tensors
        if not torch.is_tensor(wm_applied):
            wm_applied = torch.tensor(wm_applied)
        if not torch.is_tensor(key_vec):
            key_vec = torch.tensor(key_vec)

        # Put them on the same device as the model
        # device = next(hf_model.parameters()).device
        # hf_model._rope_wm_applied = wm_applied.to(device)
        # hf_model._rope_wm_key_vec = key_vec.to(device)

        for block in hf_model.transformer.h:
            attn = block.attn
            device = next(attn.parameters()).device
            attn._rope_wm_enabled = True
            attn._rope_wm_applied = wm_applied
            attn._rope_wm_key_vec = key_vec

        # for debugging
        hf_model._rope_wm_enabled = True
    
    @staticmethod
    def clear_rope_wm_context(hf_model):
        for block in hf_model.transformer.h:
            attn = block.attn
            if hasattr(attn, "_rope_wm_enabled"):
                attn._rope_wm_enabled = False
            if hasattr(attn, "_rope_wm_applied"):
                attn._rope_wm_applied = None
            if hasattr(attn, "_rope_wm_key_vec"):
                attn._rope_wm_key_vec = None

    def add_rope_and_label(self, hfmodel, *,
                           theta=10000,
                           rotary_dim = None,
                           scale = None,
                           cache_max_len = 4096):
        transformer = hfmodel.transformer
        cfg = hfmodel.config
        n_head = cfg.n_head
        head_dim = cfg.n_embd // n_head
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        rotary_dim = min(rotary_dim, head_dim)

        super()._bypass_wpe(transformer)

        for layer_idx, block in enumerate(transformer.h):
            # block.attn._rope_wm_parent = hfmodel
            attn = block.attn
            attn._rope_wm_enabled = False
            attn._rope_wm_applied = None
            attn._rope_wm_key_vec = None
            self._new_patch_attention(attn, layer_idx, theta, rotary_dim, scale, cache_max_len, n_head, head_dim)

            

        setattr(cfg, "use_rope_watermark", True)
        setattr(cfg, "rope_theta", float(theta))
        setattr(cfg, "rope_dim", int(rotary_dim))
        if scale is not None:
            setattr(cfg, "rope_scale", float(scale))
    
    def _new_patch_attention(self,
                             attn_mod: nn.Module,
                             layer_idx: int,
                             theta: int,
                             rotary_dim: Optional[int],
                             scale: Optional[float],
                             cache_max_len: Optional[int],
                             n_head: int,
                             head_dim: int) -> None:

        orig_forward = attn_mod.forward  # keep if you ever want to fall back

        attn_mod.register_buffer("rope_cos", None, persistent=False)
        attn_mod.register_buffer("rope_sin", None, persistent=False)
        attn_mod._rope_meta = dict(
            theta=theta,
            rotary_dim=rotary_dim,
            scale=scale,
            cache_max_len=cache_max_len,
            n_head=n_head,
            head_dim=head_dim,
        )
        attn_mod.layer_idx = layer_idx  # needed for Cache.update

        def forward_with_rope(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            **kwargs,
        ):
            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention:
                if not hasattr(self, "q_attn"):
                    raise ValueError(
                        "If class is used as cross attention, the weights `q_attn` have to be defined. "
                        "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                    )
                query_states = self.q_attn(hidden_states)
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                attention_mask_ = encoder_attention_mask
            else:
                # self-attention
                query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
                attention_mask_ = attention_mask

            # [B,T,d] -> [B,H,T,Dh]
            B, T_full, D = query_states.size()
            query_states = query_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            key_states   = key_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            # [B,H,T,Dh]
            T = query_states.shape[-2]
            # Only for self-attention + RoPE
            if (not is_cross_attention) and (self._rope_meta["rotary_dim"] is not None):

                # 1) compute past_len safely
                if past_key_value is None:
                    past_len = 0
                elif isinstance(past_key_value, EncoderDecoderCache):
                    pkv = past_key_value.self_attention_cache
                    past_len = pkv.get_usable_length(self.layer_idx)
                elif isinstance(past_key_value, Cache):
                    past_len = past_key_value.get_usable_length(self.layer_idx)
                else:
                    past_len = past_key_value[0].size(-2)

                # 2) base positions
                if cache_position is not None:
                    pos_base = cache_position.to(query_states.device)
                    if pos_base.dim() == 1:
                        pos_base = pos_base.unsqueeze(0).expand(B, T)   # [B,T]
                    elif pos_base.dim() == 2:
                        # assume already [B,T]
                        assert pos_base.shape[0] == B and pos_base.shape[1] == T
                    else:
                        raise ValueError(f"Unexpected cache_position shape: {pos_base.shape}")
                else:
                    pos_base = torch.arange(past_len, past_len + T, device=query_states.device).unsqueeze(0).expand(B, T)


                # 3) watermark
                # parent = getattr(self, "_rope_wm_parent", None)
                # wm_enabled = bool(getattr(parent, "_rope_wm_enabled", False)) if parent is not None else False
                wm_enabled = bool(getattr(self, "_rope_wm_enabled", False))

                if wm_enabled:
                    wm_applied = self._rope_wm_applied  # [B]
                    key_vec    = self._rope_wm_key_vec  # [K]
                    pos_prime  = build_position_ids_prime_cumulative_from_posbase(pos_base, wm_applied, key_vec)
                else:
                    pos_prime = pos_base

                # print(max(pos_prime-pos_base))
                # 4) cache extend by max_pos
                max_pos = int(pos_prime.max().item()) + 1
                rd = self._rope_meta["rotary_dim"]

                if (self.rope_cos is None) or (self.rope_cos.size(0) < max_pos):
                    max_len = max(self._rope_meta["cache_max_len"], max_pos)
                    cos_cache, sin_cache = build_rope_cache(
                        max_len=max_len,
                        rotary_dim=rd,
                        base=self._rope_meta["theta"],
                        device=query_states.device,
                        dtype=query_states.dtype,
                        scale=self._rope_meta["scale"],
                    )
                    self.rope_cos = cos_cache  # [L,rd]
                    self.rope_sin = sin_cache

                # 5) gather
                cos = self.rope_cos[pos_prime].unsqueeze(1)  # [B,1,T,rd]
                sin = self.rope_sin[pos_prime].unsqueeze(1)

                query_states, key_states = apply_rope(query_states, key_states, cos, sin, rd)

                # if getattr(self, "_rope_probe_enabled", False) and (self._rope_probe_store is not None):
                #     q_rot = query_states[..., :rd]   # [B,H,T,rd]
                #     k_rot = key_states[..., :rd]

                    # # mean pool over tokens -> [B,H,rd]
                    # self._rope_probe_store["q_pool"] = q_rot.mean(dim=-2).detach().float().cpu()
                    # self._rope_probe_store["k_pool"] = k_rot.mean(dim=-2).detach().float().cpu()

            # KV cache update
            if past_key_value is not None:
                if isinstance(past_key_value, EncoderDecoderCache):
                    if is_cross_attention:
                        pkv = past_key_value.cross_attention_cache
                    else:
                        pkv = past_key_value.self_attention_cache
                else:
                    pkv = past_key_value

                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = pkv.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
                )
            
            if getattr(self, "_rope_probe_enabled", False) and (self._rope_probe_store is not None):
                which = getattr(self, "_rope_probe_which", set())

                q = query_states  # [B,H,Tq,Dh]
                k = key_states    # [B,H,Tk,Dh]

                q_rot = q[..., :rd]
                k_rot = k[..., :rd]
                v = value_states  # [B,H,Tk,Dh]

                # ---- logits in fp32 for stability
                logits = torch.matmul(q_rot.float(), k_rot.float().transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B,H,Tq,Tk]

                # ---- additive attention mask (HF usually provides 0 for keep, -inf/-1e4 for mask)
                if attention_mask_ is not None:
                    logits = logits + attention_mask_.float()

                # ---- causal mask (safe, works for Tk>=Tq)
                B_, H_, Tq, Tk = logits.shape

                # Build causal so that query i can attend to keys <= (Tk - Tq + i)
                # If Tk==Tq => standard lower-triangular
                # If Tk>Tq (cache) => allows attending to the full prefix + up to current position
                offset = Tk - Tq
                i = torch.arange(Tq, device=logits.device).unsqueeze(1)          # [Tq,1]
                j = torch.arange(Tk, device=logits.device).unsqueeze(0)          # [1,Tk]
                causal = (j <= (i + offset))                                     # [Tq,Tk], bool

                # IMPORTANT: use large negative instead of -inf to avoid NaN softmax
                logits = logits.masked_fill(~causal.view(1, 1, Tq, Tk), -1e4)

                # ---- attn + ctx
                attn = torch.softmax(logits, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)   # safety
                ctx  = torch.matmul(attn, v.float())                              # [B,H,Tq,Dh]

                # ---- store summaries
                if "qk" in which: #diag
                    self._rope_probe_store["q_pool"] = q_rot.mean(dim=-2).detach().float().cpu()  # [B,H,Dh] (or slice rd outside)
                    self._rope_probe_store["k_pool"] = k_rot.mean(dim=-2).detach().float().cpu()
                    
                if "qk_logits_train" in which: #This is to train the sep regularization term
                    # store small sampled token slices only to limit memory
                    # expects self._rope_probe_store already contains indices
                    idx_q = self._rope_probe_store["idx_q"]   # LongTensor [Iq] on device
                    idx_k = self._rope_probe_store["idx_k"]   # LongTensor [Ik] on device

                    # optionally restrict to rotary dims only
                    rd = self._rope_meta["rotary_dim"]
                    q_sel = query_states[..., :rd]  # [B,H,T,rd]
                    k_sel = key_states[..., :rd]    # [B,H,T,rd]

                    # gather tokens
                    q_tok = q_sel.index_select(dim=-2, index=idx_q.to(q_sel.device))  # [B,H,Iq,rd]
                    k_tok = k_sel.index_select(dim=-2, index=idx_k.to(k_sel.device))  # [B,H,Ik,rd]

                    # keep on GPU for backward (do NOT detach / cpu)
                    self._rope_probe_store["q_tok"] = q_tok
                    self._rope_probe_store["k_tok"] = k_tok

                if "logits" in which: #diag
                    # finite-safe mean(abs(logits)) over (Tq,Tk)
                    finite = torch.isfinite(logits)
                    abs_sum = (logits.abs() * finite).sum(dim=(-1, -2))            # [B,H]
                    count   = finite.sum(dim=(-1, -2)).clamp(min=1)                # [B,H]
                    meanabs = abs_sum / count

                    # std over finite values is harder; this is a simple robust approx:
                    # compute std treating non-finite as 0 but renormalize by count
                    m = meanabs.unsqueeze(-1).unsqueeze(-1)                         # [B,H,1,1]
                    var_sum = (((logits - m) * finite) ** 2).sum(dim=(-1, -2))
                    std = torch.sqrt(var_sum / count.clamp(min=1))

                    self._rope_probe_store["logits_meanabs"] = meanabs.detach().cpu()
                    self._rope_probe_store["logits_std"]     = std.detach().cpu()

                if "attn" in which: #diag
                    ent = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(dim=-1)  # [B,H,Tq]
                    self._rope_probe_store["attn_entropy"] = ent.mean(dim=-1).detach().cpu()  # [B,H]
                    self._rope_probe_store["attn_max"]     = attn.max(dim=-1).values.mean(dim=-1).detach().cpu()

                if "ctx" in which: #diag
                    self._rope_probe_store["ctx_pool"] = ctx.mean(dim=-2).detach().cpu()     # [B,H,Dh]
                    self._rope_probe_store["ctx_norm"] = ctx.norm(dim=-1).mean(dim=-1).detach().cpu()  # [B,H]


            # attention core (HF helpers)
            is_causal = attention_mask_ is None and query_states.shape[-2] > 1 and not is_cross_attention

            using_eager = self.config._attn_implementation == "eager"
            attention_interface: Callable = eager_attention_forward

            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                    using_eager = True
                    print.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                        'Falling back to eager attention. Use `attn_implementation="eager"` to silence this.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            if using_eager and self.reorder_and_upcast_attn:
                attn_output, attn_weights = self._upcast_and_reordered_attn(
                    query_states, key_states, value_states, attention_mask_, head_mask
                )
            else:
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask_,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )

            # [B,H,T,Dh] -> [B,T,d]
            attn_output = attn_output.reshape(B, T_full, -1).contiguous()
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights

        attn_mod.forward = forward_with_rope.__get__(attn_mod, attn_mod.__class__)