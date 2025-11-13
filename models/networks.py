import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModel
from typing import Union, List, Optional, Callable
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import math

"""
This file is where all the backbone networks and losses will be defined. 
Each backbone will be a Class, and helper function will be available.
"""

def freeze_model(model : AutoModel,
                 num_freezed_layers : Union[int, str] = 'none',
                 specific_layer_name : str = None,
                 freeze_embeddings : bool = False,
                 freeze_all : bool = False,
                 freeze_all_expect_layer_names : Union[List, str]=None,
                 ) -> None:
    """
    This function is a helper function to freeze a number of specifyed layers in the model.

    Args :
        - model (AutoModel) : The HF model you want to freeze
        - num_freezed_layers : (Union[int, str]) : the number of freezd layers you want. Starting from the begening. eg. 16 will freeze layers up to the 16th
        - specifi_layer_name : (str) : Will freeze a specefic layer in the model 
        - freeze_embeddings : (bool) : will freeze the embeddings along with the head (tied weights)
        - freeze_all : (bool) : will freeze the whole model
    """
    if isinstance(num_freezed_layers, str):
        assert num_freezed_layers == 'all' \
            or num_freezed_layers == 'none',\
            TypeError("The only str accepted are 'all' or 'non'. If you need to specify a number of layers, please enter an int")

        if num_freezed_layers == 'none':
            return 0
    
    if specific_layer_name :
        _freeze_by_name(model, specific_layer_name)
        return 0
    elif freeze_embeddings:
        _freeze_embedings(model)
        return 0
    elif freeze_all:
        _freeze_all(model)
        return 0
    elif freeze_all_expect_layer_names:
        _freeze_all_exept_name(model, freeze_all_expect_layer_names)
        return 0
        
    else:    
        for attr in ["model.layers", "transformer.h", "bert.encoder.layer", # test the diffrent attributes for the layers 
                    "encoder.block", "decocder.block"]:
            try :
                layers = model.get_submodule(attr)
                break
            except AttributeError :
                continue
        
        else : raise ValueError("Unsuported architecture; add its stack path")

        if num_freezed_layers != None:
            if num_freezed_layers == 'all': #freez all the layers if all is specified
                num_freezed_layers = len(layers)
            else:
                num_freezed_layers = min(num_freezed_layers, len(layers))

            for i in range(num_freezed_layers):
                for p in layers[i].parameters():
                    p.requires_grad = False
            print(f'{num_freezed_layers} where freezed')
            return 0
        print(f"⚠️ \033[93m[WARNING]\033[0m\tNo layers where freezed. To freeze layers, specify :"
              f"\n\t--num_freezed_layers or\n\t--freeze_specific_layer_name or\n\t--freeze_embedding or\n\t--frezze_all or\n\t--frezze_all_exept_layer_name\n")
        return 0

def _freeze_embedings(model : AutoModel) -> None:
    """
    This frezze the model embeding and the head at the same time. The embedding weights are tied to the lm_head
    """
    for p in model.get_input_embeddings().parameters:
        p.requires_grad = False
    return 0

def _freeze_by_name(model : AutoModel, specific_name : str) -> None:
    """
    To freeze a specefic layer by name
    """
    if specific_name not in [name for name, _ in model.named_modules()]:
        raise ValueError(f"the name {specific_name} is not in the current architectre. See the models architecture: \n{model}")
    
    for name, module in model.named_modules():
        if specific_name != name :
            continue

        else:
            for p in module.parameters():
                p.requires_grad = False
    
    return 0

def _freeze_all(model : AutoModel) -> None:
    for p in model.parameters():
        p.requires_grad = False

def _freeze_all_exept_name(model : AutoModel, layer_names : Union[List, str]) -> None:
    """
    To freeze all layers, exept a specific one (or list)
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    modules_dict = dict(model.named_modules()) #Transform the named_modules into a python dict (k=name, v=module) 
    _freeze_all(model) #Freeze all the layers 

    for layer_name in layer_names:
        if layer_name not in modules_dict.keys() or layer_name == "": # check and see if the named layer are present in the model's modules
            raise ValueError(f"the name {layer_name} is not in the current architectre. See the models architecture: \n{model}")
        
        for p in modules_dict[layer_name].parameters(): #the layer_name is known to be part of the model, otherwhise it would had raised an error 
            p.requires_grad = True

def get_optimizer(optimizer_name : str = 'adamw'):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam
    
    if optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW


#------------------------Passthrough Method------------------------

class PassThroughLayer(nn.Module):
    """
    Here is defined the passthrough layer
    """
    def __init__(self, hidden_dim, LLM_hidden_dim):
        super().__init__()

        self.linear = nn.Linear(LLM_hidden_dim, hidden_dim, bias=True)
        # W1 = torch.zeros((hidden_dim, LLM_hidden_dim))
        # W1[:LLM_hidden_dim, :LLM_hidden_dim] = torch.eye(LLM_hidden_dim)
        # self.linear.weight.data = W1
        # self.linear.bias = False

        self.linear2 = nn.Linear(hidden_dim, LLM_hidden_dim, bias=True)
        # W2 = torch.zeros((LLM_hidden_dim, hidden_dim))
        # W2[:LLM_hidden_dim, :LLM_hidden_dim] = torch.eye(LLM_hidden_dim)
        # self.linear2.weight.data = W2
        # self.linear2.bias = False

        # could use an mlp with d_model and hidden_dim and residual 
        #could try without residual here

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = F.gelu(self.linear(hidden_states))
        # hidden_states = self.linear(hidden_states)
        return self.linear2(hidden_states) + residual

class PtlWithGpt2Block(nn.Module):
    """
    Wrapper module for the passthrough layer and the GPT2 block. This allows to pass the hidden_State to the ptl and pass the other arguments to the GPT2 block
    """
    def __init__(self, ptl : nn.Module, block : nn.Module):
        super().__init__()

        self.ptl = ptl
        self.block = block

    def forward(self, hidden_states, *args, **kwargs):
        block_device = next(self.block.parameters()).device

        if hidden_states.device != block_device:
            hidden_states = hidden_states.to(block_device, non_blocking = True)

        hidden_states = self.ptl(hidden_states) #forward the hidden state through the ptl
        # the forward the rest to the 
        return self.block(hidden_states, *args, **kwargs)


#------------------------RoPE Method------------------------

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
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)  # [T,1,D]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)  # [T,1,D]
    return cos, sin

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
            self._add_cos_sin_to_forward_gptblock(block)
            self._patch_attention(block.attn, layer_idx, theta, rotary_dim, scale, cache_max_len, n_head, head_dim)

        # 3) record in config so you can reconstruct at test time
        setattr(cfg, "use_rope_watermark", True)
        setattr(cfg, "rope_theta", float(theta))
        setattr(cfg, "rope_dim", int(rotary_dim))
        if scale is not None:
            setattr(cfg, "rope_scale", float(scale))

    # ---------- internals ----------
    def _add_cos_sin_to_forward_gptblock(self, gpt_block : nn.Module):
        ori_forward = gpt_block.__class__.forward

        
        def forward_with_cos_sin(self, **kwargs):
            cos_sin = build_rope_cache(
                        max_len=max_len,
                        rotary_dim=self._rope_meta["rotary_dim"],
                        base=self._rope_meta["theta"],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                        scale=self._rope_meta["scale"],
                    )
            return ori_forward(self, cos_sin=cos_sin, **kwargs)
        
        gpt_block.forward = forward_with_cos_sin.__get__(gpt_block, gpt_block.__class__)


    # def new_gpt_block_forward(self,
    #                           hidden_states: Optional[tuple[torch.FloatTensor]],
    #                           past_key_value: Optional[Cache] = None,
    #                           cache_position: Optional[torch.LongTensor] = None,
    #                           attention_mask: Optional[torch.FloatTensor] = None,
    #                           head_mask: Optional[torch.FloatTensor] = None,
    #                           encoder_hidden_states: Optional[torch.Tensor] = None,
    #                           encoder_attention_mask: Optional[torch.FloatTensor] = None,
    #                           use_cache: Optional[bool] = False,
    #                           output_attentions: Optional[bool] = False,
    #                           **kwargs,
    # )-> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
    #     cos_sin = build_rope_cache(
    #                 max_len=max_len,
    #                 rotary_dim=self._rope_meta["rotary_dim"],
    #                 base=self._rope_meta["theta"],
    #                 device=hidden_states.device,
    #                 dtype=hidden_states.dtype,
    #                 scale=self._rope_meta["scale"],
    #             )
    #     outputs = GPT2Block.forward(
    #         hidden_states,
    #         past_key_value,
    #         cache_position,
    #         attention_mask,
    #         head_mask,
    #         encoder_hidden_states,
    #         encoder_attention_mask,
    #         use_cache,
    #         output_attentions,
    #         cos_sin=cos_sin,
    #         **kwargs,)

    def _bypass_wpe(self, transformer):
        """Monkey-patch GPT2Model.forward to ignore wpe(position_ids)."""
        orig_forward = transformer.__class__.forward

        def forward_no_wpe(self, input_ids=None, attention_mask=None, **kwargs):
            # Lightly adapted from HF GPT2Model.forward:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # [B,T]

            inputs_embeds = self.wte(input_ids)              # [B,T,d]
            hidden_states = inputs_embeds                    # <-- no + wpe

            # Then proceed exactly as in original forward:
            # build causal/attention masks, iterate blocks, etc.
            # We call the original forward but inject hidden_states via kwargs:
            return orig_forward(self,
                                input_ids=None,  # force it to use inputs_embeds
                                attention_mask=attention_mask,
                                inputs_embeds=hidden_states,
                                **kwargs)

        # Bind the method
        transformer.forward = forward_no_wpe.__get__(transformer, transformer.__class__)

    def _patch_attention(self, attn_mod, layer_idx, theta, rotary_dim, scale, cache_max_len, n_head, head_dim):
        """
        Replace GPT2Attention.forward with a version that:
        - computes Q,K,V as usual,
        - applies RoPE to Q and K on the first 'rotary_dim' dims,
        - handles past_key_value (KV cache) correctly.
        """
        orig_forward = attn_mod.forward

        # buffers per layer (lazy init on first forward)
        attn_mod.register_buffer("rope_cos", None, persistent=False)
        attn_mod.register_buffer("rope_sin", None, persistent=False)
        attn_mod._rope_meta = dict(theta=theta, rotary_dim=rotary_dim, scale=scale,
                                   cache_max_len=cache_max_len, n_head=n_head, head_dim=head_dim)

        def forward_with_rope(self, hidden_states, layer_past=None, **kwargs):
            # Compute Q,K,V like GPT2Attention normally does
            # (with self.c_attn: a single proj producing concat qkv)
            # Then reshape to [B, n_head, T, head_dim]
            output = orig_forward.__wrapped__(self, hidden_states, layer_past=layer_past, **kwargs) \
                     if hasattr(orig_forward, "__wrapped__") else None
            # ^ we can't call orig_forward directly because it will do non-rotary attn.
            # So we basically reimplement GPT2Attention forward below.
            #
            # If you prefer: copy HF's GPT2Attention.forward, then insert RoPE after q/k projections.
            # For brevity here, I outline the key steps:

            # --- 1) qkv projection ---
            qkv = self.c_attn(hidden_states)                      # [B,T,3*d]
            query, key, value = qkv.split(self.split_size, dim=2) # [B,T,d] each

            # --- 2) shape to heads ---
            def shape(x):
                B, T, D = x.size()
                x = x.view(B, T, self.num_heads, D // self.num_heads)
                return x.permute(0, 2, 1, 3).contiguous()  # [B,H,T,d_h]
            q = shape(query)
            k = shape(key)
            v = shape(value)

            B, H, T, Dh = q.shape
            assert Dh == head_dim

            # --- 3) build / extend cos,sin cache ---
            if (self.rope_cos is None) or (self.rope_cos.size(0) < (T + (layer_past[0].size(-2) if layer_past else 0))):
                max_len = max(self._rope_meta["cache_max_len"], T + (layer_past[0].size(-2) if layer_past else 0) if layer_past else T)
                cos, sin = build_rope_cache(
                    max_len=max_len,
                    rotary_dim=self._rope_meta["rotary_dim"],
                    base=self._rope_meta["theta"],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                    scale=self._rope_meta["scale"],
                )
                self.rope_cos = cos  # [max_len,1,rot_dim]
                self.rope_sin = sin

            past_len = layer_past[0].size(-2) if layer_past is not None else 0
            pos_slice = slice(past_len, past_len + T)
            cos = self.rope_cos[pos_slice]  # [T,1,rot_dim]
            sin = self.rope_sin[pos_slice]  # [T,1,rot_dim]

            # --- 4) apply RoPE on Q,K (first rotary_dim dims) ---
            q, k = apply_rope(q, k, cos, sin, self._rope_meta["rotary_dim"])

            # --- 5) concat with past, compute attn, project out ---
            if layer_past is not None:
                pk, pv = layer_past
                k = torch.cat([pk, k], dim=-2)  # sequence dim
                v = torch.cat([pv, v], dim=-2)

            present = (k, v)

            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # [B,H,T,S]
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                attn_scores = attn_scores + kwargs["attention_mask"]

            attn_probs = nn.functional.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)

            context = torch.matmul(attn_probs, v)                  # [B,H,T,Dh]
            # merge heads
            context = context.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

            out = self.c_proj(context)
            out = self.resid_dropout(out)

            if self.output_attentions:
                return out, present, attn_probs
            else:
                return out, present

        # bind the new forward
        attn_mod.forward = forward_with_rope.__get__(attn_mod, attn_mod.__class__)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        cos_sin : Optional[tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else: #c_attn size [B, T, 3*d] so the split is on the second dim
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2) #split size == d
            # current q,k,v shape : [B, T, d]


        shape_q = (*query_states.shape[:-1], -1, self.head_dim) #(B, T, -1, d//num_heads) here -1 will be h (and num_heads==h)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2) #[B, h, T, d//h]
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        #apply the rotary embeding to query_states and key_states
        cos, sin = cos_sin
        query_states, key_states = apply_rope(query_states, key_states, cos, sin, self._rope_meta["rotary_dim"])


        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                if is_cross_attention:
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentioned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights