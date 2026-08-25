




















from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging
from configuration_qwen3_carmoe import Qwen3CARMoeConfig
from dataclasses import dataclass

logger = logging.get_logger(__name__)


@dataclass
class CARMoeModelOutputWithPast(MoeModelOutputWithPast):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        complexity_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_complexity_scores=True` is passed or when `config.output_complexity_scores=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size * sequence_length, )`.
            Complexity scores of the model at the output of each layer.
        num_experts_per_tok (`tuple(torch.FloatTensor)`, *optional*, returned when `output_num_experts_per_tok=True` is passed or when `config.output_num_experts_per_tok=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size * sequence_length, )`.
            Number of experts per token at the output of each layer.
    """
    complexity_scores: Optional[tuple[torch.FloatTensor]] = None
    num_experts_per_tok: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class CARMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        complexity_loss (`torch.FloatTensor`, *optional*, returned when `output_complexity_loss=True` is passed or when `config.output_complexity_loss=True`):
            Complexity loss of the model at the output of each layer.
        complexity_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_complexity_scores=True` is passed or when `config.output_complexity_scores=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size * sequence_length, )`.
            Complexity scores of the model at the output of each layer.
        num_experts_per_tok (`tuple(torch.FloatTensor)`, *optional*, returned when `output_num_experts_per_tok=True` is passed or when `config.output_num_experts_per_tok=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size * sequence_length, )`.
            Number of experts per token at the output of each layer.
    """
    complexity_loss: Optional[torch.FloatTensor] = None
    complexity_scores: Optional[tuple[torch.FloatTensor]] = None
    num_experts_per_tok: Optional[tuple[torch.FloatTensor]] = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3CARMoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3CARMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3CARMoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  
        self.k_norm = Qwen3CARMoeRMSNorm(self.head_dim,
                                         eps=config.rms_norm_eps)  
        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3CARMoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3CARMoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob

        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3CARMoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        
        self.use_gru_cp = getattr(config, "use_gru_cp", False)
        self.dual_signal = getattr(config, "dual_signal", False)

        cp_in_dim = config.gru_hidden_size if self.use_gru_cp else config.hidden_size
        output_dim = 2 if self.dual_signal else 1

        if self.use_gru_cp:
            self.cp_projector = nn.Linear(config.hidden_size, config.gru_hidden_size, bias=False)

        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(cp_in_dim, config.complexity_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(config.complexity_hidden_size, output_dim, bias=False),
            nn.Sigmoid()
        )
        self.kmax = config.kmax
        self.kmin = getattr(config, "kmin", 7)
        
        self.fmap_p = config.fmap_p

    def fmap(self, c_eff):
        


        k_t = torch.round(
            self.kmin + (c_eff ** self.fmap_p) * (self.kmax - self.kmin)
        ).clamp(min=self.kmin, max=self.num_experts)
        k_t = k_t.to(torch.int64)  
        return k_t

    def forward(self, hidden_states: torch.Tensor, cp_gru_state=None, shared_gru=None) -> tuple:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        
        if self.use_gru_cp:
            proj = self.cp_projector(hidden_states_flat)  
            proj = proj.unsqueeze(1)  
            if shared_gru is not None:
                gru_out, new_cp_gru_state = shared_gru(proj, cp_gru_state)
                cp_input = gru_out.squeeze(1)  
            else:
                cp_input = proj.squeeze(1)
                new_cp_gru_state = cp_gru_state
        else:
            cp_input = hidden_states_flat
            new_cp_gru_state = None

        
        cp_out = self.complexity_predictor(cp_input)  

        
        if self.dual_signal:
            c_s = cp_out[:, 0]
            c_h = cp_out[:, 1]
            c_eff = 1.0 - (1.0 - c_s) * (1.0 - c_h)
            complexity_scores = cp_out  
        else:
            c_eff = cp_out.squeeze(-1)
            complexity_scores = c_eff

        
        num_experts_per_tok = self.fmap(c_eff)  
        
        router_logits = self.gate(hidden_states_flat)

        routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        sorted_probs, sorted_indices = torch.sort(routing_probs, dim=-1, descending=True)  
        
        max_k = self.kmax
        
        range_k = torch.arange(max_k, device=hidden_states_flat.device).unsqueeze(0)  
        
        num_experts_per_tok_expanded = num_experts_per_tok.unsqueeze(-1)  
        
        topk_mask = range_k < num_experts_per_tok_expanded  

        
        sorted_probs_topk = sorted_probs[:, :max_k]  
        sorted_indices_topk = sorted_indices[:, :max_k]  

        
        masked_probs = sorted_probs_topk * topk_mask.float()  

        if self.norm_topk_prob:  
            
            sums = masked_probs.sum(dim=-1, keepdim=True) + 1e-9
            masked_probs = masked_probs / sums
        

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_flat.dtype, device=hidden_states_flat.device
        )

        masked_sorted_indices_topk = torch.where(topk_mask, sorted_indices_topk, -1)  
        masked_probs = torch.where(topk_mask, masked_probs, 0.0)  

        
        token_idx, slot_idx = torch.where(masked_sorted_indices_topk != -1)  
        expert_ids = masked_sorted_indices_topk[token_idx, slot_idx]  
        routing_weights = masked_probs[token_idx, slot_idx].unsqueeze(-1)  

        selected_states = hidden_states_flat[token_idx]  
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        
        for expert_id in range(self.num_experts):
            mask = expert_ids == expert_id
            if not mask.any():
                continue
            expert_token_idx = token_idx[mask]
            expert_states = selected_states[mask]
            expert_routing_weights = routing_weights[mask]

            expert_layer = self.experts[expert_id]
            expert_out = expert_layer(expert_states)
            expert_out = expert_out * expert_routing_weights
            
            final_hidden_states.index_add_(0, expert_token_idx, expert_out)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        if self.use_gru_cp:
            return final_hidden_states, router_logits, complexity_scores, num_experts_per_tok, new_cp_gru_state
        else:
            return final_hidden_states, router_logits, complexity_scores, num_experts_per_tok


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3CARMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        


        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3CARMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3CARMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3CARMoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3CARMoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3CARMoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3CARMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3CARMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            output_complexity_scores: Optional[bool] = False,
            output_num_experts_per_tok: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  
            cp_gru_state: Optional[torch.Tensor] = None,   
            shared_gru: Optional[nn.Module] = None,        
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        







        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        
        if isinstance(self.mlp, Qwen3CARMoeSparseMoeBlock):
            mlp_outputs = self.mlp(hidden_states, cp_gru_state=cp_gru_state, shared_gru=shared_gru)
            if getattr(self.config, "use_gru_cp", False):
                hidden_states, router_logits, complexity_scores, num_experts_per_tok, cp_gru_state = mlp_outputs
            else:
                hidden_states, router_logits, complexity_scores, num_experts_per_tok = mlp_outputs
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None
            complexity_scores = None
            num_experts_per_tok = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        if output_complexity_scores:
            outputs += (complexity_scores,)

        if output_num_experts_per_tok:
            outputs += (num_experts_per_tok,)
            
        if getattr(self.config, "use_gru_cp", False):
            outputs += (cp_gru_state,)  

        return outputs


class Qwen3CARMoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3CARMoeConfig, device=None):
        super().__init__()
        
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3CARMoePreTrainedModel(PreTrainedModel):
    config_class = Qwen3CARMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3CARMoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3CARMoeRMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class Qwen3CARMoeModel(Qwen3CARMoePreTrainedModel):
    def __init__(self, config: Qwen3CARMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3CARMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3CARMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3CARMoeRotaryEmbedding(config=config)
        
        
        if getattr(config, "use_gru_cp", False):
            self.cp_gru = nn.GRU(config.gru_hidden_size, config.gru_hidden_size, batch_first=True)
            
        self.gradient_checkpointing = False

        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            output_complexity_scores: Optional[bool] = None,
            output_num_experts_per_tok: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> CARMoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_complexity_scores = (
            output_complexity_scores if output_complexity_scores is not None else self.config.output_complexity_scores
        )
        output_num_experts_per_tok = (
            output_num_experts_per_tok if output_num_experts_per_tok is not None else self.config.output_num_experts_per_tok
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_complexity_scores = () if output_complexity_scores else None
        all_num_experts_per_tok = () if output_num_experts_per_tok else None

        cp_gru_state = None
        shared_gru = getattr(self, "cp_gru", None)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                output_complexity_scores=output_complexity_scores,
                output_num_experts_per_tok=output_num_experts_per_tok,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cp_gru_state=cp_gru_state,
                shared_gru=shared_gru,
                **flash_attn_kwargs,
            )

            
            hidden_states = layer_outputs[0]
            
            
            output_idx = 1

            if output_attentions:
                all_self_attns += (layer_outputs[output_idx],)
                output_idx += 1

            if output_router_logits:
                all_router_logits += (layer_outputs[output_idx],)
                output_idx += 1

            if output_complexity_scores:
                all_complexity_scores += (layer_outputs[output_idx],)
                output_idx += 1

            if output_num_experts_per_tok:
                all_num_experts_per_tok += (layer_outputs[output_idx],)
                output_idx += 1
                
            if getattr(self.config, "use_gru_cp", False):
                cp_gru_state = layer_outputs[output_idx]
                output_idx += 1

        hidden_states = self.norm(hidden_states)

        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return CARMoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            complexity_scores=all_complexity_scores,
            num_experts_per_tok=all_num_experts_per_tok,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
        num_experts: Optional[int] = None,
        num_experts_per_tok: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_k: Optional[int] = None,
) -> Union[torch.Tensor, int]:
    """
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    total_tokens = concatenated_gate_logits.shape[0]

    
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    
    sorted_probs, sorted_indices = torch.sort(
        routing_weights, dim=-1, descending=True
    )  

    
    topk_indices = sorted_indices[:, :max_k]
    topk_probs = sorted_probs[:, :max_k]

    
    if isinstance(num_experts_per_tok, tuple) and len(num_experts_per_tok) > 0:
        
        num_experts_per_tok = [tensor.to(compute_device) for tensor in num_experts_per_tok]

    
    concatenated_num_experts_per_tok = torch.cat([t.to(compute_device) for t in num_experts_per_tok], dim=0).view(-1)
    concatenated_num_experts_per_tok = concatenated_num_experts_per_tok.unsqueeze(1).expand(-1, max_k)

    
    range_k = torch.arange(max_k, device=compute_device).unsqueeze(0)  

    
    topk_mask = range_k < concatenated_num_experts_per_tok  

    
    one_hot = torch.nn.functional.one_hot(topk_indices, num_classes=num_experts)  

    
    mask_expanded = topk_mask.unsqueeze(-1)  
    masked_one_hot = one_hot * mask_expanded  

    
    expert_mask = masked_one_hot.sum(dim=1).float()

    
    tokens_per_expert = expert_mask.mean(dim=0)  

    
    router_prob_per_expert = routing_weights.mean(dim=0)  

    
    load_balancing_loss = (tokens_per_expert * router_prob_per_expert).sum() * num_experts

    return load_balancing_loss


def complexity_loss_func(
    complexity_scores: Union[torch.Tensor, tuple[torch.Tensor], None],
    ori_complexity_scores: Union[torch.Tensor, None],
    attention_mask: Optional[torch.Tensor] = None, 
    entropy_alpha: float = 1.0,
    dual_signal: bool = False
) -> Union[torch.Tensor, int]:
    """
    计算复杂度预测的 MSE Loss，自动忽略 Padding Token。
    支持双信号 (DS-RCP) 计算
    """
    
    if (complexity_scores is None or not isinstance(complexity_scores, tuple)) or \
       (ori_complexity_scores is None or not isinstance(ori_complexity_scores, torch.Tensor)):
        return 0.0
    
    
    num_layers = len(complexity_scores)
    compute_device = complexity_scores[0].device
    
    
    
    concat_pred = torch.cat([score.to(compute_device) for score in complexity_scores], dim=0)
    
    
    if dual_signal:
        
        ori_flat = ori_complexity_scores.to(compute_device).view(-1, 2)
        concat_ori = ori_flat.repeat(num_layers, 1) 
    else:
        ori_flat = ori_complexity_scores.to(compute_device).view(-1)  
        concat_ori = ori_flat.repeat(num_layers)  

    
    if attention_mask is not None:
        mask_flat = attention_mask.to(compute_device).view(-1)  
        concat_mask = mask_flat.repeat(num_layers)  
        valid_indices = concat_mask > 0.5  
        
        final_pred = concat_pred[valid_indices]
        final_ori = concat_ori[valid_indices]
    else:
        final_pred = concat_pred
        final_ori = concat_ori

    
    if final_pred.numel() == 0:
        return torch.tensor(0.0, device=compute_device, requires_grad=True)

    if dual_signal:
        
        loss_s = F.mse_loss(final_pred[:, 0], final_ori[:, 0])
        loss_h = F.mse_loss(final_pred[:, 1], final_ori[:, 1])
        loss = loss_s + entropy_alpha * loss_h
    else:
        loss = F.mse_loss(final_pred, final_ori) 
        
    return loss


@auto_docstring
class Qwen3CARMoeForCausalLM(Qwen3CARMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3CARMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.complexity_lambda = config.complexity_lambda
        self.num_experts = config.num_experts
        self.kmax = config.kmax

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            output_complexity_scores: Optional[bool] = None,
            output_num_experts_per_tok: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            complexity_scores: Optional[torch.Tensor] = None,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> CARMoeCausalLMOutputWithPast:
        






        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_complexity_scores = (
            output_complexity_scores if output_complexity_scores is not None else self.config.output_complexity_scores
        )

        output_num_experts_per_tok = (
            output_num_experts_per_tok if output_num_experts_per_tok is not None else self.config.output_num_experts_per_tok
        )

        
        outputs: CARMoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_complexity_scores=output_complexity_scores,
            output_num_experts_per_tok=output_num_experts_per_tok,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                outputs.num_experts_per_tok,
                attention_mask,
                max_k=self.kmax,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  

        complexity_loss = None
        if output_complexity_scores and complexity_scores is not None:
            complexity_loss = complexity_loss_func(
                outputs.complexity_scores,  
                complexity_scores,          
                attention_mask=attention_mask, 
                entropy_alpha=getattr(self.config, "entropy_alpha", 1.0),
                dual_signal=getattr(self.config, "dual_signal", False)
            )
            
            if labels is not None:
                
                loss += self.complexity_lambda * complexity_loss.to(loss.device)

        return CARMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            complexity_loss=complexity_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            complexity_scores=outputs.complexity_scores,
            num_experts_per_tok=outputs.num_experts_per_tok,
        )


@auto_docstring(
    custom_intro="""
    The Qwen3CARMoe Model transformer with a sequence classification head on top (linear layer).

    [`Qwen3CARMoeForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class Qwen3CARMoeForSequenceClassification(Qwen3CARMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3CARMoeModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        






        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class Qwen3CARMoeForTokenClassification(Qwen3CARMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3CARMoeModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        






        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class Qwen3CARMoeForQuestionAnswering(Qwen3CARMoePreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen3CARMoeModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3CARMoeForCausalLM",
    "Qwen3CARMoeForQuestionAnswering",
    "Qwen3CARMoeModel",
    "Qwen3CARMoePreTrainedModel",
    "Qwen3CARMoeForSequenceClassification",
    "Qwen3CARMoeForTokenClassification",
]
