import torch
import einops

from torch import Tensor,inf
from jaxtyping import Float,Bool,Int
from cs336_basics.layer.RotaryEmbeding import RotaryPositionalEmbedding
from .linear import Linear

from math import sqrt

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim,keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp,dim = dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None
) ->Float[Tensor, " ...  queries d_v"]:
    d_k, d_v = K.size(-1),  V.size(-1)
    if mask is not None:
        # seq_i表示第几个token的Q/K/V值 对一个Q 乘别的所有K 再取softmax
        score = softmax(einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys").masked_fill( ~mask, float("-inf")) / sqrt(d_k), dim=-1)
    else:
        score = softmax(einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k), dim=-1)
    #  keys values一定相等
    return einops.einsum(score, V, "... s1 s2 , ... s2 d_v -> ... s1 d_v")


class CasualMultiHeadAtten(torch.nn.Module):
    def __init__(self, d_model:Int, num_head:int, d_in:int):
        super().__init__()
        
        assert d_model % num_head == 0
        
        self.num_head = num_head
        self.d_k = d_model // num_head
        # self.q_proj = torch.nn.Parameter(torch.empty(d_model, d_in))
        # self.k_proj = torch.nn.Parameter(torch.empty(d_model, d_in))
        # self.v_proj = torch.nn.Parameter(torch.empty(d_model, d_in))
        # self.output_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.q_proj = Linear(d_in, d_model)
        self.k_proj = Linear(d_in, d_model)
        self.v_proj = Linear(d_in, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        
    def forward(self,x:Float[Tensor, " ... sequence_length d_in"])->Float[Tensor, " ... sequence_length d_out"]:
        # Wq,Wk,Wv: (d_model, d_in)
        Wqkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)  # (3*d_model, d_in)
        qkv  = einops.einsum(Wqkv, x, "qkv_out d_in, ... seq d_in -> ... seq qkv_out")          # (..., seq, 3*d_model)

        Q, K, V = qkv.chunk(3, dim=-1)  # 每个都是 (..., seq, d_model)
        Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        V = einops.rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0) #包括
        attn_out = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq, d_head)
        attn_out = einops.rearrange(attn_out,"... head seq d -> ... seq (head d)")
        # return einops.einsum(self.output_proj_weight, attn_out , "d_out d_model, ... seq_len d_model  -> ... seq_len d_out")
        return self.output_proj(attn_out)
    

class CasualMultiHeadAttenRope(CasualMultiHeadAtten):
    def __init__(self, d_model, num_head, d_in, max_seq_len: int = None, theta: float = None):
        super().__init__(d_model, num_head, d_in)
        d_head = d_model//num_head
        if max_seq_len and theta:
            self.RoPE = RotaryPositionalEmbedding(theta,d_head,max_seq_len)
    
    def forward(self, x, token_positions = None):
        Wqkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)  # (3*d_model, d_in)
        qkv  = einops.einsum(Wqkv, x, "qkv_out d_in, ... seq d_in -> ... seq qkv_out")          # (..., seq, 3*d_model)

        Q, K, V = qkv.chunk(3, dim=-1)  # 每个都是 (..., seq, d_model)
        Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        Q = self.RoPE(Q,token_positions)
        V = einops.rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        K = self.RoPE(K,token_positions)
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0) #包括
        attn_out = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq, d_head)
        attn_out = einops.rearrange(attn_out,"... head seq d -> ... seq (head d)")
        # return einops.einsum(self.output_proj.weight, attn_out , "d_out d_model, ... seq_len d_model  -> ... seq_len d_out")
        return self.output_proj(attn_out)

# # 二合一版本
# class CasualMultiHeadAttenRope(torch.nn.Module):
#     def __init__(self, d_model, num_head, d_in, max_seq_len: int = None, theta: float = None):
#         super().__init__()
#         d_head = d_model//num_head
#         if max_seq_len and theta:
#             self.RoPE = RotaryPositionalEmbedding(theta,d_head,max_seq_len)
        
#         assert d_model % num_head == 0
#         self.num_head = num_head
#         self.d_k = d_model // num_head
#         self.q_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_in))
#         self.k_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_in))
#         self.v_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_in))
#         self.output_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
    
#     def forward(self, x, token_positions = None):
#         Wqkv = torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight], dim=0)  # (3*d_model, d_in)
#         qkv  = einops.einsum(Wqkv, x, "qkv_out d_in, ... seq d_in -> ... seq qkv_out")          # (..., seq, 3*d_model)

#         Q, K, V = qkv.chunk(3, dim=-1)  # 每个都是 (..., seq, d_model)
#         Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
#         K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
#         V = einops.rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
#         if token_positions is not None:
#             Q = self.RoPE(Q,token_positions)
#             K = self.RoPE(K,token_positions)
#         seq_len = x.size(-2)
#         mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0) #包括
#         attn_out = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq, d_head)
#         attn_out = einops.rearrange(attn_out,"... head seq d -> ... seq (head d)")
#         return einops.einsum(self.output_proj_weight, attn_out , "d_out d_model, ... seq_len d_model  -> ... seq_len d_out")