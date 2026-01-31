# train/ablations/models.py
"""
消融实验模型变体
- Baseline: Pre-norm + RoPE + SwiGLU (使用原始 TransformerLM)
- PostNorm: Post-norm + RoPE + SwiGLU
- NoPE: Pre-norm + No Position Encoding + SwiGLU  
- SiLU: Pre-norm + RoPE + SiLU (无门控，d_ff=4*d_model 匹配参数量)
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import einops

from cs336_basics.layer.atten import CasualMultiHeadAttenRope, CasualMultiHeadAtten, scaled_dot_product_attention
from cs336_basics.layer.FFN import FFNlayer, silu
from cs336_basics.layer.RMSNorm import RMSNorm
from cs336_basics.layer.embeding import Embedding
from cs336_basics.layer.linear import Linear


# =============================================================================
# Ablation 1: Post-norm Transformer Block
# =============================================================================

class PostNormTransformerBlock(nn.Module):
    """
    Post-norm Transformer Block:
        z = RMSNorm(x + Attn(x))
        y = RMSNorm(z + FFN(z))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CasualMultiHeadAttenRope(
            d_model, num_heads, d_model, max_seq_len, theta
        )
        self.ffn = FFNlayer(d_model, d_ff, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # Post-norm: norm after residual
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class PostNormTransformerLM(nn.Module):
    """Post-norm + RoPE + SwiGLU"""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # Post-norm: 最后一层不需要额外的 ln_final
        # 因为每个 block 输出已经做过 norm
        
        self.layers = nn.ModuleList([
            PostNormTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(x)
        return x
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "model_type": "postnorm",
        }


# =============================================================================
# Ablation 2: NoPE (No Position Encoding)
# =============================================================================

class CasualMultiHeadAttenNoPE(CasualMultiHeadAtten):
    """Causal Multi-Head Attention without any position encoding"""
    def __init__(self, d_model: int, num_head: int, d_in: int, max_seq_len: int = None, theta: float = None):
        # 调用父类但不使用 RoPE
        super().__init__(d_model, num_head, d_in)
        # 不创建 RoPE，保持接口兼容
    
    def forward(self, x: Tensor, token_positions=None) -> Tensor:
        # 和 CasualMultiHeadAtten.forward 一样，不做任何位置编码
        Wqkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        qkv = einops.einsum(Wqkv, x, "qkv_out d_in, ... seq d_in -> ... seq qkv_out")
        
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = einops.rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        K = einops.rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        V = einops.rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_head)
        # 不应用 RoPE!
        
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = einops.rearrange(attn_out, "... head seq d -> ... seq (head d)")
        return self.output_proj(attn_out)


class NoPETransformerBlock(nn.Module):
    """Transformer Block without position encoding (Pre-norm + NoPE + SwiGLU)"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CasualMultiHeadAttenNoPE(d_model, num_heads, d_model, max_seq_len, theta)
        self.ffn = FFNlayer(d_model, d_ff, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NoPETransformerLM(nn.Module):
    """Pre-norm + NoPE + SwiGLU"""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            NoPETransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "model_type": "nope",
        }


# =============================================================================
# Ablation 3: SiLU FFN (no gating, d_ff = 4 * d_model for param matching)
# =============================================================================

class SiLUFFN(nn.Module):
    """
    Standard FFN with SiLU activation (no gating):
        FFN(x) = W2 * SiLU(W1 * x)
    
    使用 d_ff = 4 * d_model 来匹配 SwiGLU 的参数量
    """
    def __init__(self, d_model: int, d_ff: int, device=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)))


class SiLUTransformerBlock(nn.Module):
    """Transformer Block with SiLU FFN (Pre-norm + RoPE + SiLU)"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CasualMultiHeadAttenRope(d_model, num_heads, d_model, max_seq_len, theta)
        self.ffn = SiLUFFN(d_model, d_ff, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SiLUTransformerLM(nn.Module):
    """Pre-norm + RoPE + SiLU FFN (no gating)"""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,  # 应该设为 4 * d_model 以匹配参数量
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            SiLUTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "model_type": "silu",
        }


# =============================================================================
# Model Factory
# =============================================================================

MODEL_REGISTRY = {
    "baseline": None,  # 使用原始 TransformerLM
    "postnorm": PostNormTransformerLM,
    "nope": NoPETransformerLM,
    "silu": SiLUTransformerLM,
}


def create_model(
    model_type: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float = 10000.0,
    device=None,
    dtype=None,
):
    """
    创建指定类型的模型
    
    Args:
        model_type: "baseline", "postnorm", "nope", "silu"
    """
    if model_type == "baseline":
        from cs336_basics.layer.transformer import TransformerLM
        return TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
    elif model_type == "silu":
        # SiLU 使用 d_ff = 4 * d_model 匹配参数量
        return SiLUTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=4 * d_model,  # 匹配参数量
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
    else:
        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_cls(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )


def print_param_count():
    """打印各模型参数量对比"""
    d_model = 512
    num_layers = 4
    vocab_size = 10000
    context_length = 256
    num_heads = 16
    d_ff = 1344  # baseline 的 d_ff
    
    print("=" * 60)
    print("Model Parameter Comparison")
    print("=" * 60)
    
    for model_type in ["baseline", "postnorm", "nope", "silu"]:
        model = create_model(
            model_type=model_type,
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        total = sum(p.numel() for p in model.parameters())
        non_emb = model.get_num_params(non_embedding=True)
        print(f"{model_type:12s}: Total={total:,}  Non-embedding={non_emb:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_param_count()
