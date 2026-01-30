import torch
import einops

from torch import Tensor
from jaxtyping import Float,Bool,Int


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = x[..., ::2],x[..., 1::2]
    # 最后一维对齐， 其他维自动广播?
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 恢复原有顺序
    # 会有中间变量？
    # return torch.stack((y1, y2), dim=-1).flatten(start_dim=-2).to(x.dtype)
    out = torch.empty_like(x, dtype=(x1 * cos).dtype)  # 或直接 float32 再 cast 回去
    out[..., ::2] = y1
    out[..., 1::2] = y2
    return out.to(x.dtype)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        '''
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta**(torch.arange(0, d_k, 2, device = device,  dtype=torch.float) / d_k))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # i: pos  j:embeding_dim
        freqs = einops.einsum(t, inv_freq," i, j-> i j ")
        cos = freqs.cos()
        sin = freqs.sin()
        # 交叉排列
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor = None)-> torch.Tensor:
        if token_positions is not None:
            cos, sin = self.cos_cache[token_positions], self.sin_cache[token_positions]
        else:
            seq_len = x.size(-2)
            cos, sin = self.cos_cache[:seq_len], self.sin_cache[:seq_len]
        return apply_rotary_emb(x,cos,sin)
