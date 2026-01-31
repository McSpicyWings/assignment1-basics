import einops
import torch

from torch import Tensor
from jaxtyping import Float,Bool,Int
from .linear import Linear

def silu(x:Float[Tensor, "... self.W1.size(dim=-1)"]):
    return x * torch.sigmoid(x)

class FFNlayer(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int|None, device=None):
        super().__init__()
        # 保证是64的倍数
        # 向上取整未通过测试
        # d_ff = ((d_model * 8 // 3) + 63) // 64 * 64
        # 向下取整有为0风险?
        # d_ff =( ((d_model * 8 // 3) >> 6) << 6)
        # d_ff = ((d_model * 8 // 3) & ~63)
        if not d_ff:
            d_ff = ((d_model * 8 // 3) + 63)// 64 * 64
        # self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device))
        # self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device))
        # self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device))
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    # def __GLU(self,x:Float[Tensor, "... self.W1.size(dim=-1)"]):
    #     pass
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # SwiGlu
        # y = silu((einops.einsum(self.w1, x, "dff d_model, ... d_model -> ... dff"))) * (einops.einsum(self.w3, x, "dff d_model, ... d_model -> ... dff"))
        # return einops.einsum(self.w2, y,"d_model dff, ... dff -> ... d_model")
        return self.w2((silu(self.w1(x)) * self.w3(x)))