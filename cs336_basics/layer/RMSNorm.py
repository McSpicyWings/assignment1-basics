import torch

from torch import Tensor
from jaxtyping import Float,Bool,Int


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        
        Args:
            d_model:
        '''
        super().__init__()
        # 初始化为1
        self.weight = torch.nn.Parameter(torch.ones(d_model,device=device, dtype=dtype))
        self.eps = eps
        
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # inplace运算， 可能会导致无法计算grad
        # # keepdim: [bs,seq,dim] -> [bs,seq,1]
        # var = x.pow(2).mean(dim=-1, keepdim=True)
        # x.mul_(torch.rsqrt(var + self.eps))
        # # 默认对齐到最后维度, 作broadcast
        # return x.mul_(self.weight).to(in_dtype)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)
    
    