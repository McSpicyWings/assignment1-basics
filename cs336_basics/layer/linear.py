import torch
from torch import nn

from torch import Tensor
from jaxtyping import Float,Bool,Int

import einops
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.

        Args:
            in_features (int): _description_
            out_features (int): _description_
            device (torch.device | None): _description_. Defaults to None.
            dtype (torch.dty | None): _description_. Defaults to None.
        """
        super().__init__()
        # self.d_in = in_features
        # self.d_out = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=self.dtype, device=self.device))
        
        std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean = 0, std = std, a=-3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
