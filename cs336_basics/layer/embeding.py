import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float,Bool,Int

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.

        Args:
            num_embeddings (int): size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device, optional): _description_. Defaults to None.
            dtype (torch.dtype, optional): _description_. Defaults to None.
        """
        super().__init__()
        # self.device = device
        # self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.d_model = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
        
        
        
    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids
        '''
        token_ids = token_ids.long()
        return self.weight[token_ids]