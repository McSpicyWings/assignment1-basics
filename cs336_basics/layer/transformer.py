import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import math

from .atten import CasualMultiHeadAttenRope, softmax
from .FFN import FFNlayer
from .RMSNorm import RMSNorm
from .embeding import Embedding
from .linear import Linear


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len: int,
        theta: float,
        device:torch.device = None,
        dtype = torch.float32
        # weights: dict[str, Tensor],
        # 改为通过 load_state_dict
    ):
        # d_ff应该等于d_model
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CasualMultiHeadAttenRope(
            d_model, num_heads, d_model, max_seq_len, theta
        )
        self.ffn = FFNlayer(d_model, d_ff, device=device)
        """
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            """

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


        
        

class BaseLM(torch.nn.Module):
    """
    Transformer Language Model that can be trained from scratch.
    This version initializes all weights randomly and properly registers
    all submodules for gradient computation.
    
    The state_dict keys are compatible with the weight format expected by TransformerLM,
    so you can use load_state_dict() to load pretrained weights.
    """
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
        
        # Embeddings and output layers
        # Use attribute names that match the expected state_dict keys
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Use ModuleList with name "layers" to match state_dict format
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
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
        """
        Forward pass through the language model.
        
        Args:
            in_indices: Input token indices of shape (batch_size, sequence_length)
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding and lm_head parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params
    
    @classmethod
    def from_config(cls, config: dict, device=None, dtype=None):
        """Create a model from a configuration dictionary."""
        return cls(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config.get("rope_theta", 10000.0),
            device=device,
            dtype=dtype,
        )
    
    def get_config(self) -> dict:
        """Return the model configuration as a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
        }