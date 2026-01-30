import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int

from .atten import CasualMultiHeadAttenRope, softmax
from .FFN import FFNlayer
from .RMSNorm import RMSNorm
from .embeding import Embedding
from .linear import Linear

class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
    ):
        # d_ff应该等于d_model
        super().__init__()
        self.normlayer1 = RMSNorm(d_model)
        self.normlayer2 = RMSNorm(d_model)
        # dff 即d_in和d_out
        self.attnlayer = CasualMultiHeadAttenRope(
            d_model, num_heads, d_model, max_seq_len, theta
        )
        self.ffnlayer = FFNlayer(d_model, d_ff)

        self.normlayer1.load_state_dict({"weight": weights["ln1.weight"]})
        self.normlayer2.load_state_dict({"weight": weights["ln2.weight"]})
        self.attnlayer.load_state_dict(
            {
                "q_proj_weight": weights["attn.q_proj.weight"],
                "k_proj_weight": weights["attn.k_proj.weight"],
                "v_proj_weight": weights["attn.v_proj.weight"],
                "output_proj_weight": weights["attn.output_proj.weight"],
            }
        )
        self.ffnlayer.load_state_dict(
            {
                "W1": weights["ffn.w1.weight"],
                "W2": weights["ffn.w2.weight"],
                "W3": weights["ffn.w3.weight"],
            }
        )

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
        x = x + self.attnlayer(self.normlayer1(x))
        x = x + self.ffnlayer(self.normlayer2(x))
        return x

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor]
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.norm_final = RMSNorm(d_model)
        
        self.token_embeddings.load_state_dict({"weight":weights["token_embeddings.weight"]})
        self.lm_head.load_state_dict({"W":weights["lm_head.weight"]})
        self.norm_final.load_state_dict({"weight":weights["ln_final.weight"]})
        
        self.block_layers = []
        for layer_i in range(num_layers):
            lay_weights = { 
                "attn.q_proj.weight": weights[f"layers.{layer_i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{layer_i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{layer_i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{layer_i}.attn.output_proj.weight"],
                "ffn.w1.weight": weights[f"layers.{layer_i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{layer_i}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{layer_i}.ffn.w3.weight"], 
                "ln1.weight": weights[f"layers.{layer_i}.ln1.weight"],
                "ln2.weight": weights[f"layers.{layer_i}.ln2.weight"],
            }
            self.block_layers.append(TransformerLayer(d_model,num_heads,d_ff,context_length,rope_theta,lay_weights))
        
        '''
        Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
            '''
    def forward(self,in_indices: Int[Tensor, " batch_size sequence_length"]):
        x = self.token_embeddings(in_indices)
        for layer in self.block_layers:
            x = layer(x)
        x = self.norm_final(x)
        x = self.lm_head(x)
        return x