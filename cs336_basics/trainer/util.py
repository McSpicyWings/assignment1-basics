import torch

from torch import Tensor
from jaxtyping import Float, Int

import numpy as np
import numpy.typing as npt


def cross_entorpy(
    logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    m = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - m
    logsumexp = m.squeeze(-1) + torch.log(
        torch.exp(shifted_logits).sum(dim=-1, keepdim=False)
    )

    # gather函数 out和index形状相同 index和input的dimension数量相同 并且index.size(i) <= input.size(i)
    # dim=0时 out[i][j][k] = input[index[i][j][k]][j][k]
    y = torch.gather(input=logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # 下面写法也能过也可以
    # y = logits[torch.arange(logits.size(0)), targets]
    loss = logsumexp - y
    return torch.mean(loss)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = dataset.shape[0]
    if n <   context_length + 1:
        raise ValueError(
            f"Dataset too short: len(dataset)={n}, but need at least context_length + 1={context_length+1}"
        )
    starts = np.random.randint(0, n - context_length, size = batch_size)
    offsets = np.arange(context_length)
    idx = starts[:,None] + offsets[None, :]
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y
    
    # np数组存在内存
    # max_idx = len(dataset) - context_length
    
    # starts = torch.randint(0, max_idx, (batch_size,))
    # x = torch.stack([torch.from_numpy( (dataset[i : i + context_length]).astype(np.int64)) for i in starts]).to(device,dtype=torch.long)
    # y = torch.stack([torch.from_numpy( (dataset[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in starts]).to(device,dtype=torch.long)
    # return x,y