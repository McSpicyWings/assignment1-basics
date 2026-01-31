import torch
from torch import Tensor
from typing import Optional
from cs336_basics.layer import TransformerLM
from cs336_basics.layer import softmax

def _sample_top_p(prob: Tensor, top_p: float) -> Tensor:
    '''
    prob: (B, V) 概率分布
    返回 (B,) 采样得到的 token id
    '''
    if not (0.0 < top_p <= 1.0):
        raise ValueError
    # 降序排列的prob 和 对应的原始序号（即token id）
    sorted_porbs, sorted_idx = torch.sort(prob, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_porbs, dim=-1)
    mask = cum_probs <= top_p
    # 至少保留一个
    mask[..., 0] = True
    filtered_probs = sorted_porbs * mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # sampled: (B,1)
    sampled = torch.multinomial(filtered_probs, num_samples= 1)
    #           (B,1) -> (B,)
    next_token = sorted_idx.gather(dim=-1, index=sampled).squeeze(-1)
    return next_token

def decode(
    model:TransformerLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    '''
    自回归解码。

    Args:
        model: TransformerLM 调用 model(x) -> logits，shape (B, T, V) 或 (..., T, V)
        prompt_ids: LongTensor, shape (T,) 或 (B, T)
        max_new_tokens: 最多生成多少个新 token
        context_length: 模型最大上下文长度（每步喂入时做截断）
        eos_token_id: <|endoftext|> token id；生成到它就停止（对每个样本独立停止）
        temperature: 温度 t, t->0 更贪心；t=1 原分布；t>1 更随机
        top_p: nucleus sampling 阈值 p = 1 表示不截断, 做全量采样

    Returns:
        LongTensor: 生成后的 token 序列，shape (B, T+new) 或 (T+new,)
                   注意：若 batch 中不同样本提前遇到 eos，会在后面保持 eos（padding-like）。
    '''
    
    # 这里等于0贪心地取最大？
    if temperature <= 0 :
        raise ValueError
    
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.long()
    # 统一成 batch 维度
    is_batched = (prompt_ids.dim() == 2)
    if not is_batched:
        prompt_ids = prompt_ids.unsqueeze(0)  # (1,T)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    B, T = prompt_ids.shape
    
    # if prompt_ids.dim() == 1:
    #     prompt_ids = prompt_ids.unsqueeze(0) # (1, T)
        
    # device = next(model.parameters()).device
    # prompt_ids = prompt_ids.to(device)
    # B, T = prompt_ids.size(-2), prompt_ids.size(-1)
    
    out = prompt_ids
    
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    
    for _ in range(max_new_tokens):
        # Take last context_length tokens (slice sequence dimension, not batch)
        x = out[:, -context_length:]
        logits = model(x)
        logits_last = logits[:, -1, :]
        
        if temperature != 1.0:
            logits_last = logits_last / temperature
        
        # softmax over vocabulary dimension
        probs = softmax(logits_last, dim=-1)
        
        if top_p < 1.0:
            next_token = _sample_top_p(probs, top_p)
        else:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # For finished sequences, continue outputting eos
        next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
        
        out = torch.cat([out, next_token.unsqueeze(-1)], dim=-1)
        
        finished = finished | (next_token == eos_token_id)
        if torch.all(finished):
            break
    
    return out if is_batched else out.squeeze(0)