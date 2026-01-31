#!/usr/bin/env python3
# train/ablations/test_models.py
"""
验证消融实验模型是否能正常运行
用法: python -m train.ablations.test_models
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from train.ablations.models import create_model, MODEL_REGISTRY, print_param_count


def test_model(model_type: str, device: torch.device):
    """测试单个模型的前向传播和梯度计算"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}")
    print(f"{'='*60}")
    
    # 创建模型
    model = create_model(
        model_type=model_type,
        vocab_size=1000,  # 小模型用于测试
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=336,  # 遵循 8/3 规则
    )
    model = model.to(device)
    
    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # 前向传播
    batch_size = 4
    seq_len = 64
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    logits = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 1000), "Output shape mismatch!"
    
    # 损失计算
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    print(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Gradients computed: {has_grad}")
    assert has_grad, "Some gradients are None!"
    
    # 检查是否有 NaN 梯度
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"  NaN gradients: {has_nan}")
    assert not has_nan, "Gradients contain NaN!"
    
    print(f"  ✓ {model_type} passed all tests!")
    return True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 打印参数量对比
    print_param_count()
    
    # 测试所有模型
    model_types = ["baseline", "postnorm", "nope", "silu"]
    results = {}
    
    for model_type in model_types:
        try:
            results[model_type] = test_model(model_type, device)
        except Exception as e:
            print(f"  ✗ {model_type} FAILED: {e}")
            results[model_type] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for model_type, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {model_type}")
    
    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
