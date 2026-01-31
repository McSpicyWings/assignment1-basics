from collections.abc import Callable,Iterable
from typing import Optional,Tuple
import torch
import math


# 纯粹的手搓
class AdamWOptim(torch.optim.Optimizer):
    def __init__(self, params, lr:float = 0.1, betas:Tuple[float,float] = (0.9, 0.95), weight_decay:float = 1, eps:float = 1e-8):
        beta1,beta2 = betas
        # defaults = {"lr":lr,
        #             "beta1":beta1,
        #             "beta2":beta2,
        #             "weight_decay":weight_decay,
        #             "eps":eps
        #             }
        # 还能这样
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps)
        
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["t"] = 0
                m1 = state["m1"]
                m2 = state["m2"]
                state["t"] += 1
                t = state["t"]
                
                with torch.no_grad():
                    m1.mul_(beta1).add_(g, alpha = 1-beta1)
                    m2.mul_(beta2).addcmul_(g, g, value = 1-beta2)
                    # torch.Tensor.addcmul_()
                    # moment1 = beta1 * moment1 + (1 - beta1) * g
                    # moment2 = beta2 * moment2 + (1 - beta2) * g.pow(2)
                    
                    alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                    # parameter update
                    p.addcdiv_(m1, m2.sqrt().add_(eps), value=-alpha_t)
                    # 这里报错： RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
                    if weight_decay:
                        p.mul_(1 - lr * weight_decay)
        return loss

@torch.no_grad()
def lr_cosine_schedule(t, lr_max, lr_min, warmup_iter, cosine_cycle_iters):
    if t < 0:
        raise ValueError(f"t should be positive = {t}")
    if t < warmup_iter:
        return lr_max * t / warmup_iter
    elif warmup_iter <= t <= cosine_cycle_iters:
        return lr_min + 0.5 * (1.0 + math.cos(math.pi * (t - warmup_iter) / (cosine_cycle_iters - warmup_iter)) ) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm:float, eps:float = 1e-6):
    total_sq = 0.0
    params = []
    for p in parameters:
        if p.grad is None:
            continue
        params.append(p)
        total_sq += p.grad.detach().pow(2).sum()
    
    global_norm = torch.sqrt(total_sq)
    if global_norm > max_l2_norm:
        scale = max_l2_norm / (global_norm + eps)
        with torch.no_grad():
            for p in params:
                p.grad.mul_(scale)
    


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)
    
    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def main():
    for i in range(1,4):
        print(f"lr:1e{i}")
        weights = torch.nn.Parameter(5 * torch.randn((10,10)))
        opt = SGD([weights], lr=float(10**i))
        
        for t in range(10):
            opt.zero_grad()
            loss = (weights ** 2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()

if __name__ == "__main__":
    main()
