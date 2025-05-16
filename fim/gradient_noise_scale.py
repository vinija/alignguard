# ============================
# Module: alignguard/fim/gradient_noise_scale.py
# ============================
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector



@torch.no_grad()
def estimate_gns(model, dataloader, loss_fn, device="cuda", batches=100):
model.eval()
grads = []
for i, (x, y) in enumerate(dataloader):
if i >= batches:
break
x, y = x.to(device), y.to(device)
model.zero_grad()
output = model(x)
loss = loss_fn(output, y)
loss.backward()
grad = parameters_to_vector([p.grad for p in model.parameters() if p.grad is not None]).detach()
grads.append(grad.unsqueeze(0))



grads = torch.cat(grads, dim=0) # [B, D]
grad_mean = grads.mean(dim=0)
variance = ((grads - grad_mean.unsqueeze(0)) ** 2).mean(dim=0)
norm_mean = grad_mean.norm().item()
norm_std = variance.sqrt().norm().item()
gns = (norm_std / (norm_mean + 1e-8)) ** 2
return gns, norm_mean, norm_std
