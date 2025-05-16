# ============================
# Module: alignguard/fim/curvature_variance.py
# ============================
import torch
from torch.nn.utils import parameters_to_vector
from typing import List



@torch.no_grad()
def compute_curvature_variance(model, dataloader, loss_fn, device='cuda', max_batches=100):
model.eval()
grads_list: List[torch.Tensor] = []
for i, batch in enumerate(dataloader):
if i >= max_batches:
break
inputs, targets = batch
inputs, targets = inputs.to(device), targets.to(device)



model.zero_grad()
output = model(inputs)
loss = loss_fn(output, targets)
loss.backward()



grads = parameters_to_vector([p.grad for p in model.parameters() if p.grad is not None]).detach()
grads_list.append(grads.unsqueeze(0))



grads_stack = torch.cat(grads_list, dim=0) # shape: [B, D]
variance = torch.var(grads_stack, dim=0)
curvature_variance = variance.mean().item()
return curvature_variance, variance
