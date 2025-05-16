# ============================
# Module: alignguard/fim/fisher_vector.py
# ============================
import torch
from torch.nn.utils import parameters_to_vector



class FisherVectorComputer:
def __init__(self, model, loss_fn, device):
self.model = model
self.loss_fn = loss_fn
self.device = device



def compute_fisher_matrix(self, dataloader, max_batches=100):
model = self.model
model.eval()
fisher_matrix = None
for i, (x, y) in enumerate(dataloader):
if i >= max_batches:
break
x, y = x.to(self.device), y.to(self.device)
model.zero_grad()
outputs = model(x)
loss = self.loss_fn(outputs, y)
loss.backward()
grad_vector = parameters_to_vector([p.grad for p in model.parameters() if p.grad is not None])
grad_outer = grad_vector.unsqueeze(1) @ grad_vector.unsqueeze(0)
if fisher_matrix is None:
fisher_matrix = grad_outer.detach()
else:
fisher_matrix += grad_outer.detach()
fisher_matrix /= max_batches
return fisher_matrix


