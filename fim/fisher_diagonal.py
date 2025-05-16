# ============================
# Module: alignguard/fim/fisher_diagonal.py
# ============================
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import logging
from typing import Optional, Dict, Union



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class FisherDiagonalEstimator:
def __init__(self, model: nn.Module, device: torch.device, ema_decay: float = 0.95, max_steps: int = 100):
self.model = model
self.device = device
self.ema_decay = ema_decay
self.max_steps = max_steps
self.reset()



def reset(self):
self.fisher_diag = None
self.steps = 0



def update(self, batch, loss_fn, accumulate: bool = True):
inputs, labels = batch
inputs, labels = inputs.to(self.device), labels.to(self.device)



self.model.zero_grad()
outputs = self.model(inputs)
loss = loss_fn(outputs, labels)
loss.backward()



grads = []
for p in self.model.parameters():
if p.grad is not None:
grads.append(p.grad.detach().clone().flatten())
grad_vector = torch.cat(grads)



current_fisher = grad_vector ** 2



if self.fisher_diag is None:
self.fisher_diag = current_fisher
elif accumulate:
self.fisher_diag = self.ema_decay * self.fisher_diag + (1 - self.ema_decay) * current_fisher
else:
self.fisher_diag += current_fisher



self.steps += 1
if self.steps >= self.max_steps:
logger.info("Max Fisher estimation steps reached.")



def finalize(self):
if self.fisher_diag is None:
raise ValueError("No Fisher data accumulated yet.")
if self.steps > 0:
self.fisher_diag /= self.steps
return self.fisher_diag



def save(self, path: str):
torch.save(self.fisher_diag, path)
logger.info(f"Fisher diag saved to {path}")



def load(self, path: str):
self.fisher_diag = torch.load(path)
logger.info(f"Fisher diag loaded from {path}")



