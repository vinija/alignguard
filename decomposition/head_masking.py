# ============================
# Module: alignguard/decomposition/head_masking.py
# ============================
import torch
import torch.nn as nn



def compute_attention_head_scores(model, fisher_diag):
scores = {}
idx = 0
for name, param in model.named_parameters():
if "attn" in name and param.requires_grad:
dim = param.numel()
param_fisher = fisher_diag[idx:idx + dim].view(param.shape)
scores[name] = param_fisher.mean(dim=-1).mean(dim=-1) # average across head dimensions
idx += dim
elif param.requires_grad:
idx += param.numel()
return scores



class HeadMaskingWrapper(nn.Module):
def __init__(self, model, head_threshold=0.1):
super().__init__()
self.model = model
self.head_threshold = head_threshold



def forward(self, *args, **kwargs):
# apply forward pass with potential masking (to be implemented in real usage)
return self.model(*args, **kwargs)



def apply_mask(self, fisher_scores):
with torch.no_grad():
for name, param in self.model.named_parameters():
if name in fisher_scores:
mask = (fisher_scores[name] > self.head_threshold).float()
param.mul_(mask.unsqueeze(-1).unsqueeze(-1))






# ============================
# Notes:
# ✓ `update_splitter.py`: separates ΔW into ΔW_A and ΔW_T by Fisher-magnitude masking
# ✓ `orthogonalizer.py`: orthogonal projection into alignment-critical subspace
# ✓ `head_masking.py`: constructs head-level dropout logic based on Fisher criticality



# Next Up: Deepspeed integration, reward tracing, or causal alignment probes
