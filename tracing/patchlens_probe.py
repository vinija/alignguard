# ============================
# Module: alignguard/tracing/patchlens_probe.py
# ============================
import torch
import torch.nn.functional as F



class PatchLensCausalTracer:
def __init__(self, model):
self.model = model



def trace_token_path(self, input_ids, layers, positions, patch_fn=None):
outputs = {}
hooks = []



def _hook_fn(name):
def hook(module, inp, out):
if patch_fn and name in patch_fn:
out = patch_fn[name](out)
outputs[name] = out.detach()
return hook



for layer in layers:
name = f"transformer.h.{layer}.mlp"
mod = dict(self.model.named_modules())[name]
hooks.append(mod.register_forward_hook(_hook_fn(name)))



self.model.eval()
with torch.no_grad():
self.model(input_ids)



for h in hooks:
h.remove()
return outputs



def compute_contribution_change(self, base_outputs, patched_outputs, target_idx):
return {
k: F.mse_loss(patched_outputs[k][:, target_idx], base_outputs[k][:, target_idx])
for k in base_outputs
}



