# ============================
# Module: alignguard/decomposition/update_splitter.py
# ============================
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters



def split_updates(update_vec, fisher_diag, topk=128):
indices = torch.argsort(fisher_diag, descending=True)[:topk]
mask = torch.zeros_like(update_vec)
mask[indices] = 1
delta_w_a = update_vec * mask
delta_w_t = update_vec * (1 - mask)
return delta_w_a, delta_w_t
