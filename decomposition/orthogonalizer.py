# ============================
# Module: alignguard/decomposition/orthogonalizer.py
# ============================
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters



def apply_orthogonal_projection(update_vec, fisher_vecs, retain_top=128):
if fisher_vecs.size(1) < retain_top:
retain_top = fisher_vecs.size(1)
U = fisher_vecs[:, :retain_top]
P = U @ U.T # projection matrix
projected = P @ update_vec
residual = update_vec - projected
return projected, residual


