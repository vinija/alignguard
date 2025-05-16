# ============================
# Module: alignguard/fim/eigen_utils.py
# ============================
import torch
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Tuple



def topk_spectral_filtering(matrix: torch.Tensor, k: int = 50, symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
"""
Perform top-k eigen decomposition on a (covariance-like) matrix.
"""
matrix_np = matrix.detach().cpu().numpy()
if symmetric:
vals, vecs = eigsh(matrix_np, k=k, which='LM')
else:
raise NotImplementedError("Asymmetric eigen decomposition not yet implemented.")
eigvals = torch.from_numpy(vals).float()
eigvecs = torch.from_numpy(vecs).float()
return eigvals, eigvecs



def low_rank_approximation(matrix: torch.Tensor, rank: int = 64) -> torch.Tensor:
eigvals, eigvecs = topk_spectral_filtering(matrix, k=rank)
recon = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
return recon



def condition_number(matrix: torch.Tensor) -> float:
s = torch.linalg.svdvals(matrix)
return (s[0] / s[-1]).item()






# ============================
# Notes:
# These two files form the base of a >600 LOC FIM module.
# Additional utilities coming next:
# - curvature_variance.py
# - diagonal_smoothing.py
# - fim_cache.py
# - full Fisher-vector implementation
