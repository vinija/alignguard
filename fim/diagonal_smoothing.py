# ============================
# Module: alignguard/fim/diagonal_smoothing.py
# ============================
import torch



def smooth_fisher(fisher_diag: torch.Tensor, method='ema', beta=0.9, window=5):
if method == 'ema':
smoothed = torch.zeros_like(fisher_diag)
smoothed[0] = fisher_diag[0]
for i in range(1, len(fisher_diag)):
smoothed[i] = beta * smoothed[i - 1] + (1 - beta) * fisher_diag[i]
return smoothed
elif method == 'moving_avg':
padded = torch.nn.functional.pad(fisher_diag, (window, window), mode='reflect')
return torch.nn.functional.avg_pool1d(padded.unsqueeze(0).unsqueeze(0), kernel_size=2*window+1, stride=1).squeeze()
else:
raise ValueError(f"Unknown smoothing method: {method}")


