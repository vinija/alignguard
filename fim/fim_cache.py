# ============================
# Module: alignguard/fim/fim_cache.py
# ============================
import torch
import os



class FIMCache:
def __init__(self, cache_dir: str):
self.cache_dir = cache_dir
os.makedirs(self.cache_dir, exist_ok=True)



def save(self, name: str, fisher_tensor: torch.Tensor):
path = os.path.join(self.cache_dir, f"{name}.pt")
torch.save(fisher_tensor, path)



def load(self, name: str) -> torch.Tensor:
path = os.path.join(self.cache_dir, f"{name}.pt")
return torch.load(path)



def exists(self, name: str) -> bool:
path = os.path.join(self.cache_dir, f"{name}.pt")
return os.path.exists(path)



def list_keys(self):
return [f.replace('.pt', '') for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
