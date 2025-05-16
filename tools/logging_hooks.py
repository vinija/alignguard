# ============================
# Module: tools/logging_hooks.py
# ============================
import wandb
import os



def init_logger(project="AlignGuard-LoRA", name=None, tags=None):
wandb.init(project=project, name=name, tags=tags)



def log_metrics(metrics: dict, step=None):
wandb.log(metrics, step=step)



def log_config(config):
wandb.config.update(config)



def finish_logging():
wandb.finish()
