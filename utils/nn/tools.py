import torch
import torch.nn as nn
import torchmetrics


def get_activation(act: str):
    match act.lower():
        case 'gelu':
            return nn.GELU()
        case 'relu':
            return nn.ReLU(),
        case 'lrelu':
            return nn.LeakyReLU(0.01)
        case 'swiglu':
            from .blocks import SwiGLU
            return SwiGLU()
        case _:
            raise Exception("Uncorrect activation function name")


def to_device(batch: list|torch.Tensor|dict|torchmetrics.Metric, device) \
    -> list|torch.Tensor|dict:
    """Move any type of batch|nn.Module to device"""
    def recurssive_dict(batch: dict):
        for k, v in batch.items():
            if isinstance(v, dict):
                batch[k] = recurssive_dict(v)
            elif isinstance(v, (list, tuple)):
                batch[k] = recurssive_list(v)
            elif isinstance(v, (torch.Tensor, torchmetrics.Metric)):
                batch[k] = v.to(device)
        return batch
    
    def recurssive_list(batch: list):
        batch = list(batch)
        for idx, item in enumerate(batch):
            if isinstance(item, (list, tuple)):
                batch[idx] = recurssive_list(item)
            elif isinstance(item, dict):
                batch[idx] = recurssive_dict(item)
            elif isinstance(item, (torch.Tensor, torchmetrics.Metric)):
                batch[idx] = item.to(device)
        return batch

    if batch is None:
        return None
    elif isinstance(batch, dict):
        return recurssive_dict(batch)
    elif isinstance(batch, (list, tuple)):
        return recurssive_list(batch)
    elif isinstance(batch, (torch.Tensor, torchmetrics.Metric)):
        return batch.to(device)
    else:
        raise Exception("Unexcepted batch type:", type(batch))