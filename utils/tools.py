import yaml
import pathlib

import torch
import torchmetrics
from torch.utils.data import DataLoader


def get_relative_path(__file__) -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve()


class ConfigBase:
    @staticmethod
    def _get_config() -> dict:
        path = get_relative_path(__file__)
        path = path.parent.absolute()
        try:
            with open(path / 'config.yaml', 'r') as stream:
                return yaml.safe_load(stream)

        except FileNotFoundError:
            with open(path / 'config.yaml', 'r') as stream:
                return yaml.safe_load(stream)

    def _get_relative_path(self, __file__) -> pathlib.Path:
        return get_relative_path(__file__)


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
