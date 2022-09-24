import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader

from .tools import ConfigBase, to_device


class BaseTrainer(ConfigBase):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    sheduler: torch.optim.lr_scheduler._LRScheduler | None
    device: str
    accum_iter: int
    grad_clip: float

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)

        x = to_device(x, self.device)
        pred: torch.Tensor = self.model(x)
        return pred.cpu()

    @torch.no_grad()
    def evaluate(self, dataset: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)

        _pred, _true = [], []
        for batch in dataset:
            batch = to_device(batch, self.device)
            x, y = batch

            pred: torch.Tensor = self.model(x)
            _pred.append(pred.cpu())
            _true.append(y.cpu())

        return torch.cat(_pred), torch.cat(_true)

    def checkpoint(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self._get_config(),
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.optimizer.zero_grad()
        self.model.train()
        self.model.to(self.device)

        step_idx = None
        running_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = to_device(batch, self.device)
            running_loss+= self.train_step(batch)

            if ((batch_idx + 1) % self.accum_iter == 0):
                step_idx = batch_idx
                self.step()

        if step_idx != batch_idx:
            self.step()

        if self.sheduler: 
            self.sheduler.step()
            
        return running_loss/batch_idx

    def train_step(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def step(self):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.grad_clip, 
                error_if_nonfinite=False
            )
        self.optimizer.step()
        self.optimizer.zero_grad()


class SupervisedClassification(BaseTrainer):
    loss_fns = nn.BCEWithLogitsLoss | nn.CrossEntropyLoss
    metrics_type = None | torchmetrics.Metric | list[torchmetrics.Metric] | dict[str, torchmetrics.Metric]

    def __init__(self, 
        model, loss_fn, optimizer, sheduler=None,
        metrics=None, device='cuda', 
        accum_iter:int=1, grad_clip=0., **kwargs):
        assert accum_iter >= 1
        assert grad_clip >= 0

        self.model: nn.Module = model.to(device)
        self.device = device
        self.metrics: self.metrics_type = metrics
        self.accum_iter = accum_iter
        self.grad_clip = grad_clip

        self.metrics = to_device(self.metrics, device)

        self.loss_fn: self.loss_fns = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer   
        self.sheduler: torch.optim.lr_scheduler._LRScheduler = sheduler
            

    def train_step(self, batch: list[torch.Tensor, torch.Tensor]):
        x, y = batch
        outputs: torch.Tensor = self.model(x)

        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y = y.float()
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            elif len(y.shape) > 2:
                raise Exception(f"bad Y shape: {y.shape}")

            loss: torch.Tensor = self.loss_fn(outputs, y)
            outputs = outputs.sigmoid()

        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y = y.long()
            if len(y.shape) == 2:
                # |y| : (Batch, Num_clases)
                outputs = outputs.softmax(dim=1)
                loss: torch.Tensor = self.loss_fn(outputs, y)
            elif len(y.shape) == 1:
                # |y| : (Batch)
                loss: torch.Tensor = self.loss_fn(outputs, y)
                outputs = outputs.softmax(dim=1)
            else:
                raise Exception(f"bad Y shape: {y.shape}")

        else:
            raise Exception("Uncorrect LossFN")

        loss.backward()
        with torch.no_grad():
            if isinstance(self.metrics, torchmetrics.Metric):
                self.metrics(outputs, y)

            elif isinstance(self.metrics, list):
                for metric in self.metrics:
                    metric(outputs, y)

            elif isinstance(self.metrics, dict):
                for key in self.metrics:
                    metric = self.metrics[key]
                    metric(outputs, y)

        return loss.item()


