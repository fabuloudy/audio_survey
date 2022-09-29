from contextlib import suppress

import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .tools import to_device
from ..tools import ConfigBase


class BaseTrainer(ConfigBase):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    sheduler: torch.optim.lr_scheduler._LRScheduler | None
    device: str
    accum_iter: int
    grad_clip_norm: float
    grad_clip_value: float

    def __init__(self, accum_iter:int=1, grad_clip_norm=0., grad_clip_value=0., **kwargs):
        assert accum_iter >= 1
        assert grad_clip_norm >= 0
        assert grad_clip_value >= 0

        self.accum_iter = accum_iter
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)

        x = to_device(x, self.device)
        pred: torch.Tensor = self.model(x)
        return pred.cpu()

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)

        _pred, _true = [], []
        for batch in dataloader:
            batch = to_device(batch, self.device)
            x, y = batch

            pred: torch.Tensor = self.model(x)
            _pred.append(pred.cpu())
            _true.append(y.cpu())

        return torch.cat(_pred), torch.cat(_true)
    
    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

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
                self.update_gradients()
                step_idx = batch_idx

        if step_idx != batch_idx:
            self.update_gradients()

        if self.sheduler: 
            self.sheduler.step()
            
        return running_loss/batch_idx

    def train_step(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def update_gradients(self):
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.grad_clip_norm, 
                error_if_nonfinite=False
            )
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), 
                clip_value=self.grad_clip_value, 
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

    @staticmethod
    def get_writer(name:str):
        log_dir = f"logs/fit/{name}"
        writer = SummaryWriter(log_dir)
        return writer

class SupervisedClassification(BaseTrainer):
    loss_fns = nn.BCEWithLogitsLoss | nn.CrossEntropyLoss
    metrics_type = None | torchmetrics.Metric | list[torchmetrics.Metric] | dict[str, torchmetrics.Metric]

    def __init__(self, 
        model: nn.Module, loss_fn: nn.Module, optimizer: nn.Module, 
        sheduler=None, metrics=None, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.model: nn.Module = model.to(device)
        self.device = device
        self.metrics: self.metrics_type = metrics
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
    
    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        y_pred, y_true = super().predict(dataloader=dataloader)
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y_pred = y_pred.sigmoid()

        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_pred = y_pred.softmax(dim=1)

        return y_pred, y_true

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        y_pred, y_true = self.predict(dataloader=dataloader)
        metrics = self.compute_metrics(y_pred, y_true)
        return metrics

    @staticmethod
    @torch.no_grad()
    def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred, y_true = y_pred.cpu(), y_true.cpu()
        num_classes = y_pred.shape[1]
        return {
            'Acc': float(torchmetrics.functional.accuracy(y_pred, y_true, top_k=1, num_classes=num_classes)),
            'Acc3': float(torchmetrics.functional.accuracy(y_pred, y_true, top_k=3, num_classes=num_classes)),
            'AUC': float(torchmetrics.functional.auroc(y_pred, y_true, num_classes=num_classes)),
        }

    def __train_condition(self, wait:int, patience:int, epoch:int, epochs:int, mandatory_epochs:int) -> bool:
        """Returns `True` to continue training"""
        if patience == 0:
            return True if epoch < epochs else False

        bool_ = epoch < mandatory_epochs or wait < patience
        if epochs != 0 and bool_ and epoch < epochs:
            return True
        elif bool_:
            return True
        else:
            return False

    def fit(self, 
        epochs: int, 
        patience: int, 
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader|None = None,
        monitor: str = 'Acc',
        mandatory_epochs: int = 0,
        ) -> dict[str, int] | None:

        """Train a model

        Args:
            epochs: num of total train epochs
            patience: if positive non zero value train will early stopped after `patience` epochs without improving `monitor` metrics
            name: model and tensor board name
            train loader: 
            //TODO: fill this out
        
        """
        assert epochs >= 0
        assert patience >= 0
        assert isinstance(self.metrics, dict)
        if patience > 0: assert val_loader is not None

        torch.cuda.empty_cache()

        wait = 0
        epoch = 0
        writer = None
        best_score = -torch.inf
        best_metrics = None
        with suppress(KeyboardInterrupt):
            while self.__train_condition(wait, patience, epoch, epochs, mandatory_epochs):
                # --------------------------------------------- #
                print("Train epoch")
                train_loss = self.train_epoch(train_loader)
                if writer is None: writer = self.get_writer(name)

                for _metric in self.metrics:
                    writer.add_scalar(f'{_metric}/train', self.metrics[_metric].compute(), epoch)

                # --------------------------------------------- #
                if val_loader is None:
                    wait, epoch = wait+1, epoch+1
                    torch.save(self.checkpoint(), f'weights/{name}.torch')
                else:
                    print("Evaluate")
                    metrics = self.evaluate(val_loader)
                    for _metric in metrics:
                        writer.add_scalar(f'{_metric}/val', metrics[_metric], epoch)
                    
                    # --------------------------------------------- #
                    # EarlyStopping
                    wait, epoch = wait+1, epoch+1
                    if metrics[monitor] > best_score:
                        torch.save(self.checkpoint(), f'weights/{name}.torch')
                        best_score = metrics[monitor]
                        best_metrics = metrics
                        wait = 0
                
                

        return best_metrics

