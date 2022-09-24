import json
import pprint
import datetime
from contextlib import suppress

import tqdm
import torch
import torchsummary
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import utils
    
def get_writer(name:str):
    log_dir = f"logs/fit/esc50/{name}"
    writer = SummaryWriter(log_dir)
    return writer

@torch.no_grad()
def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred, y_true = y_pred.cpu(), y_true.cpu()
    num_classes = y_pred.shape[1]
    return {
        'Acc': float(torchmetrics.functional.accuracy(y_pred, y_true, top_k=1, num_classes=num_classes)),
        'Acc5': float(torchmetrics.functional.accuracy(y_pred, y_true, top_k=5, num_classes=num_classes)),
        'AUC': float(torchmetrics.functional.auroc(y_pred, y_true, num_classes=num_classes)),
    }


def train():
    config = utils.tools.ConfigBase._get_config()
    date = datetime.datetime.now().strftime("%Y.%m.%d - %H-%M")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(date)
    print(device)
    
    results = {}
    for fold in range(5):
        torch.cuda.empty_cache()
        
        folds = [x for x in range(1, 6)]
        rm_fold = folds.pop(fold)

        writer = None
        name = f'Fold {rm_fold} {date}'
        print(f"Start train fold:", rm_fold)
        # --------------------------------------------------------- #
        train_dataset = utils.datasets.ESCDataset(
            audio_length=5,
            folds=folds,
        )
        test_dataset = utils.datasets.ESCDataset(
            audio_length=5,
            folds=[rm_fold],
        )
        
        num_classes = train_dataset.meta['y'].max()+1
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['train']['batch_size'],
            num_workers=4,
            drop_last=True,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['train']['batch_size'],
            num_workers=4,
            drop_last=False,
            shuffle=False,
            collate_fn=test_dataset.collate_fn
        )
        
        # --------------------------------------------------------- #
        for batch in DataLoader(train_dataset, batch_size=1, num_workers=0, collate_fn=train_dataset.collate_fn):
            x, y = batch
            break

        model = utils.nn.Model(x.shape, num_classes=num_classes, build_verbose=True if fold == 0 else False).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        
        if fold == 0:
            torchsummary.summary(model)

        trainer = utils.trainers.SupervisedClassification(
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(),
            accum_iter=config['train']['accum_iter'],
            grad_clip=config['train']['grad_clip'],
            metrics={
                "Acc": torchmetrics.Accuracy(num_classes=num_classes),
                "AUC": torchmetrics.AUROC(num_classes=num_classes),
            }
        )
        
        # --------------------------------------------------------- #
        # Train
        wait = 0
        epoch = 0
        best_score = -torch.inf
        best_metrics = None
        with suppress(KeyboardInterrupt):
            print("start train")
            while wait < config['train']['patience']:
                # --------------------------------------------- #
                # Train
                train_loss = trainer.train_epoch(train_loader)
                if writer is None: writer = get_writer(name)

                writer.add_scalar('Acc/train', trainer.metrics['Acc'].compute(), epoch)
                writer.add_scalar('AUC/train', trainer.metrics['AUC'].compute(), epoch)

                # --------------------------------------------- #
                # Test    
                test_pred, test_true = trainer.evaluate(test_loader)
                test_pred = test_pred.softmax(dim=1)
                test_metrics = compute_metrics(test_pred, test_true)
                for _metric in test_metrics:
                    writer.add_scalar(f'{_metric}/test', test_metrics[_metric], epoch)
                    
                # --------------------------------------------- #
                # EarlyStopping
                wait, epoch = wait+1, epoch+1
                if test_metrics['AUC'] > best_score:
                    torch.save(trainer.checkpoint(), f'weights/esc50/{name}.torch')
                    best_score = test_metrics['AUC']
                    best_metrics = test_metrics
                    wait = 0
            
        results[rm_fold] = [name, best_metrics]
    
    with open(f'logs/results/{date}.json', 'w') as fp:
        json.dump(results, fp)

    pprint.pprint(results)


if __name__ == '__main__':
    train()