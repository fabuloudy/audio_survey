def get_folds(dataset: str) -> list:
    """returns num of dataset folds
    
    возращает количество фолдов в датасете
    """
    match dataset.lower():
        case 'esc50':
            return utils.datasets.Meta.esc50['fold'].unique().tolist()
        case 'urban':
            return utils.datasets.Meta.urban['fold'].unique().tolist()
        case _:
            raise Exception("Not found")


def train_esc50(pretrain_name: str=None, tune=False):
    if tune: assert pretrain_name is not None

    def load_pretrain_model():
        """load pretrain model weiths
        
        загружает веса претрейн модели"""
        checkpoint = torch.load(f'weights/{pretrain_name}.torch')
        model.load_state_dict(checkpoint['model'])
        model.change_classfication(num_classes+1)
        if False: optimizer.load_state_dict(checkpoint['optimizer'])

    results = {}
    for fold in get_folds('esc50'):
        # fold - фолд для валидации
        # folds - список с тренировочными фолдами
        folds: list = get_folds('esc50')
        folds.remove(fold)

        name = f'esc50/Fold {fold} {date}'
        print(f"Start train fold:", fold)
        # --------------------------------------------------------- #
        # Создаем тренировочный и валидационный датасеты.
        # Датасеты загружают и предобрабатывают аудио.
        # Каждый датасет имеет методы `__len__` и `__getitem__`
        # которые возращает длину датасета (количество семплов)
        # и конкретный семпл соотвественно.
        # Аргумент `train_dataset` отвечает за приминение аугментации на аудио
        # соотвественно, на валидационном датасете аугментацию применять в нашем кейсе не стоит
        train_dataset = utils.datasets.ESCDataset(
            audio_length=5,
            folds=folds,
            train_dataset=True,
        )
        test_dataset = utils.datasets.ESCDataset(
            audio_length=5,
            folds=[fold],
            train_dataset=False,
        )
        
        # Количество классов в esc50 датасете
        num_classes = utils.datasets.Meta.esc50_num_classes

        # `__iter__()` method in `Dataloader` with >0 `workers` is too slow. 
        # with small dataset like ESC50 we call this method (every epoch) too many times.
        # use `persistent_workers=True` to avoid this 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # `__iter__()` метод в классе `Dataloader` с количеством `workers` больше нуля очень медленный. 
        # Это означает что мы будем вызывать этот метод при начале итерации по тренировочному датасету, т.е. каждую эпоху
        # С маленьким датасетом как ESC50 это может существенно замедлить обучение
        # Стоит указать `persistent_workers=True` для избежания этого

        # Создаем загрузчик данных. 
        # shuffle означает что мы будем перемешивать семплы
        # drop_last означает что мы сбросим последний батч если нам не хватит семплов для его
        # Это когда в датасете количество семплов не кратно размеру батча
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['train']['batch_size'],
            num_workers=2,
            collate_fn=test_dataset.collate_fn,
            persistent_workers=True,
        )
        
        # --------------------------------------------------------- #
        # Формируем первый батч, он нам понадобиться для создания модели
        # для просчёта входных размерностей в слоях и блоках
        for batch in DataLoader(train_dataset, batch_size=1, num_workers=0, collate_fn=train_dataset.collate_fn):
            x, y = batch
            break
        
        # Создаём модель и помещаем её в видеокарточку, если она доступна
        model = utils.nn.models.CNNModel(
            x.shape, 
            num_classes=utils.datasets.Meta.urban_num_classes+1 if tune else num_classes+1, 
            build_verbose=True if fold == 1 else False
        ).to(device)
        # Инициализируем оптимизатор Adam с weight decay 
        # О weight decay можно почитать в источниках [10]
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
        
        # Загружаем веса предобученной модельки
        if tune: load_pretrain_model()
        # Отрисовываем архитектуру модели и количество параметров в консоль
        if fold == 1: torchsummary.summary(model)

        # Инициализируем трейнер
        # Класс для обучения, со всеми необходимыми и удобными методами
        trainer = utils.nn.trainers.SupervisedClassification(
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(),
            accum_iter=config['train']['accum_iter'],
            grad_clip_value=config['train']['grad_clip_value'],
            grad_clip_norm=config['train']['grad_clip_norm'],
            metrics={
                "Acc": torchmetrics.Accuracy(num_classes=num_classes+1),
                "AUC": torchmetrics.AUROC(num_classes=num_classes+1),
            }
        )
        # --------------------------------------------------------- #
        # Обучаем модель
        # Количество эпох указано 0, т.к. мы обучаем модель до тех пор
        # пока улучшается `monitor` метрика на валидационной выборке
        test_metrics = trainer.fit(
            epochs=0,
            name=name,
            patience=config['train']['patience'],
            train_loader=train_loader,
            val_loader=test_loader,
            monitor="Acc",
        )
        # Записываем результаты фолда
        results[fold] = [name, test_metrics]
    
    # Сохраняем конфиг и результаты
    with open(f'logs/results/{name}.json', 'w') as fp:
        json.dump({'config': config, 'results': results}, fp, indent=2)


def pretrain_urban(test_folds: list[int]) -> str:
    assert isinstance(test_folds, list)

    train_folds: list = get_folds('urban')
    [train_folds.remove(fold) for fold in test_folds]

    print("train folds", train_folds)
    print("test folds", test_folds)

    name = f'urban/{date}'
    print(f"Start train")
    # --------------------------------------------------------- #
    # Создаем тренировочный и валидационный датасеты.
    # Balanced датасет случайным образом выбирает класс из датасета 
    # и случайным образом выбирает семпл принадлежащий этому классу.
    # Это позволяет решить проблему не сбалансированности этого датасета.
    # При дизбалансе классов, когда каких-то классов больше или меньше чем других
    # модель хуже обучается и больше зацикливается на доминирующих классах
    train_dataset = utils.datasets.UrbanDatasetBalanced(
        num_iteration=len(utils.datasets.Meta.urban),
        audio_length=5,
        folds=train_folds,
        train_dataset=True,
    )
    test_dataset = utils.datasets.UrbanDataset(
        audio_length=5,
        folds=test_folds,
        train_dataset=False,
    )

    num_classes = utils.datasets.Meta.urban_num_classes

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['pretrain']['batch_size'],
        num_workers=config['pretrain']['num_workers'],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=False,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['pretrain']['batch_size'],
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
        persistent_workers=False,
    )

    # --------------------------------------------------------- #
    for batch in DataLoader(train_dataset, batch_size=1, num_workers=0, collate_fn=train_dataset.collate_fn):
        x, y = batch
        break

    model = utils.nn.models.CNNModel(x.shape, num_classes=num_classes+1, build_verbose=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain']['lr'], weight_decay=config['pretrain']['weight_decay'])
    torchsummary.summary(model)

    model.verbose = True
    trainer = utils.nn.trainers.SupervisedClassification(
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        accum_iter=config['pretrain']['accum_iter'],
        grad_clip_value=config['pretrain']['grad_clip_value'],
        grad_clip_norm=config['pretrain']['grad_clip_norm'],
        metrics={
            "Acc": torchmetrics.Accuracy(num_classes=num_classes+1),
            "AUC": torchmetrics.AUROC(num_classes=num_classes+1),
        }
    )
    
    # --------------------------------------------------------- #
    test_metrics = trainer.fit(
        epochs=0,
        patience=config['pretrain']['patience'],
        name=name,
        train_loader=train_loader,
        val_loader=test_loader,
        monitor="Acc",
    )
    with open(f'logs/results/{name}.json', 'w') as fp:
        json.dump({'config': config, 'results': test_metrics}, fp, indent=2)
        
    return name


if __name__ == '__main__':
    # Импортируем библеотеки в этом блоке 
    # что бы не импортировать их в каждом сабпроцессе 
    # и не занимать оперативную память
    # по крайней мере так работает мультипоточность python на windows
    import json
    import datetime

    import torch
    import torchsummary
    import torchmetrics
    from torch.utils.data import DataLoader

    import utils
    
    config = utils.tools.ConfigBase._get_config()
    date = datetime.datetime.now().strftime("%Y.%m.%d - %H-%M")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(date)
    print(device)

    # Запускаем предобучение на urbansound датасете, если то указано в конфиге
    # 1 и 2 фолды будут использоваться как валидационные, а остальные как тренировочные
    # Функция предобучения возращает название модели, что бы мы могли потом её загрузить на
    # дообучении на ESC50
    name = pretrain_urban([1, 2]) if config['pretrain_urban'] else None
    train_esc50(pretrain_name=name, tune=config['pretrain_urban'])
    
    