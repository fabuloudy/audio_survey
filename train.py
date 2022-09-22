import yaml
import time

import torch

import utils


if __name__ == "__main__":
    try:
        with open('config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        with open('config.yaml', 'r') as stream:
            config = yaml.safe_load(stream)


    wave_transforms = [
        utils.augmentation.wave.RIR(config['sample_rate'], p=0.5),
        utils.augmentation.wave.LPHPFilter(config['sample_rate'], p=0.5),
        utils.augmentation.wave.TimeShift(None, p=0.5),
        utils.augmentation.wave.AdditiveUN(min_snr_db=30, max_snr_db=1, p=0.8)
    ]

    spec_transforms = [
        utils.augmentation.spec.DropBlock2D(0.5, 16),
        utils.augmentation.spec.TimeMasking(90, 5, 0.5),
        utils.augmentation.spec.FrequencyMasking(5, 5, 0.5),
    ]

    esc_dataset = utils.datasets.ESCDataset(
        audio_length=5,
        folds=[1, 2, 3],
        wave_transforms=wave_transforms,
        spec_transforms=spec_transforms,
    )
    
    start_time = time.time()
    for _ in range(128):
        x, y = esc_dataset[_]
        
    print(time.time() - start_time)
    