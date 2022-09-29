import glob
import random
import pathlib

import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd

from .tools import ConfigBase
from .augmentation import spec, wave
from .augmentation.typing import WaveAug, SpecAug, BatchAug, Aug


def get_urban_meta() -> pd.DataFrame:
    root = ConfigBase._get_relative_path(__file__)
    root = root.parent / 'data'
    root = root.resolve()
    
    meta = pd.read_csv(root / 'UrbanSound8k/UrbanSound8k.csv').set_index('slice_file_name')
    meta.rename(columns={'classID': 'y'}, inplace=True)
    return meta

def get_esc50_meta()  -> pd.DataFrame:
    root = ConfigBase._get_relative_path(__file__)
    root = root.parent / 'data/ESC-50'
    root = root.resolve()

    meta = pd.read_csv(root / 'meta/esc50.csv').set_index('filename')
    meta.rename(columns={'target': 'y'}, inplace=True)
    return meta

class Meta:
    esc50 = get_esc50_meta()
    urban = get_urban_meta()

    esc50_num_classes = esc50['y'].max()
    urban_num_classes = urban['y'].max()


class AudioDataset(ConfigBase, torch.utils.data.Dataset):
    def __init__(self, train_dataset=True):
        self.root: pathlib.Path
        self.meta: pd.DataFrame
        self.fpath: list[str]
        self.fnames: list[str]
        self.sample_rate: int
        self.audio_length: float
        self.wave_transforms: list[WaveAug|BatchAug]
        self.spec_transforms: list[SpecAug|BatchAug]

        self.train_dataset = train_dataset
        self.config = self._get_config()
        self.sample_rate = self.config['sample_rate']

        config = self.config['augmentation']
        self.wave_transforms = [
            wave.RIR(self.sample_rate, **config['reverb']),
            wave.TimeShift(None, **config['time_shift']),
            wave.AdditiveUN(**config['noise'])
        ] if train_dataset else []
        
        self.spec_transforms = [
            spec.DropBlock2D(
                config['dropblock']['p'], 
                config['dropblock']['dp'], 
                [
                config['dropblock']['block_size']['t'], 
                config['dropblock']['block_size']['f']
                ]
            ),
            spec.TimeMasking(
                config['time_masking']['tmp'], 
                config['time_masking']['num'], 
                config['time_masking']['p']
            ),
            spec.FrequencyMasking(
                config['freq_masking']['fmp'], 
                config['freq_masking']['num'], 
                config['freq_masking']['p']
            ),
        ] if train_dataset else []

        config = self.config['mel_spectrogram']
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_fft=int(self.sample_rate*config['win_length']),
            win_length=int(self.sample_rate*config['win_length']),
            hop_length=int(self.sample_rate*config['hop_length']),
            n_mels=config['n_mels'],
            f_min=config['f_min'],
            f_max=config['f_max'],
        )

    def convert_to_mel_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """Work as well as single sample or batch"""
        mel_x = self.mel_spec(audio)
        mel_x = torch.log10(mel_x + 1)
        return mel_x.swapaxes(-2, -1)

    def normalize(self, audio: torch.Tensor, lvl:float=0.95) -> torch.Tensor:
        shape = audio.shape
        if len(shape) == 1:
            _max = audio.abs().max()

        elif len(shape) == 2:
            _max = audio.abs().max(dim=1, keepdim=True).values
            
        else:
            raise Exception

        return lvl*(audio / _max)
    
    def add_noise(self, audio: torch.Tensor, scale:float=1e3) -> torch.Tensor:
        # -1 to 1 noise
        noise = torch.rand_like(audio) * 2 - 1
        # Scale 
        noise = noise * audio.abs().max()
        noise/= scale
        return audio+noise

    def disering(self, audio: torch.Tensor, scale:float=1e-3) -> torch.Tensor:
        raise NotImplementedError

    def pad(self, audio: torch.Tensor, audio_length:int) -> torch.Tensor:
        assert audio.ndim == 1
        
        if audio.shape[0] >= audio_length:
            max_audio_start = audio.size(0) - audio_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + audio_length]
        else:
            audio = F.pad(audio, (0, audio_length - audio.size(0)), "constant")
            
        return audio

    def __getitem__(self, index):
        fname = self.fnames[index]
        fpath = self.fpath[index]
        label = self.meta.loc[fname]['y']
    
        audio, sr = torchaudio.load(self.root / fpath)
        assert sr == self.sample_rate

        audio = self.pad(audio[0], self.audio_length * sr)
        audio = self.normalize(audio)

        return audio, label

    def __len__(self):
        return len(self.fnames)

    def validate_audio(self, audio: torch.Tensor):
        shape = audio.shape
        assert len(shape) == 1

        _max = audio.abs().max()

        return True if _max > 0 else False

    def collate_fn(self, batch: list[ tuple[torch.Tensor, torch.Tensor] ]):
        """Forms a batch

        Merges a list of samples to form a mini-batch.
        Optimally applies batch augmentation and other things

        batch: list of batch size with 2 items in each: x, y

        - Output Shape: `(B, C, T, F)`
            where B - batch size, C - channel number == 1, T - temporal size, F - frequency size
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Формирует батч

        Объединяет список семплов и формирует мини батч.
        Оптимально применяет аугментацию и другие штучки

        batch: список размером с батч и двумя элементами в каждом: x, y

        - Выходная размерность: `(B, C, T, F)`
            где B - размер батча, C - количество каналов, равно единице, 
            T - временной размер (зависит от длины аудио), 
            F - частотный размер, зависит от параметров алгоритма мел

        """
        x, y = [x[0] for x in batch], [x[1] for x in batch]
        # ---------------------------------------------------------------------- #
        x = [self.apply_augs(_x, self.wave_transforms, batch=False) for _x in x]

        # Skip 0-zero signal audio, idk why this happens
        # Пропускаем аудио с 0 уровнем сигнала, я не знаю почему это случается
        clean_x, clean_y = [], []
        for sample_x, sample_y in zip(x, y):
             if self.validate_audio(sample_x):
                clean_x.append(sample_x)
                clean_y.append(sample_y)

        x = torch.vstack(clean_x)
        y = torch.LongTensor(clean_y)

        x = self.apply_augs(x, self.wave_transforms, batch=True) 
        x = self.normalize(x)

        # ---------------------------------------------------------------------- #
        x = [self.convert_to_mel_spec(_x).unsqueeze(0) for _x in x]
        x = [self.apply_augs(_x, self.spec_transforms, batch=False) for _x in x]
        x = torch.vstack(x)

        x = x.unsqueeze(1)
        x = self.apply_augs(x, self.spec_transforms, batch=True) 

        # If odd - add one empty vector to F-dim
        # Если нечётное - добавляет одно пустое измерение по оси F
        if x.shape[-1]%2 == 1:
            x = F.pad(x, (1, 0, 0, 0))

        return x, y

    def apply_augs(self, x: torch.Tensor, transforms: list[Aug], batch=False):
        if transforms is not None:
            for transform in transforms:
                if isinstance(transform, BatchAug) == batch:
                    x = transform(x)
        return x


class ESCDataset(AudioDataset):
    def __init__(self, audio_length:int, folds:list[int], train_dataset: bool):
        super().__init__(train_dataset=train_dataset)

        self.folds = folds
        self.audio_length = audio_length

        root = self._get_relative_path(__file__)
        root = root.parent / 'data/ESC-50'
        self.root = root.resolve()

        fnames = glob.glob("audio_converted/*.wav", root_dir=self.root)
        self.fpath = [f for f in fnames if int(f.split('\\')[1].split('-')[0]) in folds]
        self.fnames = [f.split('\\')[-1] for f in self.fpath]

        self.names_index = {n:idx for idx, n in enumerate(self.fnames)}

        meta = Meta.esc50.copy()
        meta['_b'] = meta['fold'].apply(lambda x: x in folds)
        self.meta = meta[meta['_b']]


class ESCDatasetBalanced(ESCDataset):
    r"""Balanced ESC50 Dataset

    Randomly choosing samples per randomly choosed class.

    Args:
        num_iteration - total number samples per epoch. Total number of batches:
            N//B + s, where N - num_iteration, B - batch size, s - 1 if `DataLoader`
            `drop_last=False` and N%B != 0  else 0
    ------------------------------------------------------------------------------
    TODO:Комментарий на русском
    
    """
    def __init__(self, num_iteration, *args, **kwars):
        super().__init__(*args, **kwars)
        self.rand_range = super().__len__() - 1
        self.num_iteration = num_iteration
        self.classes = self.meta['y'].unique()

    def __len__(self):
        return self.num_iteration

    def __getitem__(self, _):
        _class = random.choice(self.classes)
        fname = random.choice(self.meta[self.meta['y'] == _class].index)
        return super().__getitem__(self.names_index[fname])


class UrbanDataset(AudioDataset):
    def __init__(self, audio_length:int, folds:list[int], train_dataset: bool):
        super().__init__(train_dataset=train_dataset)

        self.folds = folds
        self.audio_length = audio_length

        root = self._get_relative_path(__file__)
        root = root.parent / 'data/UrbanSound8k_converted'
        self.root = root.resolve()
        
        fnames = glob.glob("**/*.wav", root_dir=self.root)
        self.fpath = [f for f in fnames if int(f.split('\\')[0].replace('fold', '')) in folds]
        self.fnames = [f.split('\\')[-1] for f in self.fpath]

        self.names_index = {n:idx for idx, n in enumerate(self.fnames)}

        meta = Meta.urban.copy()
        meta['_b'] = meta['fold'].apply(lambda x: x in folds)
        self.meta = meta[meta['_b']]
        

class UrbanDatasetBalanced(UrbanDataset):
    r"""Balanced Urban Dataset

    Randomly choosing samples per randomly choosed class.

    Args:
        num_iteration - total number samples per epoch. Total number of batches:
            N//B + s, where N - num_iteration, B - batch size, s - 1 if `DataLoader`
            `drop_last=False` and N%B != 0  else 0
    ------------------------------------------------------------------------------
    TODO:Комментарий на русском
    
    """
    def __init__(self, num_iteration, *args, **kwars):
        super().__init__(*args, **kwars)
        self.rand_range = super().__len__() - 1
        self.num_iteration = num_iteration
        self.classes = self.meta['y'].unique()

    def __len__(self):
        return self.num_iteration

    def __getitem__(self, _):
        _class = random.choice(self.classes)
        fname = random.choice(self.meta[self.meta['y'] == _class].index)
        return super().__getitem__(self.names_index[fname])
