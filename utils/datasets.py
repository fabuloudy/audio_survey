import glob
import time
import random
import pathlib

import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd

from .tools import ConfigBase
from .augmentation import spec, wave
from .augmentation.typing import WaveAug, SpecAug, BatchAug, Aug


class AudioDataset(ConfigBase, torch.utils.data.Dataset):
    def __init__(self):
        self.root: pathlib.Path
        self.meta: pd.DataFrame
        self.fpath: list[str]
        self.fnames: list[str]
        self.sample_rate: int
        self.audio_length: float
        self.wave_transforms: list[WaveAug|BatchAug]
        self.spec_transforms: list[SpecAug|BatchAug]

        self.config = self._get_config()
        self.sample_rate = self.config['sample_rate']

        config = self.config['augmentation']
        self.wave_transforms = [
            wave.RIR(self.sample_rate, **config['reverb']),
            wave.TimeShift(None, **config['time_shift']),
            wave.AdditiveUN(**config['noise'])
        ]
        self.spec_transforms = [
            spec.DropBlock2D(
                config['dropblock']['p'], 
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
        ]

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
            return lvl*(audio / audio.abs().max())

        elif len(shape) == 2:
            return lvl*(audio / audio.abs().max(dim=1, keepdim=True).values)

        else:
            raise Exception
    
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
        
        audio = self.add_noise(audio[0])
        audio = self.pad(audio, self.audio_length * sr)
        return audio, label

    def __len__(self):
        return len(self.fnames)

    def collate_fn(self, batch: list[ tuple[torch.Tensor, torch.Tensor] ]):
        """Forms a batch

        Merges a list of samples to form a mini-batch.
        Optimally applies batch augmentation and other things

        batch: list of batch size with 2 items in each: x, y

        - Output Shape: `(B, C, T, F)`
            where B - batch size, C - channel number == 1, T - temporal size, F - frequency size

        
        """
        x, y = [x[0] for x in batch], [x[1] for x in batch]
        y = torch.LongTensor(y)

        # ---------------------------------------------------------------------- #
        x = [self.apply_augs(_x, self.wave_transforms, batch=False) for _x in x]

        x = torch.vstack(x)
        x = self.apply_augs(x, self.wave_transforms, batch=True) 
        x = self.normalize(x)

        # ---------------------------------------------------------------------- #
        x = [self.convert_to_mel_spec(_x).unsqueeze(0) for _x in x]
        x = [self.apply_augs(_x, self.spec_transforms, batch=False) for _x in x]
        x = torch.vstack(x)
        x = x.unsqueeze(1)
        x = self.apply_augs(x, self.spec_transforms, batch=True) 

        # If odd - add one empty vector to F-dim
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
    def __init__(self, audio_length:int, folds:list[int]):
        super().__init__()

        self.folds = folds
        self.audio_length = audio_length

        root = self._get_relative_path(__file__)
        root = root.parent / 'data/ESC-50'
        self.root = root.resolve()

        fnames = glob.glob("audio_converted/*.wav", root_dir=self.root)
        self.fpath = [f for f in fnames if int(f.split('\\')[1].split('-')[0]) in folds]
        self.fnames = [f.split('\\')[-1] for f in self.fpath]
        self.meta = pd.read_csv(self.root / 'meta/esc50.csv').set_index('filename')
        self.meta.rename(columns={'target': 'y'}, inplace=True)
        

class UrbanDataset(AudioDataset):
    def __init__(self, audio_length:int, folds:list[int]):
        super().__init__()

        self.folds = folds
        self.audio_length = audio_length

        root = self._get_relative_path(__file__)
        root = root.parent / 'data'
        self.root = root.resolve()
        
        fnames = glob.glob("**/*.wav", root_dir=self.root / "UrbanSound8k_converted/")
        self.fpath = [f for f in fnames if int(f.split('\\')[0].replace('fold', '')) in folds]
        self.fnames = [f.split('\\')[-1] for f in self.fpath]

        self.meta = pd.read_csv(self.root / 'UrbanSound8k/UrbanSound8k.csv').set_index('slice_file_name')
        self.meta.rename(columns={'classID': 'y'}, inplace=True)
        self.root = self.root / "UrbanSound8k_converted/"