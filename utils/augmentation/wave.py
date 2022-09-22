"""Waveform-based augmentations and tools"""

import random

import torch
import torch.nn.functional as F
import torchaudio
import scipy.signal
import numpy as np

from .typing import WaveAug, BatchAug
from ..tools import get_relative_path


class RIR(WaveAug, BatchAug):
    rirs_samples = {
        'hall': (0.01, 0.3),
        'room': (1.01, 1.3),
        'church': (0.02, 0.3),
    }

    def __init__(self, sr:int, p:float):
        assert 0 <= p <= 1
        # raise NotImplementedError
        self.sr = sr
        self.p = p

        path = get_relative_path(__file__)
        path = path.parent.parent / 'data/RIR_converted'

        self.rirs: dict[str, torch.Tensor] = {}
        for sample in self.rirs_samples:
            rir_raw, _sr = torchaudio.load(path / f'rir {sample}.wav')
            assert _sr == sr

            start, end = self.rirs_samples[sample]
            rir = rir_raw[:, int(_sr * start) : int(_sr * end)]
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])

            self.rirs[sample] = rir

    def __call__(self, audio: torch.Tensor, p=None, pad=True):
        shape: list[int] = audio.shape
        assert len(shape) in [1, 2]
        if p is None: p = self.p

        if random.random() < p:
            if len(shape) == 1:
                audio = audio[None]
        
            sample = random.choice(list(self.rirs_samples.keys()))
            rir = self.rirs[sample]

            if pad: 
                audio = F.pad(audio, (rir.shape[1] - 1, 0))
            audio = F.conv1d(audio[:, None], rir[None, ...])[:, 0]

            if len(shape) == 1:
                # Return the same shape as input
                audio = audio[0]

        return audio

    def __str__(self):
        return f'RIR | sr={self.sr} | p={self.p}'


class LPHPFilter(WaveAug):
    def __init__(self, sr, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.sr = sr
        self.num_taps = int(sr/11025 * 5)
        if self.num_taps%2 == 0:
            self.num_taps+=1
            
    def __call__(self, audio: torch.Tensor, p=None):
        assert len(audio.shape) == 1
        if p is None: p = self.p

        if random.random() < p:
            _max = audio.abs().max()
            if random.random() < 0.5:
                fc = 0.5 + random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming')
            else:
                fc = random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming', pass_zero=False)
            filt = torch.from_numpy(filt).float()
            filt = filt / filt.sum()
            audio = F.pad(audio.view(1, 1, -1), (filt.shape[0]//2, filt.shape[0]//2), mode="reflect")
            audio = F.conv1d(audio, filt.view(1, 1, -1), stride=1, groups=1)
            audio = audio.view(-1)
            audio/= audio.abs().max() 
            audio*= _max
        return audio

    def __str__(self):
        return f'LPHPFilter | sr={self.sr} | p={self.p}'


class TimeShift(WaveAug):
    def __init__(self, max_time_shift=None, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.max_time_shift = max_time_shift

    def __call__(self, audio: torch.Tensor, p=None):
        assert len(audio.shape) == 1
        if p is None: p = self.p

        if random.random() < p:
            if self.max_time_shift is None:
                self.max_time_shift = audio.shape[-1] // 10
            int_d = 2*random.randint(0, self.max_time_shift)-self.max_time_shift
            frac_d = np.round(100*(random.random()-0.5)) / 100
            if int_d + frac_d == 0:
                return audio
            if int_d > 0:
                pad = torch.zeros(int_d, dtype=audio.dtype)
                audio = torch.cat((pad, audio[:-int_d]), dim=-1)
            elif int_d < 0:
                pad = torch.zeros(-int_d, dtype=audio.dtype)
                audio = torch.cat((audio[-int_d:], pad), dim=-1)
            else:
                pass
            if frac_d == 0:
                return audio
            n = audio.shape[-1]
            dw = 2 * np.pi / n
            if n % 2 == 1:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi, -dw).flip(dims=(-1,))
            else:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi - dw, -dw).flip(dims=(-1,))
            w = torch.cat((wp, wn), dim=-1)
            phi = frac_d * w
            audio = torch.fft.ifft(torch.fft.fft(audio) * torch.exp(-1j * phi)).real
        return audio

    def __str__(self):
        return f'TimeShift | max_time_shift={self.max_time_shift} | p={self.p}'


class AdditiveUN(WaveAug):
    def __init__(self, min_snr_db=30, max_snr_db=10, p=0.5):
        assert 0 <= p <= 1
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def __call__(self, audio: torch.Tensor, p=None):
        assert len(audio.shape) == 1
        if p is None: p = self.p

        if random.random() < p:
            s = torch.sqrt(torch.mean(audio ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.max_snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.) * np.sqrt(3)
            w = torch.rand_like(audio).mul_(2 * sgm).add_(-sgm)
            return audio + w
        return audio

    def __str__(self):
        return f'AdditiveUN | min_snr_db={self.min_snr_db} | max_snr_db={self.max_snr_db} | p={self.p}'
