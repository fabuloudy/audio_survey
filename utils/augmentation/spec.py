"""Scpectogram-based augmentations and tools"""

import random

import torch
import torch.nn.functional as F
import torchaudio

from .typing import SpecAug, BatchAug


class FrequencyMasking(SpecAug, BatchAug):
    def __init__(self, freq_mask_param:int, n=1, p=0.5):
        self.p = p
        self.n = n
        
        self.aug = torchaudio.transforms.TimeMasking(time_mask_param=freq_mask_param)

    def __call__(self, spec: torch.Tensor, p=None):
        # |spec| : (batch_size, C, T, F)
        # C - optional, channel nubmer, probably 1
        # T - temporal size
        # F - frequency size
        assert len(spec.shape) in [3, 4]
        if p is None: p = self.p

        if random.random() < p:
            for _ in range(self.n):
                spec = self.aug(spec)
        return spec
        
        
class TimeMasking(SpecAug, BatchAug):
    def __init__(self, time_mask_param:int, n=1, p=0.5):
        self.p = p
        self.n = n
        self.aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=time_mask_param)

    def __call__(self, spec: torch.Tensor, p=None):
        # |spec| : (batch_size, C, T, F)
        # C - optional, channel nubmer, probably 1
        # T - temporal size
        # F - frequency size
        assert len(spec.shape) in [3, 4]
        if p is None: p = self.p

        if random.random() < p:
            for _ in range(self.n):
                spec = self.aug(spec)
        return spec


class DropBlock2D(SpecAug, BatchAug):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    source: https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
    """

    def __init__(self, drop_prob, block_size: tuple|list|int):
        self.drop_prob = drop_prob
        if isinstance(block_size, int):
            self.block_size = (block_size, block_size)
        else:
            assert len(block_size) == 2
            self.block_size = block_size

        self.gamma = self.drop_prob / (sum(self.block_size)/2)**2

    def __call__(self, x: torch.Tensor):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.drop_prob == 0.:
            return x
        else:
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < self.gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask: torch.Tensor):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=self.block_size,
            stride=(1, 1),
            padding=(self.block_size[0]//2, self.block_size[1]//2)
        )

        if self.block_size[0] % 2 == 0:
            block_mask = block_mask[:, :, :, :-1]

        if self.block_size[1] % 2 == 0:
            block_mask = block_mask[:, :, :-1, :]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

