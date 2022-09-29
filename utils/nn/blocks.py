from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from ..augmentation import spec


class SwiGLU(nn.Module):
    """SwiGLU activation function. Source: [3]"""

    def forward(self, x:torch.Tensor):
        # |x| : (..., Any)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        # |x| : (..., Any//2)
        return x


class DropBlock2D(nn.Module):
    def __init__(self, p, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_block = spec.DropBlock2D(p, drop_prob, block_size)

    def forward(self, x: torch.Tensor):
        # shape: (bsize, channels, height, width)
        if self.training:
            x = self.drop_block(x)

        return x


class GlobalPooling(nn.Module):
    """Global pooling 
    
    Args:
        mode (str): pooling mode any of: `avg`, `max`, `sum`, `mul`, `swiglu`, `swiglu2`
            avg - average pooling
            max - max pooling
            sum - sum of avg and max pooling
            mul - mul of avg and max pooling
    Shape:
        - Input: `(B, C, T, F)`
        - Output: `(B, C*F)`

    source: [2] II. AUDIO TAGGING SYSTEMS A. CNNs
    """
    def __init__(self, mode: Literal['avg', 'max', 'sum', 'mul'] = None, flat=False):
        super(GlobalPooling, self).__init__()
        self.mode = mode
        self.flat = flat

    def forward(self, x: torch.Tensor, mode=None, flat=None):
        # |x| : (B, C, T, F)
        if mode is None: mode = self.mode
        if flat is None: flat = self.flat
        assert mode is not None
        assert flat is not None

        if flat:
            x = self.flatten(x)

        match self.mode:
            case 'avg':
                return self.avg(x)
            case 'max':
                return self.max(x)
            case 'sum':
                return self.avg(x) + self.max(x)
            case 'mul':
                return self.avg(x) * self.max(x)
    
    def flatten(self, x: torch.Tensor):
        """Flatten x:
        B, C, T, F = x.shape
        x.view(B, T, C*F)
        """
        return x.transpose(2, 3).flatten(1, 2).transpose(1, 2)

    def avg(self, x: torch.Tensor):
        shape = x.shape
        if len(shape) == 4:
            return x.mean(dim=[1, 2])
        elif len(shape) == 3:
            return x.mean(dim=1)
        else:
            raise Exception(f"Unexecpted shape: {shape}")

    def max(self, x: torch.Tensor):
        shape = x.shape
        if len(shape) == 4:
            return x.max(dim=1).values.max(dim=1).values
        elif len(shape) == 3:
            return x.max(dim=1).values
        else:
            raise Exception(f"Unexecpted shape: {shape}")


class ResBlock(nn.Module):
    def __init__(self, 
        kernel_size: list[int], kernel_size_res: list[int], activation: str, 
        c_in: int, c_emb: int, c_out: int, batch_first:bool=True):
        super(ResBlock, self).__init__()

        # Move to this place due circle importing
        from .tools import get_activation

        dilation = [2 - kernel_size[0]%2, 2 - kernel_size[0]%2]
        dilation_res = [2 - kernel_size_res[0]%2, 2 - kernel_size_res[0]%2]
        
        self.batchnorm_1 = nn.BatchNorm2d(c_in)
        self.batchnorm_2 = nn.BatchNorm2d(c_emb)
        self.batchnorm_2 = nn.BatchNorm2d(c_emb)
        
        self.conv1 = nn.Conv2d(c_in, c_emb, kernel_size, dilation=dilation, padding='same')
        self.conv2 = nn.Conv2d(c_emb, c_out, kernel_size, dilation=dilation, padding='same')
        self.conv_res = nn.Conv2d(c_in, c_out, kernel_size_res, dilation=dilation_res, padding='same')

        self.batch_first = batch_first
        self.act = get_activation(activation)
        
    def forward(self, x: torch.Tensor):
        # |x| : (batch_size, C, H, W)
        # ------------------------- #
        if self.batch_first:
            x = self.batchnorm_1(x)
            x = self.act(x)
        else:
            x = self.act(x)
            x = self.batchnorm_1(x)
        res = self.conv_res(x)
        x = self.conv1(x)
    
        # ------------------------- #
        if self.batch_first:
            x = self.batchnorm_2(x)
            x = self.act(x)
        else:
            x = self.act(x)
            x = self.batchnorm_2(x)
        x = self.conv2(x)
        
        return x + res