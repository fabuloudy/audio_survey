from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .tools import ConfigBase
from .augmentation import spec
from torch.utils.checkpoint import checkpoint


def get_activation(act: str):
    match act.lower():
        case 'gelu':
            return nn.GELU()
        case 'relu':
            return nn.ReLU(),
        case 'lrelu':
            return nn.LeakyReLU(0.01)
        case 'swiglu':
            return SwiGLU()
        case _:
            raise Exception("Uncorrect activation function name")


class SwiGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""
    def forward(self, x:torch.Tensor):
        # |x| : (..., Any)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        # |x| : (..., Any//2)
        return x


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_block = spec.DropBlock2D(drop_prob, block_size)

    def forward(self, x: torch.Tensor):
        # shape: (bsize, channels, height, width)
        return self.drop_block(x)


class GlobalPooling(nn.Module):
    """Global pooling 
    
    Args:
        mode (str): pooling mode any of: `avg`, `max`, `sum`, `mul`, `swiglu`, `swiglu2`
            avg - average pooling
            max - max pooling
            sum - sum of avg and max pooling
            mul - mul of avg and max pooling
    Shape:
        - Input: `(N, C, T, F)`
        - Output: `(N, E)`

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

        dilation = [2 - kernel_size[0]%2, 2 - kernel_size[0]%2]
        dilation_res = [2 - kernel_size_res[0]%2, 2 - kernel_size_res[0]%2]
        
        self.batchnorm_1 = nn.BatchNorm2d(c_in)
        self.batchnorm_2 = nn.BatchNorm2d(c_emb)
        
        self.conv1 = nn.Conv2d(c_in, c_emb, kernel_size, dilation=dilation, padding='same')
        self.conv2 = nn.Conv2d(c_emb, c_out, kernel_size, dilation=dilation, padding='same')
        self.conv_res = nn.Conv2d(c_in, c_out, kernel_size_res, dilation=dilation_res, padding='same')

        self.batch_first = batch_first
        self.act = get_activation(activation)
        
    def forward(self, x: torch.Tensor):
        # |x| : (batch_size, C, H, W)
        res = self.conv_res(x)
        
        # ------------------------- #
        if self.batch_first:
            x = self.batchnorm_1(x)
            x = self.act(x)
        else:
            x = self.act(x)
            x = self.batchnorm_1(x)
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


class Model(ConfigBase, nn.Module):
    chckpoint_blocks = (ResBlock, nn.Linear)

    @torch.no_grad()
    def __init__(self, input_shape: list[int, int, int, int], num_classes:int, build_verbose=False):
        super(Model, self).__init__()
        # |input_shape| : (B, C, T, F)

        config = self._get_config()
        config = config['model']

        self.verbose = False
        self.use_checkpoint = config['use_checkpoint']

        self.blocks = nn.ModuleList()
        tensor: torch.Tensor = torch.rand(*tuple(input_shape))
        get_key = lambda x: list(x.keys())[0]

        if build_verbose:
            print("Input shape")
            print(f"{tensor.shape=}")
            print()

        for block in config['blocks']:
            assert isinstance(block, dict)

            key: str = get_key(block)
            values = block[key]
            match key.lower():
                case 'res':
                    _block = ResBlock(
                        **config['res_block'],
                        c_in=tensor.shape[1], 
                        c_emb=values[0], 
                        c_out=values[1]
                    )
                    tensor = _block(tensor)
                    self.blocks.append(_block)
                    
                case 'max_pool':
                    _block = nn.MaxPool2d(kernel_size=[values[0], 1], stride=[values[1], 1])
                    tensor = _block(tensor)
                    self.blocks.append(_block)

                case 'drop':
                    _block = DropBlock2D(values[0], values[1:])
                    tensor = _block(tensor)
                    self.blocks.append(_block)

                case 'pooling':
                    _block = GlobalPooling(**values)
                    tensor = _block(tensor)
                    self.blocks.append(_block)
                
                case 'fc':
                    _block = nn.Linear(tensor.shape[-1], values)
                    tensor = _block(tensor)
                    self.blocks.append(_block)

                case 'activation':
                    _block = get_activation(values)
                    tensor = _block(tensor)
                    self.blocks.append(_block)

                case _:
                    raise Exception(f"Unexcepted block: {key}")

            if build_verbose:
                print(f"block={type(_block).__name__}")
                print(f"{tensor.shape=}")
                print()

        self.blocks.append(nn.Linear(tensor.shape[-1], num_classes))

    def forward(self, x: torch.Tensor):
        # |x| : (B, C, T, F)
        if self.verbose:
            print("Input shape")
            print(f"{x.shape=}")
            print()
        
        if self.use_checkpoint:
            x = torch.autograd.Variable(x, requires_grad=True)

        for block in self.blocks:
            if self.use_checkpoint and isinstance(block, self.chckpoint_blocks):
                x = checkpoint(block, x)
                
            else:
                x = block(x)

            if self.verbose:
                print(f"block={type(block).__name__}")
                print(f"{x.shape=}")
                print()

        return x

    
