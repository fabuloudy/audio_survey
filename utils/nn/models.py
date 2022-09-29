import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


from .tools import get_activation
from .blocks import (ResBlock, DropBlock2D, GlobalPooling)
from ..tools import ConfigBase


class CNNModel(ConfigBase, nn.Module):
    chckpoint_blocks = (ResBlock, nn.Linear)

    @torch.no_grad()
    def __init__(self, input_shape: list[int, int, int, int], num_classes:int, build_verbose=False):
        super(CNNModel, self).__init__()
        # |input_shape| : (B, C, T, F)

        config = self._get_config()
        self.use_checkpoint = config['train']['use_checkpoint']
        
        config = config['model']
        self.verbose = False
        self.blocks = nn.ModuleList()
        tensor: torch.Tensor = torch.rand(*tuple(input_shape))
        get_key = lambda x: list(x.keys())[0]

        if build_verbose: self._verbose(tensor)
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
                    _block = DropBlock2D(values[0], values[1], values[2:])
                    tensor = _block(tensor)
                    self.blocks.append(_block)

                case 'dropout':
                    _block = nn.Dropout2d(values)
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

            if build_verbose: self._verbose(tensor, _block)

        _block = nn.Linear(tensor.shape[-1], num_classes)
        tensor = _block(tensor)
        self.blocks.append(_block)
        if build_verbose: self._verbose(tensor, _block)

    def change_classfication(self, out_features):
        """Remove last classification layer and add new one with correct output shape
        
        Убирает последний классификационный слой и добавляет новый с нужным количеством выходных классов
        """
        last_linear: nn.Linear = self.blocks[-1]
        new_linear = nn.Linear(last_linear.in_features, out_features)
        del self.blocks[-1]
        self.blocks.append(new_linear)

    def forward(self, x: torch.Tensor):
        # |x| : (B, C, T, F)
        if self.use_checkpoint and self.training:
            x = torch.autograd.Variable(x, requires_grad=True)

        for block in self.blocks:
            if self.use_checkpoint and isinstance(block, self.chckpoint_blocks) and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        return x

    @torch.no_grad()
    def _verbose(self, x: torch.Tensor, block: nn.Module|None = None) -> int:
        if block is None:
            print("Input shape")
        else:
            print(f"block={type(block).__name__}")

        print(f"{x.shape=}")
        print()
