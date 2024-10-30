import torch

from models.unet import UNetEncoder


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in [torch.nn.CrossEntropyLoss]:
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


def build_encoder(input_size: int):
    from encoder import DenseEncoder
    
    cnn = UNetEncoder()
    return DenseEncoder(cnn, input_size=input_size)
