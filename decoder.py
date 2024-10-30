import torch
import torch.nn as nn
from typing import List
from torch.distributions.normal import Normal


class UNetDecoder(nn.Module):
    def __init__(self, 
                 out_chans=1, 
                 depths=[2, 2, 2, 2, 0], 
                 dims=[32, 64, 128, 256, 320],
                 ):
        super().__init__()

        self.stages = nn.ModuleList() 
        self.upsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        self.width = dims
        
        for i in range(len(depths)-1, -1, -1):
            if depths[i] > 0:
                stage = nn.Sequential(
                    self._get_bottom_layer(dims[i] * 2, dims[i]),
                    *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
                )
                self.stages.append(stage)

            if i != 0:
                self.upsample_layers.append(self._get_up_layer(dims[i], dims[i-1]))   

        self.proj = nn.Conv3d(dims[0], out_chans, kernel_size=1, stride=1, bias=True)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def _get_up_layer(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = to_dec[0]
        for i, d in enumerate(self.stages):
            x = self.upsample_layers[i](x)
            if i + 1 < len(to_dec):
                x = torch.cat([x, to_dec[i+1]], dim=1)
            x = d(x)

        # return self.proj(x)
        return x

@torch.no_grad()
def convnet_test():
    model = UNetDecoder().cuda()
    input = [
        torch.randn(1, 320, 4, 4, 4).cuda(),
        torch.randn(1, 256, 8, 8, 8).cuda(),
        torch.randn(1, 128, 16, 16, 16).cuda(),
        torch.randn(1, 64, 32, 32, 32).cuda(),
        torch.randn(1, 32, 64, 64, 64).cuda(),
    ]
    print(model)
    print(model(input).shape)


if __name__ == "__main__":
    convnet_test()