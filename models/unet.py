from typing import List

import torch
import torch.nn as nn


class UNetEncoder(nn.Module):
    def __init__(self, 
                 in_chans=1, 
                 depths=[2, 2, 2, 2, 1], 
                 dims=[32, 64, 128, 256, 320],
                 ):
        super().__init__()

        self.stem = self._get_bottom_layer(in_chans, dims[0])
        self.stages = nn.ModuleList() 
        self.downsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
            )
            if i != len(depths) - 1:
                self.downsample_layers.append(self._get_down_layer(dims[i], dims[i+1]))

            self.stages.append(stage)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )

    def _get_down_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def get_downsample_ratio(self) -> int:
        return 2 ** (self.n_stages - 1)
    
    def get_feature_map_channels(self) -> List[int]:
        return self.dims
    
    def forward(self, inp_bchwd: torch.Tensor, hierarchical=False):
        if hierarchical:
            x = self.stem(inp_bchwd)
            ls = []
            for i in range(self.n_stages):
                x = self.stages[i](x)             
                ls.append(x)
                if i != self.n_stages - 1:
                    x = self.downsample_layers[i](x)
            return ls
        else:
            raise NotImplementedError


@torch.no_grad()
def convnet_test():
    cnn = UNetEncoder().cuda()
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W, D = 4, 1, 96, 96, 96
    inp = torch.rand(B, C, H, W, D).cuda()
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 5
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()