import torch.nn as nn
    

class DenseEncoder(nn.Module):
    def __init__(self, cnn, input_size):
        super(DenseEncoder, self).__init__()
        self.sp_cnn = cnn
        self.input_size, self.downsample_ratio, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()
    
    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)
    