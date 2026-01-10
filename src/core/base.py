import torch
from torch import nn

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(self.in_channels,self.out_channels,(3,3),1,1)
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,(3,3),1,1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        if self.in_channels != self.out_channels:
            self.resConv = nn.Conv2d(self.in_channels,self.out_channels,(1,1),1,0)
        else:
            self.resConv = None
    def forward(self,x):
        res = x
        _x = self.conv1(x)
        _bn = self.bn(_x)
        _act = self.act(_bn)
        _x_ = self.conv2(_act)
        if self.resConv:
            res = self.resConv(res)
        _x_+=res
        _bn_ = self.bn(_x_)
        out = self.act(_bn_)
        return out
    
if __name__ == "__main__":
    model = nn.Sequential(
        Resblock(3,8),
        Resblock(8,8),
        Resblock(8,16),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()
    )
    
    data = torch.randn((5,3,64,64))
    out = model.forward(data)
    print(out.shape)