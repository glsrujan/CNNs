import torch
from torch import nn
from base import Resblock

class CNNBackbone(nn.Module):
    def __init__(self,in_channels, out_channels,bocks = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.model = nn.Sequential(
                    Resblock(self.in_channels,8),
                    Resblock(8,8),
                    Resblock(8,8),
                    Resblock(8,self.out_channels),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten()
                )
        
    def forward(self,x):
        return self.model.forward(x)

if __name__ == "__main__":
    resnet8 = CNNBackbone(3,16)
    
    
    data = torch.randn((5,3,64,64))
    
    logits = resnet8.forward(data)
    print(resnet8)
    print(logits.shape)