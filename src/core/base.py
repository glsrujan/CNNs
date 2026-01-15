import torch
from torch import nn
import torch.nn.functional as F
class Resblock(nn.Module):
    def __init__(self,in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,3, 1,1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        if self.in_channels!=self.out_channels:
            self.skip = nn.Conv2d(self.in_channels,self.out_channels,1,1,0)
        else:
            self.skip = nn.Identity()
        pass
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out+=self.skip(x)
        return F.relu(out)
    
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