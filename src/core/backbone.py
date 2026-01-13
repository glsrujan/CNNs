import torch
from torch import nn
from src.core.base import Resblock

class CNNBackbone(nn.Module):
    def __init__(self,in_channels, out_channels,bocks = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.model = nn.Sequential(
                    Resblock(self.in_channels,8),
                    # Resblock(8,16),
                    nn.MaxPool2d(2,stride=2),
                    nn.Dropout2d(p=0.5),
                    # Resblock(16,32),
                    # nn.MaxPool2d(2,stride=2),
                    Resblock(8,self.out_channels),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten()
                )
        
    def forward(self,x):
        return self.model.forward(x)

class BCmodel(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.model = nn.Sequential(
            # CNNBackbone(in_channels, features),
            nn.Conv2d(in_channels,8,(3,3),1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,features,(3,3),1,1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(features, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    resnet8 = CNNBackbone(3,16)
    
    
    data = torch.randn((5,3,64,64))
    
    logits = resnet8.forward(data)
    print(resnet8)
    print(logits.shape)