import torch
from torch import nn
from base import Resblock

class CNNBackbone(nn.Module):
    def __init__(self,in_channels, out_channels,bocks = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # self.model = nn.Sequential(
        #             nn.Conv2d(in_channels,8,(3,3),1,1),
        #             nn.BatchNorm2d(8),
        #             nn.ReLU(),
        #             nn.MaxPool2d(2),
        #             nn.Conv2d(8,self.out_channels,(3,3),1,1),
        #             nn.BatchNorm2d(self.out_channels),
        #             nn.ReLU(),
        #         )
        self.model = nn.Sequential(
                    nn.Conv2d(in_channels,8,(3,3),1,1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8,self.out_channels,(3,3),1,1),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(),
                )
    def forward(self,x):
        return self.model.forward(x)

class resnet4(nn.Module):
    def __init__(self, in_channels,out_channels,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.backbone = nn.Sequential(
            Resblock(self.in_channels,8),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            Resblock(8,self.out_channels),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        pass
    
    def forward(self,x):
        return self.backbone(x)

class BCmodel(nn.Module):
    def __init__(self, in_channels, out_channels, nc,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nc = nc
        self.model = nn.Sequential(
            CNNBackbone(self.in_channels, self.out_channels),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.out_channels, self.nc)
        )

    def forward(self, x):
        return self.model(x)

class Resnet4Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, nc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nc = nc
        
        self.model = nn.Sequential(
            resnet4(self.in_channels, self.out_channels),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.out_channels,self.nc)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    resnet8 = Resnet4Classifier(3,16,1)
    
    data = torch.randn((5,3,64,64))
    
    logits = resnet8.forward(data)
    print(resnet8)
    print(logits.shape)