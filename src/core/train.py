from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from backbone import CNNBackbone
from torch import nn, optim
import torch
import wandb
import yaml

transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    root="cnn/data/synthetic",
    transform=transform
)

datasetSize = len(dataset)
trainSize = int(0.8*datasetSize)
valSize = datasetSize - trainSize

trainData, valData = random_split(dataset,[trainSize,valSize])

trainLoader = DataLoader(trainData,batch_size=64,shuffle=True)
valLoader = DataLoader(valData,batch_size=64,shuffle=False)

class BCmodel(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.encoder = CNNBackbone(in_channels, features)
        self.norm = nn.BatchNorm1d(features)
        self.classifier = nn.Linear(features, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


# Training Config
with open("cnn/config/cnn-classify-1.yaml", "r") as f:
    trainingCofig = yaml.safe_load(f)
# wandb stuff
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="dammu",
    # Set the wandb project where this run will be logged.
    project="cnn-classify-1",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": float(trainingCofig["lr"]),
        "architecture": "CNN",
        "dataset": trainingCofig["dataset"],
        "epochs": trainingCofig["epochs"],
    },
)

model = BCmodel(3,16)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),lr = float(trainingCofig["lr"]))

model.train()
for epoch in range(trainingCofig["epochs"]):
    epoch_loss = 0.0
    for batch_idx, (images, labels) in enumerate(trainLoader):
        labels = labels.float().unsqueeze(1)
        output = model(images)
        loss = criterion(output,labels)
        # print(output.shape,labels.shape, loss.item())
        # quit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()/len(output)
        batch_acc = torch.sum(labels.int() == (torch.sigmoid(output)>0.5).int())/len(output)
        print(f"Batch Loss = {loss.item()}, batch Accuracy = {batch_acc}")
        wandb.log({"loss": loss.item(),"Accuracy":batch_acc})
    print(f"Epoch = {epoch}, Epoch Loss = {epoch_loss/len(trainLoader)}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"checkpoint_epoch{epoch}.pth")

wandb.finish()