from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from backbone import CNNBackbone
from torch import nn, optim
import torch
import wandb
import yaml

# Training Config
with open("config/cnn-classify-1.yaml", "r") as f:
    trainingCofig = yaml.safe_load(f)
    
transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    root=trainingCofig["dataset"],
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
if "loadModel" in trainingCofig.keys() and trainingCofig["loadModel"]:
    state_dict = torch.load(trainingCofig["loadModel"])
    model.load_state_dict(state_dict)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),lr = float(trainingCofig["lr"]))

for epoch in range(trainingCofig["epochs"]):
    epoch_loss_train = 0.0
    
    model.train()
    for batch_idx, (images, labels) in enumerate(trainLoader):
        labels = labels.float().unsqueeze(1)
        output = model(images)
        loss = criterion(output,labels)
        # print(output.shape,labels.shape, loss.item())
        # quit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
        batch_acc_train = torch.sum(labels.int() == (torch.sigmoid(output)>0.5).int())/len(output)
        print(f"Batch Loss train = {loss.item()}, batch Accuracy train= {batch_acc_train}")
        wandb.log({"loss_train": loss.item(),"Test Accuracy":batch_acc_train})
    print(f"Epoch = {epoch}, Epoch Loss train= {epoch_loss_train/len(trainLoader)}")
    
    epoch_loss_val = 0.0
    model.eval()
    for batch_idx, (images, labels) in enumerate(valLoader):
        labels = labels.float().unsqueeze(1)
        with torch.no_grad():
            output = model(images)
            loss = criterion(output,labels)
        epoch_loss_val+=loss.item()
        batch_acc_val = torch.sum(labels.int() == (torch.sigmoid(output)>0.5).int())/len(output)
        wandb.log({"loss_val": loss.item(),"Val Accuracy":batch_acc_val})
        print(f"Batch Loss train = {loss.item()}, batch Accuracy train= {batch_acc_val}")
        pass
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{trainingCofig["modelsavepath"]}/checkpoint_epoch{epoch}.pth")

wandb.finish()