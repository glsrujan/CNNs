from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from backbone import Resnet4Classifier, BCmodel
from torch import nn, optim
import torch
import wandb
import yaml

# Training Config
with open("config/resnet-classify-1.yaml", "r") as f:
    trainingCofig = yaml.safe_load(f)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(
    root=trainingCofig["dataset"],
    transform=transform
)

datasetSize = len(dataset)
trainSize = int(0.8*datasetSize)
valSize = datasetSize - trainSize

trainData, valData = random_split(dataset,[trainSize,valSize])

trainLoader = DataLoader(trainData,batch_size=128,shuffle=True)
valLoader = DataLoader(valData,batch_size=128,shuffle=False)





# # wandb stuff
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="dammu",
    # Set the wandb project where this run will be logged.
    project="resnet-classify-1",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": float(trainingCofig["lr"]),
        "architecture": "RESNET4",
        "dataset": trainingCofig["dataset"],
        "epochs": trainingCofig["epochs"],
    },
)

model = Resnet4Classifier(3,16,1)
if "loadModel" in trainingCofig.keys() and trainingCofig["loadModel"]:
    state_dict = torch.load(trainingCofig["loadModel"])
    model.load_state_dict(state_dict)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),lr = float(trainingCofig["lr"]))

for epoch in range(trainingCofig["epochs"]):
    epoch_loss_train = 0.0
    
    model.train()
    for batch_idx, (images, labels) in enumerate(trainLoader):
        labels = labels.float()
        output = model(images).squeeze()
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
        batch_acc_train = torch.sum(labels.int() == (torch.sigmoid(output)>0.5).int())/len(output)
        print(f"Batch Loss train = {loss.item()}, batch Accuracy train = {batch_acc_train}")
        wandb.log({"loss_train": loss.item(),"Train Accuracy":batch_acc_train})
    print(f"Epoch = {epoch}, Epoch Loss train= {epoch_loss_train/len(trainLoader)}")
    
    epoch_loss_val = 0.0
    model.eval()
    for batch_idx, (images, labels) in enumerate(valLoader):
        labels = labels.float()
        with torch.no_grad():
            output = model(images).squeeze()
            loss = criterion(output,labels)
        epoch_loss_val+=loss.item()
        batch_acc_val = torch.sum(labels.int() == (torch.sigmoid(output)>0.5).int())/len(output)
        wandb.log({"loss_val": loss.item(),"Val Accuracy":batch_acc_val})
        print(f"Batch Loss train = {loss.item()}, batch Accuracy Val = {batch_acc_val}")
        pass
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{trainingCofig["modelsavepath"]}/checkpoint_epoch{epoch}.pth")

wandb.finish()