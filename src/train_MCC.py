from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from core.backbone import Resnet4Classifier, BCmodel
from torch import nn, optim
import torch.nn.functional as F
import torch
import wandb
import yaml
from pathlib import Path

# Training Config
with open("config/resnet-classify-2.yaml", "r") as f:
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

print(dataset.classes)

# manual see to restart training from last checkpoint. (If not the model would mix training and val datasets everytime the training restarts)
datasetGenerator = torch.Generator().manual_seed(trainingCofig["seed"])
trainData, valData = random_split(dataset,[trainSize,valSize],generator=datasetGenerator)

trainLoader = DataLoader(trainData,batch_size=trainingCofig["batchsize"],shuffle=True)
valLoader = DataLoader(valData,batch_size=trainingCofig["batchsize"],shuffle=False)

# wandb stuff
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="dammu",
    # Set the wandb project where this run will be logged.
    project="resnet-classify-3",
    id="n7cpsksz",
    resume="must",   # or "allow"
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": float(trainingCofig["lr"]),
        "architecture": "RESNET4",
        "dataset": trainingCofig["dataset"],
        "epochs": trainingCofig["epochs"],
    },
)

model = Resnet4Classifier(3,16,len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr = float(trainingCofig["lr"]))

if "loadModel" in trainingCofig.keys() and trainingCofig["loadModel"]:
    state_dict = torch.load(trainingCofig["loadModel"])
    model.load_state_dict(state_dict)
    
modelDir = Path(trainingCofig["modelsavepath"])
modelDir.mkdir(parents=True,exist_ok=True)
# #if dir already has model checkpoints , load the last model and continue retraining
modelChkPaths = sorted(
    modelDir.glob("checkpoint_epoch*.pth"),
    key=lambda p: int(p.stem.split("epoch")[-1])
)

if len(modelChkPaths):
    modelChkPts = torch.load(modelChkPaths[-1])
    model.load_state_dict(modelChkPts["model_state_dict"])
    optimizer.load_state_dict(modelChkPts["optimizer_state_dict"])

for epoch in range(len(modelChkPaths),len(modelChkPaths)+trainingCofig["epochs"]):
    epoch_loss_train = []
    epoch_acc_train = []
    model.train()
    for batch_idx, (images, labels) in enumerate(trainLoader):
        labels = labels.long()
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_train.append(loss.item())
        preds = output.argmax(dim=1)     # (N,)
        epoch_acc_train.append((preds == labels).float().mean())
    epoch_loss_train=torch.tensor(epoch_loss_train).mean()
    epoch_acc_train=torch.tensor(epoch_acc_train).mean()
    
    wandb.log({"Loss(Train)":epoch_loss_train, "Accuracy(Train)":epoch_acc_train})
    print(f"Loss(Train):{epoch_loss_train: .4f} | Accuracy(Train):{epoch_acc_train: .4f}")
    
    epoch_loss_val = []
    epoch_acc_val = []
    model.eval()
    for batch_idx, (images, labels) in enumerate(valLoader):
        labels = labels.long()
        with torch.no_grad():
            output = model(images)
            loss = criterion(output,labels)
        epoch_loss_val.append(loss.item())
        preds = output.argmax(dim=1)     # (N,)
        epoch_acc_val.append((preds == labels).float().mean())
        pass
    epoch_loss_val=torch.tensor(epoch_loss_val).mean()
    epoch_acc_val=torch.tensor(epoch_acc_val).mean()
    
    wandb.log({"Loss(Val)":epoch_loss_val, "Accuracy(Val)":epoch_acc_val})
    print(f"Loss(Val):{epoch_loss_val: .4f} | Accuracy(Val):{epoch_acc_val: .4f}")
    
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{trainingCofig["modelsavepath"]}/checkpoint_epoch{epoch}.pth")

wandb.finish()