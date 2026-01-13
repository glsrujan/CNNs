import torch
from torchvision import transforms
from src.utils.gen import Circle,Rectangle
import matplotlib.pyplot as plt
import os
from src.core.backbone import BCmodel


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
def infer(model, data, gtLabel):
    data = transform(data).unsqueeze(0)
    with torch.no_grad():
        out = model(data)
        out = torch.sigmoid(out)
    print(out,gtLabel)
    return out
if __name__ == "__main__":
    model = BCmodel(3,16)
    state_dict = torch.load("models/cnn-classify-4/checkpoint_epoch45.pth")
    model.load_state_dict(state_dict["model_state_dict"])
    
    circle = Circle(baseDir="data/synthetic",type="Mixed")
    rectangle = Rectangle(baseDir="data/synthetic",type="Mixed")
    config = {}
    # config["bgColor"] = (0,0,0)
    # config["fgColor"] = (255,255,255)
    model.eval()
    
    for iter in range(50):
        circle.createShape(config)
        pred = infer(model,circle.image,0)
        rectangle.createShape(config)
        pred = infer(model,rectangle.image,1)
        plt.title(f"GT Label = {1}, Model Pred = {pred.item()}")
        plt.imshow(rectangle.image)
        plt.show()