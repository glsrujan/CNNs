import torch
from torchvision import transforms
from utils.gen import Circle,Rectangle,Polygon
import matplotlib.pyplot as plt
import os
from core.backbone import BCmodel,Resnet4Classifier
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
def infer(model, data, gtLabel):
    data = transform(data).unsqueeze(0)
    with torch.no_grad():
        out = model(data)
        out = torch.softmax(out,dim=1)
    # print(out,gtLabel)
    return out

if __name__ == "__main__":
    model = Resnet4Classifier(3,16,4)
    state_dict = torch.load("models/resnet-classify-3/checkpoint_epoch79.pth")
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    circle = Circle(128,128,baseDir="data/synthetic_2",dirName="circle")
    rectangle = Rectangle(128,128,baseDir="data/synthetic_2",dirName="rectangle")
    triangle = Polygon(128,128,vert=3,baseDir="data/synthetic_2",dirName="triangle")
    pentagon = Polygon(128,128,vert=5,baseDir="data/synthetic_2",dirName="pentagon")
    
    config = {}
    # config["bgColor"] = (0,0,0)
    # config["fgColor"] = (255,255,255)
    iters = 10000
    correctPreds = 0
    for iter in range(iters):
        circle.createShape(config)
        rectangle.createShape(config)
        triangle.createShape(config)
        pentagon.createShape(config)
        gtlabel = np.random.randint(0,4)
        if(gtlabel==0):
            img = circle.image
        elif (gtlabel == 1):
            img = pentagon.image
        elif (gtlabel == 2):
            img = rectangle.image
        else:
            img = triangle.image
        pred = infer(model,img,gtlabel)
        predLabel = torch.argmax(pred)
        if gtlabel == predLabel:
            correctPreds+=1
        else:
            plt.title(f"GT Label = {gtlabel}, Predicted Lable = {predLabel}")
            plt.imshow(img)
            plt.show()
            
    print(f"Accuracy is {correctPreds/iters}")