import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Shapes:
    def __init__(self, imageH = 64, imageW = 64, imgformat = ".png",baseDir = "sythetic"):
        self.imageH = imageH
        self.imageW = imageW
        self.baseDir = Path(baseDir,self.__class__.__name__)
        self.baseDir.mkdir(parents=True,exist_ok=True)
        self.imgformat = imgformat
        self.idx = len(list(self.baseDir.glob("*"+self.imgformat)))
        pass
    
    def createShape(self):
        raise NotImplementedError
    
    def saveshape(self):
        cv2.imwrite((self.baseDir/(self.__class__.__name__+"_"+str(self.idx)+self.imgformat)).absolute(),self.image)
        self.idx+=1
        
class Circle(Shapes):
    def __init__(self, imageH=64, imageW=64, imgformat=".png", baseDir="sythetic", type="Uniform"):
        super().__init__(imageH, imageW, imgformat, baseDir)
        self.type = type
        pass
    
    def createShape(self,kwargs):
        if "bgColor" in kwargs.keys():
            self.bgColor = kwargs["bgColor"]
        else:
            self.bgColor = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        if "fgColor" in kwargs.keys():
            self.fgColor = kwargs["fgColor"]
        else:
            self.fgColor = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        self.image = np.ones((self.imageH, self.imageW,3),dtype=np.uint8)
        self.image[:,:,0]*=self.bgColor[0]
        self.image[:,:,1]*=self.bgColor[1]
        self.image[:,:,2]*=self.bgColor[2]
        
        self.cent = (np.random.randint(5,self.imageW),np.random.randint(5,self.imageH))
        r_max = min(self.cent[0], self.cent[1], self.imageW - self.cent[0], self.imageH - self.cent[1])

        self.radius = min(np.random.randint(5,max(self.imageH,self.imageW)),r_max)
        if self.type == "Mixed":
            if np.random.uniform(0,1)<0.5:
                cv2.circle(self.image,self.cent,self.radius,self.fgColor,np.random.randint(1,self.radius+1),cv2.FILLED)
            else:
                cv2.circle(self.image,self.cent,self.radius,self.fgColor,cv2.FILLED)
        elif self.type == "Uniform":
            cv2.circle(self.image,self.cent,self.radius,self.fgColor,cv2.FILLED)
    # def saveshape(self):
    #     cv2.imwrite((self.baseDir/(self.__class__.__name__+"_"+str(self.idx)+self.imgformat)).absolute(),self.image)
    #     self.idx+=1

class Rectangle(Shapes):
    def __init__(self, imageH=64, imageW=64, imgformat=".png", baseDir="sythetic", type="Uniform"):
        super().__init__(imageH, imageW, imgformat, baseDir)
        self.type = type
        pass
    
    def createShape(self, kwargs):
        if "bgColor" in kwargs.keys():
            self.bgColor = kwargs["bgColor"]
        else:
            self.bgColor = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        if "fgColor" in kwargs.keys():
            self.fgColor = kwargs["fgColor"]
        else:
            self.fgColor = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        self.image = np.ones((self.imageH, self.imageW,3),dtype=np.uint8)
        self.image[:,:,0]*=self.bgColor[0]
        self.image[:,:,1]*=self.bgColor[1]
        self.image[:,:,2]*=self.bgColor[2]
        
        x1 = np.random.randint(0, self.imageW - 1)
        y1 = np.random.randint(0, self.imageH - 1)

        x2 = np.random.randint(x1 + 1, self.imageW)
        y2 = np.random.randint(y1 + 1, self.imageH)

        self.pt1 = (x1, y1)
        self.pt2 = (x2, y2)
        if self.type == "Mixed":
            if np.random.uniform(0,1)<0.5:
                cv2.rectangle(self.image,self.pt1,self.pt2,np.random.randint(1,min(self.pt2[0],self.pt2[1])+1),cv2.FILLED)
            else:
                cv2.rectangle(self.image,self.pt1,self.pt2,self.fgColor,cv2.FILLED)
        elif self.type == "Uniform":
            cv2.rectangle(self.image,self.pt1,self.pt2,self.fgColor,cv2.FILLED)
    

        
if __name__ == "__main__":
    circle = Circle(baseDir="data/synthetic")
    rectangle = Rectangle(baseDir="data/synthetic")
    config = {}
    # config["bgColor"] = (0,0,0)
    # config["fgColor"] = (255,255,255)
    
    for iter in range(10000):
        circle.createShape(config)
        circle.saveshape()
        rectangle.createShape(config)
        rectangle.saveshape()
    # baseclass.saveshape()
    # plt.imshow(circle.image)
    # plt.show()
    # print(baseclass.baseDir)