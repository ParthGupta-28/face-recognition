import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pymongo import MongoClient
from torchvision import transforms
from PIL import Image

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["faceRecog"]
collection = db["faceData"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceRecogModel(nn.Module):
    def __init__(self):
        super(FaceRecogModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.linear_dims = None
        x = torch.randn(1, 3, 128, 128)
        self.convs(x)
        self.fc1 = nn.Linear(self.linear_dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()

    def convs(self ,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.linear_dims is None:
            self.linear_dims = x.shape[1]*x.shape[2]*x.shape[3]
            print(self.linear_dims)
        return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = torch.load('model_weights.pt')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def saveFace(name, face):
    imgarr = Image.fromarray(face)
    tensor_image = transform(imgarr).unsqueeze(0).cuda()
    tensor_image.to(device)
    output = model(tensor_image).cpu().detach().numpy()
    list_output = output.tolist()
    dto = {
        "name": name,
        "face_vector": list_output[0]
    }

    collection.insert_one(dto)

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX
text = ""
name = input("Enter your name: ")

while True:
    
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    border = cascade.detectMultiScale(gray)

    if len(border) == 0:
        text = "No Face Detected"
     
    if len(border) > 1:
        text = "Multiple Faces Detected"
    
    for (x, y, w, h) in border:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        ROI = frame[y:y+h, x:x+w]
    
    if text != "":
        cv2.putText(frame, text, (frame.shape[0]//2,frame.shape[1]//2), font, 1, (0,0,255), 2)

    cv2.imshow("frame", frame)
    text = ""
    k = cv2.waitKey(1)
    if k%256==27:
        break

    if k%256==32:
        if len(border) == 1:
            saveFace(name, ROI)
        break
        
cap.release()
cv2.destroyAllWindows()