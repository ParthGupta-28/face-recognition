import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pymongo import MongoClient
from torchvision import transforms
from PIL import Image
from model.model_class import FaceRecogModel

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["faceRecog"]
collection = db["faceData"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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