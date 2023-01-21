import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("model.h5")

face_label = {
    0: "not admin",
    1: "admin"
}

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX


while True:
    
    ret, frame = cap.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    border = cascade.detectMultiScale(gray)
    
    frame2 = cv2.resize(frame2, (40,40))   
    frame2 = frame2/255
    frame2 = frame2[np.newaxis,:]
    
    y_pred = model.predict(frame2)
    y_pred = np.round(y_pred)
    
    
    for (x, y, w, h) in border:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv2.putText(frame, face_label[y_pred[0][0]], (x,y), font, 1, (0,0,255), 2)
        break
    else:
        cv2.putText(frame, "noface", (frame.shape[0]//2,frame.shape[0]//2), font, 1, (0,0,255), 2)
        
    cv2.imshow("frame", frame)
    
    k = cv2.waitKey(1)
    
    if k%256==27:
        break
        
cap.release()
cv2.destroyAllWindows()
