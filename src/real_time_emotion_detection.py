import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("emotion_model.keras")

emotion_labels = {
    0:"Angry",1:"Disgust",2:"Fear",
    3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"
}

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
   
    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face,(96,96))
        face = face/255.0
        face = np.reshape(face,(1,96,96,3))

        pred = model.predict(face)
        
        label = emotion_labels[np.argmax(pred)]
       

        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),
                      (0,255,0),2)

    cv2.imshow("Emotion Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()