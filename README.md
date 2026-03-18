#  Real-Time Emotion Detection using Deep Learning

##  Project Overview

This project detects "human emotions in real-time" using a webcam.
It uses "deep learning" for emotion classification and "face detection" to locate faces in images.

The system can recognize multiple emotions such as:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

---

##  Model Architecture

* MobileNetV2 (Transfer Learning)
* Trained on facial expression dataset
* Output layer with 7 emotion classes (Softmax)

---

##  Project Structure

```
Real-Time-Emotion-Detection/
│
├── assets/
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   └── emotion_model.keras
│
├── src/
│   └── real_time_emotion_detection.py
│
├── .gitattributes
├── README.md
└── requirements.txt
```
---

## Model File

The trained model (`emotion_model.keras`) is stored using Git LFS due to its large size.

To download:
Click on the file → Click "View Raw"

---

##  Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

##  Dataset

* FER2013 (Facial Expression Recognition Dataset)
* Contains grayscale facial images of size 48×48
* 7 emotion classes

---

##  Features

✔ Real-time emotion detection using webcam
✔ Face detection using Haarcascade
✔ Deep learning-based classification
✔ Lightweight and fast execution

---

##  How It Works

1. Webcam captures live video
2. Face is detected using Haarcascade
3. Detected face is preprocessed
4. Model predicts emotion
5. Emotion is displayed on screen

---

##  Future Improvements

* Deploy as a web app using Streamlit
* Add face recognition + emotion tracking
* Improve performance for low-light conditions