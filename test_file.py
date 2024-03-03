import cv2
import numpy as np
import torch
from torchvision import models
from keras.models import load_model
import torch.nn as nn
import os

data_dir='./face_recognition'
img_size=[48,48]
batch_size=128

num_class=len(os.listdir(data_dir))
classes= os.listdir(data_dir)

# Load the trained face detection model
model1_weights = torch.load('resnet_face_detection_model.pth', weights_only=True)
print(type(model1_weights))
model1 = models.resnet18()
num_ftrs = model1.fc.in_features  #num_ftrs=512
model1.fc=nn.Sequential(
    nn.Linear(num_ftrs, num_class),
    nn.Softmax(dim=1)
) #modifying fully connected layer to classify num_classes
model1.load_state_dict(torch.load('resnet_face_detection_model.pth')) # Load your trained face detection model here
model2 = load_model('vgg16emotion_detection_model.h5')  # Load your trained emotion detection model here

# Define the emotion labels
face_labels = ['Al-Faiz_Ali', 'Aryan', 'divyanshu', 'jayanta', 'Ranveer', 'Riktom', 'sai_dushwanth', 'Tejas', 'unknown']
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Preprocess the frame (resize and convert to grayscale)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    faces_f = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    # Loop through the detected faces and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the face

        # Crop the face region from the frame and preprocess it for emotion detection
        face_region = frame[y:y + h, x:x + w]
        face_region = cv2.resize(face_region, (48, 48))
        face_region = np.expand_dims(face_region, axis=0)  # Add batch dimension
        face_region = face_region / 255.0  # Normalize the face region

        face_region_f = frame[y:y + h, x:x + w]
        face_region_f = cv2.resize(face_region_f, (48, 48))
        face_region_f = np.expand_dims(face_region_f, axis=0)  # Add batch dimension
        face_region_f = face_region_f / 255.0  # Normalize the face region

        # Make predictions using the emotion detection model
        # print("Type of model1, model2", type(model1), type(model2))
        face_region_f= torch.from_numpy(face_region_f)
        face_region_f= face_region_f.permute(0,3,1,2)
        # print("face_region_f shape= ", face_region_f.shape)
        predictions1=model1(face_region_f.float())
        predictions2 = model2.predict(face_region)
        predicted_emotion = emotion_labels[np.argmax(predictions2)]
        predictions1= predictions1.detach().numpy()
        predicted_face=face_labels[np.argmax(predictions1)]

        # Display the emotion text on the frame
        cv2.putText(frame, f'Name:{predicted_face} is {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame with face detection and emotion prediction
    cv2.imshow('Face Detection and Emotion Recognition', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
