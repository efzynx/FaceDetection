import cv2
import numpy as np
from deepface import DeepFace


# Path to the image file
img_path = '/content/drive/MyDrive/saya.jpg'

# Load the image
img = cv2.imread(img_path, 1)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Check if the image is loaded properly
if img is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face from the image
            cropped_face = img[y:y+h, x:x+w]

            # Scale the cropped face
            scaled_face = cv2.resize(cropped_face, None, fx=2, fy=2)

# Display the original image with detected face
print("Original Image with Detected Face:")
cv2_imshow(img)

# Display the cropped face
print("Cropped Face:")
cv2_imshow(cropped_face)

# Predict the age using DeepFace
try:
# Analyze the cropped and scaled face
      obj = DeepFace.analyze(img_path=scaled_face, actions=['age'])
# verif = demography['age']
      print(f'Predicted Age: {obj}')
except Exception as e:
      print(f"Error in predicting age: {e}")