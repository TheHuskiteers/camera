import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Setup camera
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# Let camera warmup
time.sleep(0.1)

# Capture image from camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Init face_cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# largest_face = (0, 0, 0, 0)
# for (x, y, w, h) in faces:
#     if w * h > largest_face[2] * largest_face[3]:
#         largest_face = (x, y, w, h)

# Take the first face and draw a square around it
(x, y, w, h) = faces[0]
cv2.rectangle(image, (x, y),(x+w, y+h), (0, 255, 0), 2)

# Output image to file
cv2.imwrite("output_faces.jpg", image)