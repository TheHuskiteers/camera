import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import process

# Setup camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Let camera warmup
time.sleep(0.1)

# Init face_cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture frames from camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    # Grab raw array representing the image
    image = frame.array

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Take the first face and draw a square around it
    if len(faces) != 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(image, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Frame", image)

    # Clear the stream in prep for next frame
    rawCapture.truncate(0)