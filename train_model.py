import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os

# Init face_cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# List of emotions to train the model for
emotions = ['neutral', 'happy', 'sad']

# Where dataset is located
dataset_dir = './dataset/'

def main():
    take_pictures()

def take_pictures():
    # Setup camera
    (camera, cap) = setup_camera()

    # Number of pictures to take for each emotion
    number_of_each = 40

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for emotion in emotions:
        pictures_left = number_of_each
        base_path = dataset_dir + emotion
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        while pictures_left != 0:
            camera.capture(cap, format="bgr")
            image = cap.array
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = find_faces(image)
            if len(faces) != 0:
                normalized_faces = normalize_faces(gray, faces)
                cv2.imshow("Image", normalized_faces[0])
                cv2.waitKey(0)
                cap.truncate(0)

# Setup camera
def setup_camera():
    camera = PiCamera()
    camera.resolution = (320, 224)
    camera.framerate = 32
    cap = PiRGBArray(camera, size=(320, 224))
    time.sleep(0.1)

    return (camera, cap)

def find_faces(image):
    # Detect faces in image
    faces = face_cascade.detectMultiScale(image, 1.3, 3)

    # Return list of faces
    return faces

def normalize_faces(image, faces):
    # Cut faces from image
    cutted_faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

    # Resize
    normalized_faces = [cv2.resize(f, (350, 350)) for f in cutted_faces]

    return normalized_faces

# Execute main() if run
if __name__ == "__main__":
    main()
