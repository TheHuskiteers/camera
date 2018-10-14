#!/usr/bin/env python2.7

import time
import os
import cv2
import numpy as np

from picamera.array import PiRGBArray
from picamera import PiCamera

# Define font to use for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Init face_cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Init the fisherface model
fisher_face = cv2.createFisherFaceRecognizer()
fisher_face.load('models/emotion_detection_model.xml')

# List of emotions current model is trained for
emotions = ['neutral', 'angry', 'disgusted', 'happy', 'sad', 'surprised']

def main():
    # Setup camera
    (camera, cap) = setup_camera()

    # Count and list for determining mode of recent emotions
    face_count = 0
    recent_emotions = []

    # Write initial neutral emotion to `~/.emotion`
    last_emotion = 'neutral'
    write_emotion(last_emotion)

    # Capture frames from camera
    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
        # Grab raw array representing the image
        image = frame.array

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find faces in image
        faces = find_faces(image)

        # Take the first face and draw a square around it
        if len(faces) != 0:
            # Increment number of faces found
            face_count += 1

            # Draw a green rectangle around first face found
            (x, y, w, h) = faces[0]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Normalize faces found
            normalized_faces = normalize_faces(gray, faces)

            # Get emotion from first face found
            emotion = get_emotion_from_face(normalized_faces[0])

            # Append that emotion to recent_emotions
            recent_emotions.append(emotion)

            # Determine if average emotion has changed over 10-face interval
            if face_count >= 10:
                # Get mode of recent_emotions
                mode = get_mode_emotion(recent_emotions)

                # Detect if mode has changed
                if last_emotion != mode:
                    last_emotion = mode
                    write_emotion(last_emotion)
                
                # Reset recent_emotions and face_count
                recent_emotions = []
                face_count = 0

        # Write last_emotion to bottom left of frame
        cv2.putText(image, last_emotion, (5, 220), font, 1, (0, 255, 0), 2)

        # Make sure frame is full-screen
        image = cv2.resize(image, (576, 416))

        # Display frame
        cv2.imshow("Frame", image)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break
        
        # Clear stream in prep for next frame
        cap.truncate(0)

    # Cleanup windows when done
    cv2.destroyAllWindows()

# Setup camera
def setup_camera():
    res = (320, 224)
    camera = PiCamera()
    camera.resolution = res
    camera.framerate = 32
    cap = PiRGBArray(camera, size=res)
    time.sleep(0.1)

    return (camera, cap)

# Find all the faces in an image
def find_faces(image):
    return face_cascade.detectMultiScale(image, 1.3, 3)

# Normalize faces (cut them out and resize)
def normalize_faces(image, faces):
    # Cut faces from image
    cut_faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

    # Resize
    normalized_faces = [cv2.resize(f, (350, 350)) for f in cut_faces]

    return normalized_faces

# Return prediction of face
def get_emotion_from_face(face):
    return emotions[fisher_face.predict(face)[0]]

# Return the most common emotion in recent_emotions
def get_mode_emotion(recent_emotions):
    return max(recent_emotions, key=recent_emotions.count)

# Write given emotion to `~/.emotion`
def write_emotion(emotion):
    f = open(os.path.expanduser('~/.emotion'), 'w')
    f.write(emotion)
    f.close()

# Execute main() if executed
if __name__ == "__main__":
    main()
