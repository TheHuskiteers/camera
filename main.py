import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from multiprocessing import Pool

# Define font to use for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Init face_cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Init the fisherface model
fisher_face = cv2.createFisherFaceRecognizer()
fisher_face.load('models/emotion_detection_model.xml')

# List of emotions current model is trained for
emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']

def main():
    # Setup camera
    (camera, cap) = setup_camera()

    # Capture frames from camera
    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
        # Grab raw array representing the image
        image = frame.array

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find faces in image
        faces = find_faces(image)

        # Normalize faces found
        normalized_faces = normalize_faces(gray, faces)

        # Take the first face and draw a square around it
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            emotion = get_emotion_from_face(normalized_faces[0])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, emotion, (5, 220), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Face Detected', (5, 220), font, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Frame", cv2.resize(image, (576, 416)))

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break
        
        # Clear stream in prep for next frame
        cap.truncate(0)

    # Cleanup windows when done
    cv2.destroyAllWindows()

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
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    # Return list of faces
    return faces

def normalize_faces(image, faces):
    # Cut faces from image
    cutted_faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

    # Resize
    normalized_faces = [cv2.resize(f, (350, 350)) for f in cutted_faces]

    return normalized_faces

def get_emotion_from_face(face):
    return emotions[fisher_face.predict(face)[0]]

# Execute main() if run
if __name__ == "__main__":
    main()
