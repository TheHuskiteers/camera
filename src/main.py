import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def main():
    # Setup camera
    camera = PiCamera()
    camera.resolution = (320, 220)
    camera.framerate = 32
    cap = PiRGBArray(camera, size=(320, 220))
    time.sleep(0.1)

    # Init face_cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Define font to use for text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Capture frames from camera
    for frame in camera.capture_continuous(cap, format='bgr', use_video_port=True):
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
            cv2.putText(image, 'Face Detected', (5, 220), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Face Detected', (5, 220), font, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Clear stream in prep for next frame
        cap.truncate(0)

    # Cleanup windows when done
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
