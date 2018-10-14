# Camera
Real-time face and emotion recognition via raspi camera

## About
This project uses OpenCV to detect faces and determines their emotions using the models under `models/`.  Frames are streamed from a Raspberry Pi camera continuously, allowing emotions to be determined in real-time.  Once emotions are determined, output in the form of the mode of the last 10 emotions is written to `~/.emotion`, to be read by [TheHuskiteers/midi-generator](https://github.com/TheHuskiteers/midi-generator)

## Requirements
Install requirements via `apt` on raspbian stretch:

`sudo apt install python-opencv python-numpy python-picamera`

## Running
Make sure X11 is running (via `startx` or similar), then run `./main.py`