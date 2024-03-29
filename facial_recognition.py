#!/usr/bin/python

#import the necessary packages
import imp
from imutils.video import VideoStream
from imutils.video import FPS 
import face_recognition
import imutils 
import pickle
import time 
import cv2
import numpy as np
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

factory =  PiGPIOFactory()
servo_x = Servo(17, pin_factory=factory)
servo_y = Servo(18, pin_factory=factory)

def return_valid_value(position):
    if position > 1:
        position = 1
    elif position < -1:
        position = -1
    return position

# Initialize currentname to trigger only when a new person is identified... currentname "unknown"
currentname = "Unknown"

# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...") 
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up
# Set the src to the following
# src = 8: for the built-in single webcam, could be your laptop webcam
# src = 2: I had to set it to 2 in order to use the USB webcam attached to my laptop
vs = VideoStream(usePiCamera=True, framerate=10).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()
x_position = 0.3 
servo_x.value = x_position

# PID control parameters
Kp = 0.1  # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.001  # Derivative gain
prev_error = 0
integral = 0

# Loop over frames from the video file stream
while True:
    # Grab the frame from the threaded video stream and resize it to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    height, width, _ = frame.shape
    y_medium = int(height/2) 
    x_medium = int(width/2)
    center = int(width/2)

    # Detect the face boxes
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # If face is not recognized, then print Unknown

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie, Python
            # will select the first entry in the dictionary)
            name = max(counts, key=counts.get)

            # If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)
        
        # Update the list of names
        names.append(name)

    # Loop over the recognized faces
    for ((x, y, w, h), name) in zip(boxes, names):
        # Draw the predicted face name on the image (color is in BGR)
        cv2.rectangle(frame, (h, x), (y, w), (0, 255, 225), 2)
        x_medium = int((x + x + w) / 2)
        z = x - 15 if x - 15 > 15 else x + 15
        cv2.putText(frame, name, (h, z), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        break

    # Compute the error for PID control
    error = x_medium - center

    # Compute PID components
    proportional = error
    integral += error
    derivative = error - prev_error

    # Compute the new servo position using PID control
    x_position += Kp * proportional + Ki * integral + Kd * derivative

    # Limit the servo position to [-1, 1]
    x_position = return_valid_value(x_position)

    # Update the servo position
    servo_x.value = x_position

    # Display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when ‘q’ key is pressed 
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed())) 
print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) 

# Do a bit of cleanup
cv2.destroyAllWindows() 
vs.stop()
