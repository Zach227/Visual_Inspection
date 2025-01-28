# """
# ECEn-631 Visual Inspection Project created by Harrison Garrett in 2020

# """

# import cv2 as cv
# import numpy as np
# '''
# Set WEBCAM to 1 to use your webcam or 0 to use the Flea2 cameras on the lab machine
# Set CATCHER to 1 to use the catcher connected to the lab machine or 0 to use your own computer
# '''
# WEBCAM = 1
# CATCHER = 0
# i = 0
# if WEBCAM:
#     camera = cv.VideoCapture(0)
# else:
#     from src.Flea2Camera2 import FleaCam
#     camera = FleaCam()

# width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
# videoout = cv.VideoWriter('./sample_video.avi', cv.VideoWriter_fourcc(*'MJPG'), 25, (width, height))


# while True:
#     # Get Opencv Frame
#     if WEBCAM:
#         ret0, frame = camera.read()
#     else:
#         frame = camera.getFrame()

#     # print(cv_image.shape)
#     cv.imshow('frame',frame)
#     # tempName = 
#     # cv.imwrite(f"img{i}.png", frame)
#     # break
#     i += 1

#     # Press Q on keyboard to  exit
#     if cv.waitKey(40) & 0xFF == ord('q'):
#       break

#     videoout.write(frame)
    
# camera.release()
# videoout.release()

# cv.destroyAllWindows()

import cv2 as cv
import numpy as np
import subprocess

WEBCAM = 1  # Set to 1 for webcam
i = 0

# Initialize camera
camera = cv.VideoCapture(0) if WEBCAM else None
if not camera.isOpened():
    print("Error: Camera not opened.")
    exit()

# Get frame dimensions
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = 25

# Initialize video writer
videoout = cv.VideoWriter('./sample_video_4.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

#Camera Settings
bash_script = "./SVM_Copy/camera_settings.sh"
try:
    result = subprocess.run(
        ['bash', './SVM_Copy/camera_settings.sh'],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Success:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
    
while True:
    # Capture frame
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame")
        break

    # Display frame
    cv.imshow('frame', frame)

    # Write frame to video
    videoout.write(frame)

    # Exit on 'q' key
    if cv.waitKey(40) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
videoout.release()
cv.destroyAllWindows()
