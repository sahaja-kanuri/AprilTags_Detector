import cv2
import numpy as np
from pupil_apriltags import Detector
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # from github
# from apriltag import apriltag
# imagepath = 'test.jpg'
# image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# detector = apriltag("tagStandard52h13")
# detections = detector.detect(image)

# # or
# # from pyimagesearch
# import apriltag
import argparse
# .yaml 

# ap = argparse.,ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image containing AprilTag")
# args = vars(ap.parse_args())
# print("[INFO] loading image...")
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print("[INFO] detecting AprilTags...")
# options = apriltag.DetectorOptions(families="tagStandard52h13")
# detector = apriltag.Detector(options)
# results = detector.detect(gray)
# print("[INFO] {} total AprilTags detected".format(len(results))) # total number of april tags detected

with open('cam.json', 'r') as camera:
    data = json.load(camera)
# Extract camera parameters
fx = data['fx']
fy = data['fy']
px = data['px']
py = data['py']

# Create detector
detector = Detector(families='tagStandard52h13')

# Open the video file
video_capture = cv2.VideoCapture('plantage_shed.mp4')

# Check if the video opened successfully
if not video_capture.isOpened():
    print("Error opening video file")
    exit()

# Get frame dimensions
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get total frame count
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Get frames per second (FPS)
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame = 0
# Read frames until the video ends
while video_capture.isOpened():
    frame += 1
    # Capture frame-by-frame
    ret, img = video_capture.read()

    # If frame is read correctly ret is True
    if not ret:
        print("End of video or error reading frame")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect tags
    results = detector.detect(gray_img, estimate_tag_pose=True, camera_params=(fx, fy, px, py), tag_size=0.042) # 42mm

    # detect(img: ndarray[Any, dtype[uint8]], estimate_tag_pose: bool = False, camera_params: Optional[Tuple[float, float, float, float]] = None, tag_size: Optional[float] = None) → Detection
    # If you also want to extract the tag pose, estimate_tag_pose should be set to True and camera_params ([fx, fy, cx, cy]) and tag_size (in meters) should be supplied.
    for result in results:
        tag_id = result.tag_id
        center_coord = result.center
        corner_coords = result.corners







# # Read frames until the video ends
# while video_capture.isOpened():
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
    
#     # If frame is read correctly ret is True
#     if not ret:
#         print("End of video or error reading frame")
#         break
    
#     # Process the frame here
#     # For example, convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Display the resulting frame
#     cv2.imshow('Frame', gray_frame)
    
#     # Press Q on keyboard to exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close windows
# video_capture.release()
# cv2.destroyAllWindows()