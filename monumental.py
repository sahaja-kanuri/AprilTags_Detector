
import cv2
import numpy as np


# from github
from apriltag import apriltag
imagepath = 'test.jpg'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tagStandard52h13")
detections = detector.detect(image)

# or
# from pyimagesearch
import apriltag
import argparse

ap = argparse.,ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing AprilTag")
args = vars(ap.parse_args())
print("[INFO] loading image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(families="tagStandard52h13")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(results))) # total number of april tags detected

# Open the video file
video_capture = cv2.VideoCapture('plantage_shed.mp4')
