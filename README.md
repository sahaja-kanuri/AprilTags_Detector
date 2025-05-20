# AprilTags Detector

## Description

This codebase estimates the 3D positions of AprilTags in the video sequence, `plantage_shed.mp4`. Each AprilTag in the video has a unique ID. It leverages multi-view geometry and optimization techniques to produce accurate spatial coordinates of tags, given real-world constraints that the AprilTags have a side length of 42mm, and there are 3 markers in an L-shape, with tag IDs and the distances between their respective centers:

```
2
^
|
| 1090mm
|
v     1940mm
3  <----------> 39
```

The Tag positions get saved in the `apriltag_coordinates.json` file.

## Setup

<pre><code>
$ python3 -m venv aprilTags
$ source aprilTags/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt

$ python3 main.py
</code></pre>