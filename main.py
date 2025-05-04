import numpy as np
import json 
import time
# time_import = time.time()
# print("Starting AprilTag import...")
from pupil_apriltags import Detector
# print("AprilTag import successful.")
# print(f"Time taken to import AprilTag: {time.time() - time_import} seconds")
from monumental import estimate_tag_positions_3d, visualize_tag_positions, save_tag_positions, visualize_tags_3d, visualize_tag_positions_old

video_path = "plantage_shed.mp4"

with open('cam.json', 'r') as camera:
    camera_intrinsics = json.load(camera)

fx = camera_intrinsics["fx"]
fy = camera_intrinsics["fy"]
cx = camera_intrinsics["px"]
cy = camera_intrinsics["py"]
dist_coeffs = np.array(camera_intrinsics["dist_coeffs"], dtype=np.float32)

camera_matrix = np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=np.float32)

tag_size = 42. # in mm

print("Loading Detector...")
time_detector = time.time()
detector = Detector(
        families='tagStandard52h13',
        nthreads=4,
        )
print(f"Time taken to load detector: {time.time() - time_detector} seconds")

# Initialize AprilTag detector
# detector = Detector(
#     families='tagStandard52h13',  # Tag family to use
#     nthreads=4,           # Number of threads
#     quad_decimate=2.0,    # Image decimation factor
#     quad_sigma=0.0,       # Gaussian blur sigma
#     refine_edges=1,       # Refine edges of detected quads
#     decode_sharpening=0.25,  # Sharpening factor
#     debug=0               # Debug level
# )

constraints = [
    (2, 3, 1090),    # Distance between tags 2 and 3 is 1090mm
    (3, 39, 1940)    # Distance between tags 3 and 39 is 1940mm
]

def main():
    tag_positions, all_observations, reference_tag_id = estimate_tag_positions_3d(video_path, 
                                                                                  detector, 
                                                                                  camera_matrix, 
                                                                                  dist_coeffs, 
                                                                                  tag_size,
                                                                                  known_constraints=constraints) 

    if tag_positions:
        # Save results to JSON file
        save_tag_positions(tag_positions, "AprilTag_coordinates.json")
        
        # Visualize tags in 3D
        visualize_tags_3d(tag_positions, reference_tag_id)
        # visualize_tag_positions(video_path, all_observations, tag_positions, camera_matrix, dist_coeffs)
        visualize_tag_positions_old(video_path, detector, tag_positions, camera_matrix, dist_coeffs)
    print("All done! Tag positions have been estimated and saved.")

if __name__ == "__main__":
    main()