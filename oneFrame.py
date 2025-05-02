import cv2
import numpy as np
from pupil_apriltags import Detector
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('cam.json', 'r') as camera:
    camera_intrinsics = json.load(camera)
# Extract camera parameters
# fx = data['fx']
# fy = data['fy']
# px = data['px']
# py = data['py']

# Create detector
# detector = Detector(families='tagStandard52h13')

# Initialize AprilTag detector
# detector = Detector(
#     families='tagStandard52h13',  # Tag family to use
#     nthreads=1,           # Number of threads
#     quad_decimate=1.0,    # Image decimation factor
#     quad_sigma=0.0,       # Gaussian blur sigma
#     refine_edges=1,       # Refine edges of detected quads
#     decode_sharpening=0.25,  # Sharpening factor
#     debug=0               # Debug level
# )

# Open the video file
# video_capture = cv2.VideoCapture('plantage_shed.mp4')


def detect_apriltags_in_frame(video_path, frame_index, camera_params, tag_size):
    """
    Detect AprilTags in a specific frame of a video and calculate their 3D coordinates
    
    Args:
        video_path (str): Path to the input video file
        frame_index (int): Index of the frame to process
        camera_params (dict): Dictionary containing camera intrinsic parameters
        tag_size (float): Size of the AprilTag in meters
        
    Returns:
        dict: Dictionary containing the detected tags with their IDs and 3D coordinates
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
        
    # Get total frame count
    # total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # Validate frame index
    # if frame_index >= total_frames:
    #     print(f"Error: Frame index {frame_index} exceeds video length ({total_frames} frames)")
    #     video_capture.release()
    #     return None
    
    # Set the video to the desired frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = video_capture.read()
    
    # Release the video capture object
    video_capture.release()
    
    if not ret:
        print(f"Error: Failed to read frame {frame_index}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract camera parameters
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["px"]
    cy = camera_params["py"]
    
    # Get distortion coefficients
    dist_coeffs = np.array(camera_params["dist_coeffs"]) #[:8]  # 8 coefficients are provided
    
    # Create camera matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Undistort the image using camera parameters
    undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)
    
    # Initialize AprilTag detector
    detector = Detector(
    families='tagStandard52h13',  # Tag family to use
    nthreads=1,           # Number of threads
    quad_decimate=1.0,    # Image decimation factor
    quad_sigma=0.0,       # Gaussian blur sigma
    refine_edges=1,       # Refine edges of detected quads
    decode_sharpening=0.25,  # Sharpening factor
    debug=0               # Debug level
)
    # Detect AprilTags - use undistorted image for better accuracy
    detections = detector.detect(undistorted, estimate_tag_pose=True, 
                                camera_params=(fx, fy, cx, cy), tag_size=tag_size)
    
    print(f"Detected {len(detections)} tags in frame {frame_index}")
    
    # Create a dictionary to store the results
    result = {
        "frame_index": frame_index,
        "tags": []
    }
    
    # Draw detections on the frame for visualization
    visual_frame = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
    
    # Process each detection
    for detection in detections:
        tag_id = detection.tag_id
        center = detection.center
        corners = detection.corners
        
        # Extract pose information
        pose_R = detection.pose_R  # 3x3 rotation matrix
        pose_t = detection.pose_t  # 3x1 translation vector
        
        print(f"Tag ID: {tag_id}, Center: {center}, Corners: {corners}")
        print(f"\n")
        print(f"Pose R: {pose_R}, Pose t: {pose_t}")
        # Convert to more usable format - 3D coordinates in camera frame
        x = pose_t[0][0]  # X coordinate
        y = pose_t[1][0]  # Y coordinate
        z = pose_t[2][0]  # Z coordinate (distance from camera)
        print(f"X: {x}, Y: {y}, Z: {z}")
        
        # Store tag information
        tag_info = {
            "id": int(tag_id),
            "position": {
                "x": float(x),
                "y": float(y),
                "z": float(z)
            },
            "pixel_center": {
                "x": float(center[0]),
                "y": float(center[1])
            },
            "corners": corners.tolist()  # Save corner points for reference
        }
        
        result["tags"].append(tag_info)
        
        # Draw on image for visualization
        cv2.polylines(visual_frame, [np.int32(corners)], True, (0, 255, 0), 2)
        cv2.circle(visual_frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        cv2.putText(visual_frame, f"ID: {tag_id}", (int(center[0]) + 10, int(center[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(visual_frame, f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}", 
                   (int(center[0]) + 10, int(center[1]) + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the visualization image
    cv2.imwrite("apriltag_detection.jpg", visual_frame)
    print("Visualization saved as 'apriltag_detection.jpg'")
    
    return result

def visualize_tags_3d(tags_data):
    """
    Create a 3D visualization of detected AprilTags
    
    Args:
        tags_data (dict): Dictionary containing tag information
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different tags
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Lists to store all coordinates for axis scaling
    xs = []
    ys = []
    zs = []

    # Plot each tag
    for i, tag in enumerate(tags_data["tags"]):
        tag_id = tag["id"]
        x = tag["position"]["x"]
        y = tag["position"]["y"]
        z = tag["position"]["z"]

        # Add coordinates to lists for scaling
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        # Use modulo to cycle through colors for many tags
        color_idx = i % 20
        
        # Plot the tag as a point
        ax.scatter(x, y, z, marker='o', s=100, color=colors[color_idx])
        
        # Add a text label with the tag ID
        ax.text(x, y, z, f"ID: {tag_id}", size=10, zorder=1, color='k')
    
    # Add camera at origin
    xs.append(0)
    ys.append(0)
    zs.append(0)

    # Set axis labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    
    # Set plot title
    ax.set_title('3D Visualization of AprilTags')
    
    # Add the camera position at origin
    ax.scatter(0, 0, 0, marker='^', color='red', s=150, label='Camera')
    
    # Add coordinate system axes at camera position
    axis_length = 0.1  # Length of coordinate axes
    
    # X-axis (red)
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
    # Y-axis (green)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
    # Z-axis (blue) - pointing outward from camera
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)
    
    # Calculate axis limits with padding
    if xs and ys and zs:
        # Determine the range of the data
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        
        # Calculate the center of the data
        x_center = (max(xs) + min(xs)) / 2
        y_center = (max(ys) + min(ys)) / 2
        z_center = (max(zs) + min(zs)) / 2
        
        # Determine the maximum range for equal aspect ratio
        max_range = max(x_range, y_range, z_range) * 0.6
        
        # Set limits with padding
        padding = max_range * 0.3
        ax.set_xlim([x_center - max_range - padding, x_center + max_range + padding])
        ax.set_ylim([y_center - max_range - padding, y_center + max_range + padding])
        ax.set_zlim([z_center - max_range - padding, z_center + max_range + padding])
    
    # Add a legend
    ax.legend()
    
    # Add a grid for better depth perception
    ax.grid(True)

    # Enhance the 3D view with a better perspective
    ax.view_init(elev=30, azim=45)

    # # Set axis limits for better visualization
    # # These can be adjusted based on your tag positions
    # max_range = 0.5  # Adjust this based on your expected tag distances
    # ax.set_xlim([-max_range, max_range])
    # ax.set_ylim([-max_range, max_range])
    # ax.set_zlim([0, max_range * 2])  # Assuming tags are in front of camera (positive Z)
    
    # Save the figure
    plt.savefig("apriltags_3d.png", dpi=300)
    print("3D visualization saved as 'apriltags_3d.png'")
    
    # Show the plot
    # plt.show()

def main():
    # Path to your video file
    video_path = 'plantage_shed.mp4'
    
    # Frame index to process (0 is the first frame)
    frame_index = 0
    
    # Camera parameters from your provided intrinsics
    # camera_params = camera_intrinsics
    
    # AprilTag size in meters
    tag_size = 0.042 # 42mm tag
    
    # Detect AprilTags and get their 3D coordinates
    result = detect_apriltags_in_frame(video_path, frame_index, camera_intrinsics, tag_size)
    
    if result:
        # Save results to JSON file
        with open("apriltag_coordinates.json", "w") as f:
            json.dump(result, f, indent=4)
        print("Results saved to 'apriltag_coordinates.json'")
        
        # Visualize tags in 3D
        visualize_tags_3d(result)

if __name__ == "__main__":
    main()