import cv2
import numpy as np
# import apriltag
# from pupil_apriltags import Detector
import json

def estimate_multiple_tag_positions(video_path, detector, camera_matrix, dist_coeffs, tag_size):
    # Video capture
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    
    # Store tag observations by frame: {frame_idx: {tag_id: corners, ...}}
    all_observations = {}
    frame_idx = 0
    # all_centers = {}
    
    # Collect observations from all frames
    print("Collecting tag observations from video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        
        if len(results) == 0:
            continue
            
        # Store observations for this frame: {tag_id: corners}, corners are 2D points as numpy arrays
        frame_observations = {}
        # frame_centers = {}
        for r in results:
            frame_observations[r.tag_id] = np.array(r.corners, dtype=np.float32)
            # if r.tag_id in (2, 3, 39):
            #     #TODO: save centers in a new dict maybe
            #     frame_centers[r.tag_id] = np.array(r.center, dtype=np.float32)
            
        all_observations[frame_idx] = frame_observations
        # if frame_centers:
        #     all_centers[frame_idx] = frame_centers
        
        if frame_idx == 0:
            print(f"Detected {len(frame_observations)} tags in frame {frame_idx}")
            print(f"Printing frame observations for the 0th frame: {all_observations[frame_idx]}")

        frame_idx += 1
        
    cap.release()
    print(f"Collected observations from {frame_idx} frames")
    
    # Choose reference tag (most frequently observed)
    tag_counts = {}
    for frame_obs in all_observations.values():
        for tag_id in frame_obs.keys():
            tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
            
    reference_tag_id = max(tag_counts, key=tag_counts.get)
    print(f"Selected tag {reference_tag_id} as reference")
    
    # Initialize tag positions
    tag_positions = {} # {tag_id: corners}
    # tag_centers = {} # {tag_id: center} for ids 2, 3, 39
    
    # Set reference tag at origin
    tag_positions[reference_tag_id] = np.array([
        [-tag_size/2, -tag_size/2, 0.],
        [tag_size/2, -tag_size/2, 0.],
        [tag_size/2, tag_size/2, 0.],
        [-tag_size/2, tag_size/2, 0.]
    ], dtype=np.float32)
    
    # Estimate other tag positions
    # This is simplified - in practice you'd implement full SfM or bundle adjustment
    tag_position_estimates = []
    # tag_center_estimates = []
    
    # For each frame where the reference tag is visible
    for frame_idx, frame_obs in all_observations.items():
        if reference_tag_id not in frame_obs:
            continue
            
        # Get reference tag corners in this frame
        ref_corners = frame_obs[reference_tag_id]
        
        # Solve PnP for reference tag
        success, rvec, tvec = cv2.solvePnP(
            tag_positions[reference_tag_id],
            ref_corners,
            camera_matrix,
            dist_coeffs
        )
        
        if not success:
            continue
            
        # For each other tag in this frame
        for tag_id, corners in frame_obs.items():
            if tag_id == reference_tag_id:
                continue
                
            # Estimate 3D corners for this tag
            tag_corners_3d = []
            for corner in corners:
                # Convert image point to 3D
                corner_3d = image_to_world(
                    corner, 
                    camera_matrix, 
                    rvec, tvec
                )
                
                if corner_3d is not None:
                    tag_corners_3d.append(corner_3d)
            
            # Only use if we got all 4 corners
            if len(tag_corners_3d) == 4:
                tag_position_estimates.append({
                    'tag_id': tag_id,
                    'corners': np.array(tag_corners_3d, dtype=np.float32)
                })
    
    # Average multiple estimates for each tag
    for tag_id in set(est['tag_id'] for est in tag_position_estimates):
        if tag_id in tag_positions:
            continue  # Skip if already have position
            
        # Collect all estimates for this tag
        estimates = [est['corners'] for est in tag_position_estimates if est['tag_id'] == tag_id]
        
        # Average the estimates
        avg_corners = np.mean(estimates, axis=0)
        tag_positions[tag_id] = avg_corners
        
    print(f"Estimated positions for {len(tag_positions)} tags")
    
    # Output the results
    for tag_id, corners in tag_positions.items():
        print(f"Tag {tag_id} corners:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: {corner}")
    
    return tag_positions

# Helper function to convert image point to 3D
def image_to_world(image_point, camera_matrix, rvec, tvec):
    """
    Convert an image point to a 3D point on the tag plane (Z=0)
    This uses ray-plane intersection
    """

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Get camera center in world coordinates
    C = -np.dot(R.T, tvec)
    
    # Camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Direction of ray from camera center through image point
    ray_dir = np.array([
        (image_point[0] - cx) / fx,
        (image_point[1] - cy) / fy,
        1.0
    ])
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # Transform ray direction to world coordinates
    ray_dir_world = np.dot(R.T, ray_dir)
    
    # Plane equation: Z = 0
    plane_normal = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 0])
    
    # Ray-plane intersection
    # Line equation: C + t * ray_dir_world
    # Plane equation: dot(p - plane_point, plane_normal) = 0
    # Solve for t: dot(C + t * ray_dir_world - plane_point, plane_normal) = 0
    numerator = np.dot(plane_point - C.flatten(), plane_normal)
    denominator = np.dot(ray_dir_world, plane_normal)
    
    # Check if ray is parallel to plane
    if abs(denominator) < 1e-6:
        return None
        
    t = numerator / denominator
    
    # Compute 3D point
    point_3d = C.flatten() + (t * ray_dir_world)
    
    return point_3d

def visualize_tag_positions(video_path, detector, tag_positions, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(video_path)
    # detector = Detector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        
        # Draw detected tags
        for r in results:
            # Draw detected corners
            for i, corner in enumerate(r.corners):
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
            
            # If tag has estimated 3D position
            if r.tag_id in tag_positions:
                # Solve PnP to get pose
                success, rvec, tvec = cv2.solvePnP(
                    tag_positions[r.tag_id],
                    np.array(r.corners, dtype=np.float32),
                    camera_matrix,
                    dist_coeffs
                )
                
                if success:
                    # Draw axis
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                    
                    # Display tag ID
                    center = np.mean(r.corners, axis=0).astype(int)
                    cv2.putText(frame, str(r.tag_id), tuple(center), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite("Tag_Positions.jpg", frame)
        # print("Visualization saved as 'Tag_Positions.jpg'")

        cv2.imshow('Tag Positions', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def save_tag_positions(tag_positions, filename):
    # Create a serializable version of tag_positions
    serializable_tag_positions = []
    tag_dict = {}
    
    for tag_id, corners in tag_positions.items():
        # Convert NumPy arrays to Python lists
        tag_dict["id"] = tag_id
        tag_dict["corners"] = corners.tolist()
        serializable_tag_positions.append(tag_dict.copy())
        # serializable_tag_positions[str(tag_id)] = corners.tolist()
    
    # Write to JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_tag_positions, f, indent=4)
    
    print(f"Tag positions saved to {filename}")