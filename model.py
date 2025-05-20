import cv2
import numpy as np
# import apriltag
# from pupil_apriltags import Detector
import json
# from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares

def estimate_tag_positions_3d(video_path, detector, camera_matrix, dist_coeffs, tag_size, known_constraints=None):
    """
    Main function to estimate 3D positions of AprilTags in a video. Utilizes the functions below
    
    Args:
        video_path (str): Path to the video file containing AprilTag footage
        detector (Detector): Initialized AprilTag detector object
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Camera distortion coefficients
        tag_size (float): Physical size of the AprilTag in mm
        known_constraints (list, optional): List of distance constraints between tags, 
                                           each tuple contains (tag_id1, tag_id2, distance_mm)
    
    Returns:
        tuple: (refined_tag_positions, all_observations, reference_tag_id)
            - refined_tag_positions (dict): Dict mapping tag_id to 3D corner positions after optimization
            - all_observations (dict): Dict mapping frame indices to tag observations
            - reference_tag_id (int): ID of the reference tag used as origin
    """

    # Step 1: Collect observations from all frames
    all_observations = collect_tag_observations(video_path, detector)
    
    # Step 2: Find the most frequently seen tag to use as reference
    reference_tag_id = find_reference_tag(all_observations)

    # Step 3: Initialize reference tag at origin
    tag_positions = initialize_reference_tag(reference_tag_id, tag_size)
    
    # Step 4: Triangulate positions of other tags
    print("Triangulating tag positions...")
    tag_positions = triangulate_tag_positions(
        all_observations, reference_tag_id, tag_positions, camera_matrix, dist_coeffs
    )
    
    # Step 5: Optimize tag positions with bundle adjustment with distance constraints
    print("Optimizing tag positions with bundle adjustment...")
    refined_tag_positions = optimize_tag_positions(
        all_observations, tag_positions, camera_matrix, dist_coeffs, reference_tag_id, known_constraints
    )

    # Output the results
    for tag_id, corners in refined_tag_positions.items():
        print(f"Tag {tag_id} corners:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: {corner}")
    
    # Step 6: Validate distance constraints if provided
    validate_distance_constraints(refined_tag_positions, known_constraints, verbose=True)

    return refined_tag_positions, all_observations, reference_tag_id

def collect_tag_observations(video_path, detector, frame_interval=10):
    """
    Processes the video to detect AprilTags in frames and collects observations
    
    Args:
        video_path (str): Path to the video file to process
        detector (Detector): Initialized AprilTag detector object
        frame_interval (int, optional): Process every Nth frame to reduce computation
    
    Returns:
        dict: Dictionary mapping frame indices to tag observations
              Format: {frame_idx: {tag_id: corners, ...}, ...}
              where corners are 2D points as numpy arrays
    """

    print("Opening video...")
    cap = cv2.VideoCapture(video_path)

    # # Get frame dimensions
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # # Get total frame count
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # Get frames per second (FPS)
    # fps = cap.get(cv2.CAP_PROP_FPS)

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

        # Process only every Nth frame
        if frame_idx % frame_interval == 0:
            
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
            
            # if frame_idx == 0:
                # print(f"Detected {len(frame_observations)} tags in frame {frame_idx}")
                # print(f"Printing frame observations for the 0th frame: {all_observations[frame_idx]}")

            # print(f"Processed frame {frame_idx}: {len(results)} tags detected")

        frame_idx += 1
        
    cap.release()
    print(f"Collected observations from {frame_idx} frames")

    return all_observations

def find_reference_tag(all_observations):
    """
    Identifies the most frequently observed tag to use as reference point/origin
    
    Args:
        all_observations (dict): Dictionary of tag observations by frame
                               Format: {frame_idx: {tag_id: corners, ...}, ...}
    
    Returns:
        int: ID of the tag that appears most frequently in the video
    """

    # Choose reference tag (most frequently observed)
    tag_counts = {}
    for frame_obs in all_observations.values():
        for tag_id in frame_obs.keys():
            tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
            
    reference_tag_id = max(tag_counts, key=tag_counts.get)
    print(f"Selected tag {reference_tag_id} as reference")

    return reference_tag_id

def initialize_reference_tag(reference_tag_id, tag_size):
    """
    Initializes the 3D position of the reference tag, placing it at the origin
    
    Args:
        reference_tag_id (int): ID of the tag to use as reference/origin
        tag_size (float): Physical size of the AprilTag in mm
    
    Returns:
        dict: Dictionary mapping the reference tag ID to its 3D corner positions
              with the tag centered at the origin and lying on the XY plane
    """

    # AprilTag corners are typically defined as:
    # - Bottom left, bottom right, top right, top left (clockwise from bottom left)
    # - In the tag's coordinate system, z=0 for all corners (tag is flat)

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

    return tag_positions

def triangulate_tag_positions(all_observations, reference_tag_id, tag_positions, camera_matrix, dist_coeffs):
    """
    Triangulates positions of all tags using multiple views of the reference tag
    
    Args:
        all_observations (dict): Dictionary of tag observations by frame
        reference_tag_id (int): ID of the reference tag
        tag_positions (dict): Dictionary of initialized tag positions (only contains reference tag)
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Camera distortion coefficients
    
    Returns:
        dict: Updated dictionary mapping tag IDs to their estimated 3D corner positions
    """

    # Initialize a dictionary to collect observations for each tag
    tag_observations = {}

    # For each frame where the reference tag is visible
    for frame_idx, frame_observations in all_observations.items():
        
        # Skip frame if reference tag not visible
        if reference_tag_id not in frame_observations:
            continue

        # Get reference tag corners in this frame
        ref_corners_2d = frame_observations[reference_tag_id]
        ref_corners_3d = tag_positions[reference_tag_id]
        
        # Solve PnP to get camera pose relative to reference tag
        success, rvec, tvec = cv2.solvePnP(
            ref_corners_3d, ref_corners_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        
        if not success:
            continue
        
        # For each other tag in this frame
        for tag_id, corners in frame_observations.items():
            
            # Skip reference tag and already triangulated tags
            if tag_id == reference_tag_id or tag_id in tag_positions:
                continue
            
            # Initialize observation list for this tag if needed
            if tag_id not in tag_observations:
                tag_observations[tag_id] = []

            # For now, use a simple back-projection method
            # Store the observation for multi-view triangulation
            tag_observations[tag_id].append({
                'corners_2d': corners,
                'rvec': rvec.copy(),
                'tvec': tvec.copy()
            })
    
    # Now triangulate each tag using observations from multiple frames
    for tag_id, observations_list in tag_observations.items():
        # Skip if fewer than 2 observations
        if len(observations_list) < 2:
            continue
        
        # Triangulate each corner separately
        tag_corners_3d = []
        
        for corner_idx in range(4):
            # Collect 2D points and projection matrices from all observations
            img_points = []
            proj_matrices = []
            
            for obs in observations_list:
                # Get corner 2D point
                corner_2d = obs['corners_2d'][corner_idx]
                img_points.append(corner_2d)
                
                # Get camera projection matrix
                rvec, tvec = obs['rvec'], obs['tvec']
                R, _ = cv2.Rodrigues(rvec)
                
                # Projection matrix P = K[R|t]
                P = np.zeros((3, 4))
                P[:3, :3] = R
                P[:3, 3] = tvec.ravel()
                P = camera_matrix @ P
                
                proj_matrices.append(P)
            
            # Convert to numpy arrays
            img_points = np.array(img_points)
            proj_matrices = np.array(proj_matrices)
            
            # Triangulate this corner (Direct Linear Transform)
            corner_3d = triangulate_point_dlt(img_points, proj_matrices)
            tag_corners_3d.append(corner_3d)
        
        # Store triangulated tag corners
        tag_positions[tag_id] = np.array(tag_corners_3d, dtype=np.float32)
    
    return tag_positions

def triangulate_point_dlt(img_points, proj_matrices):
    """
    Triangulates a 3D point from multiple 2D observations using Direct Linear Transform
    
    Args:
        img_points (numpy.ndarray): 2D points in multiple images
        proj_matrices (numpy.ndarray): Projection matrices for each view
    
    Returns:
        numpy.ndarray: 3D point coordinates
    """

    n_views = len(img_points)
    A = np.zeros((2 * n_views, 4))
    
    for i in range(n_views):
        x, y = img_points[i]
        P = proj_matrices[i]
        
        A[2*i] = x * P[2] - P[0]
        A[2*i+1] = y * P[2] - P[1]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    point_3d = Vt[-1]
    
    # Convert to homogeneous coordinates with safety check
    if abs(point_3d[3]) > 1e-8:  # Avoid division by zero or near-zero values
        point_3d = point_3d / point_3d[3]
    else:
        # In case w is zero or very small, we're dealing with a point at infinity
        # In this case, we can normalize the direction vector
        norm = np.linalg.norm(point_3d[:3])
        if norm > 1e-8:
            point_3d[:3] = point_3d[:3] / norm
    
    return point_3d[:3]

def optimize_tag_positions(
        all_observations, initial_tag_positions, camera_matrix, 
        dist_coeffs, reference_tag_id, constraints=None
        ):
    """
    Optimizes tag positions using bundle adjustment to minimize reprojection errors
    and enforce distance constraints
    
    Args:
        all_observations (dict): Dictionary of tag observations by frame
        initial_tag_positions (dict): Initial estimates of tag 3D positions
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Camera distortion coefficients
        reference_tag_id (int): ID of the reference tag (will remain fixed)
        constraints (list, optional): List of tuples (tag_id1, tag_id2, distance_mm)
                                     representing known distances between tag centers
    
    Returns:
        dict: Optimized tag positions mapping tag IDs to their refined 3D corner positions
    """

    # Use empty list if no constraints provided
    if constraints is None:
        constraints = []

    # Extract all tag IDs
    tag_ids = list(initial_tag_positions.keys())

    # Remove reference tag from optimization parameters
    if reference_tag_id in tag_ids:
        tag_ids.remove(reference_tag_id)
    
    # Initialize parameter vector with tag positions
    params = []
    for tag_id in tag_ids:
        # Add each corner's coordinates
        corners = initial_tag_positions[tag_id]
        for corner in corners:
            params.extend(corner)
    
    params = np.array(params)
    
    # Pre-compute a list of valid frames with the reference tag visible
    valid_frames = []
    for frame_idx, frame_observations in all_observations.items():
        if reference_tag_id in frame_observations:
            valid_frames.append(frame_idx)

    # Define cost function for optimization that uses fixed reference tag
    def cost_function(params, distance_weight = 10.0):
        # Adjust distance_weight to balance reprojection vs. distance errors
        # Reconstruct tag positions from parameters
        # tag_positions = {}
        tag_positions = {reference_tag_id: initial_tag_positions[reference_tag_id]}  # Start with fixed reference tag
        param_idx = 0
        
        for tag_id in tag_ids: # Only non-reference tags
            corners = []
            for _ in range(4):
                corner = params[param_idx:param_idx+3]
                corners.append(corner)
                param_idx += 3
            
            tag_positions[tag_id] = np.array(corners)
        
        # Compute reprojection errors
        # Pre-allocate a large enough error array
        # This is critical for maintaining a consistent size
        max_possible_errors = len(valid_frames) * len(tag_ids) * 8  # 8 = 4 corners * 2 (x,y)
        errors = np.zeros(max_possible_errors)
        error_idx = 0
        # errors = []
        
        for frame_idx in valid_frames:
            frame_observations = all_observations[frame_idx]
            
            # Skip if reference tag not in this frame (should be impossible due to valid_frames)
            if reference_tag_id not in frame_observations:
                continue
            
            # Get reference tag corners
            ref_corners_3d = tag_positions[reference_tag_id]
            ref_corners_2d = frame_observations[reference_tag_id]
            
            # Get camera pose using reference tag
            try:
                success, rvec, tvec = cv2.solvePnP(
                    ref_corners_3d, ref_corners_2d, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )
            except Exception as e:
                print(f"solvePnP failed: {e}")
                continue
            
            if not success:
                continue
            
            # For each tag in this frame
            for tag_id, obs in frame_observations.items():
                if tag_id not in tag_positions:
                    continue
                
                # Project 3D corners to 2D
                corners_3d = tag_positions[tag_id]
                try:
                    projected_corners, _ = cv2.projectPoints(
                        corners_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    projected_corners = projected_corners.reshape(-1, 2)
                except Exception as e:
                    print(f"projectPoints failed: {e}")
                    continue

                # Compute reprojection error
                observed_corners = obs #['corners']
                for i in range(4):
                    # error = observed_corners[i] - projected_corners[i]
                    # errors.append(error)

                    # Only add errors if we haven't exceeded our pre-allocated size
                    if error_idx + 2 <= max_possible_errors:
                        error = observed_corners[i] - projected_corners[i]
                        errors[error_idx] = error[0]  # x error
                        errors[error_idx + 1] = error[1]  # y error
                        error_idx += 2
        
        # Add distance constraints
        for tag_id1, tag_id2, expected_dist in constraints:
            if tag_id1 in tag_positions and tag_id2 in tag_positions:
                # Calculate center of each tag
                center1 = np.mean(tag_positions[tag_id1], axis=0)
                center2 = np.mean(tag_positions[tag_id2], axis=0)
                
                # Calculate actual distance
                actual_dist = np.linalg.norm(center2 - center1)
                
                # Add weighted distance error
                dist_error = distance_weight * (actual_dist - expected_dist)
                
                # Add to errors array
                if error_idx < max_possible_errors:
                    errors[error_idx] = dist_error
                    error_idx += 1


        # return np.array(errors).flatten()
        # Return only the filled portion of the errors array
        return errors[:error_idx]
    
    # Run optimization (trf is more robust than lm)
    result = least_squares(cost_function, params, method='trf', ftol=1e-4, verbose=1)
    
    # Reconstruct optimized tag positions
    optimized_params = result.x
    # optimized_tag_positions = {}
    optimized_tag_positions = {reference_tag_id: initial_tag_positions[reference_tag_id]}  # Reference tag is unchanged
    param_idx = 0
    
    for tag_id in tag_ids:
        corners = []
        for _ in range(4):
            corner = optimized_params[param_idx:param_idx+3]
            corners.append(corner)
            param_idx += 3
        
        optimized_tag_positions[tag_id] = np.array(corners)
    
    return optimized_tag_positions

def validate_distance_constraints(tag_positions, constraints, verbose=True):
    """
    Validates how well the optimized tag positions match the expected distances
    
    Args:
        tag_positions (dict): Dictionary mapping tag IDs to their 3D corner positions
        constraints (list): List of tuples (tag_id1, tag_id2, distance_mm)
                          representing expected distances between tag centers
        verbose (bool): Whether to print detailed validation information
    
    Returns:
        dict: Statistics about distance errors (if verbose=False, otherwise None)
    """

    # Initialize statistics
    stats = {
        'num_constraints': len(constraints),
        'constraints_evaluated': 0,
        'mean_absolute_error': 0,
        'max_absolute_error': 0,
        'errors': []
    }
    
    if verbose:
        print(f"\n--- Distance Constraint Validation ---")
        print(f"Number of constraints: {stats['num_constraints']}")
    
    # Evaluate each constraint
    for tag_id1, tag_id2, expected_distance in constraints:
        if tag_id1 not in tag_positions or tag_id2 not in tag_positions:
            if verbose:
                print(f"Skipping constraint between tags {tag_id1} and {tag_id2}: One or both tags not detected")
            continue
            
        # Calculate centers of both tags
        center1 = np.mean(tag_positions[tag_id1], axis=0)
        center2 = np.mean(tag_positions[tag_id2], axis=0)
        
        # Calculate current distance
        current_vector = center2 - center1
        current_distance = np.linalg.norm(current_vector)
        
        # Calculate error
        error = current_distance - expected_distance
        absolute_error = abs(error)
        
        # Update statistics
        stats['constraints_evaluated'] += 1
        stats['errors'].append(error)
        stats['max_absolute_error'] = max(stats['max_absolute_error'], absolute_error)
        
        if verbose:
            print(f"Tags {tag_id1} and {tag_id2}:")
            print(f"  Expected distance: {expected_distance:.2f} mm")
            print(f"  Actual distance:   {current_distance:.2f} mm")
            print(f"  Error:             {error:.2f} mm ({error/expected_distance*100:.2f}%)")
    
    # Calculate mean error if constraints were evaluated
    if stats['constraints_evaluated'] > 0:
        stats['mean_absolute_error'] = np.mean(np.abs(stats['errors']))
        stats['rms_error'] = np.sqrt(np.mean(np.array(stats['errors'])**2))
        
        if verbose:
            print("\n--- Summary Statistics ---")
            print(f"Constraints evaluated: {stats['constraints_evaluated']} / {stats['num_constraints']}")
            print(f"Mean absolute error:   {stats['mean_absolute_error']:.2f} mm")
            print(f"RMS error:             {stats['rms_error']:.2f} mm")
            print(f"Maximum absolute error: {stats['max_absolute_error']:.2f} mm")
            
            # Print histogram-like distribution of errors
            if len(stats['errors']) > 5:
                errors = np.array(stats['errors'])
                percentiles = [0, 25, 50, 75, 100]
                values = np.percentile(np.abs(errors), percentiles)
                print("\nError distribution (absolute errors):")
                for p, v in zip(percentiles, values):
                    print(f"  {p}th percentile: {v:.2f} mm")
    else:
        if verbose:
            print("No constraints could be evaluated!")
    
    #return stats

def save_tag_positions(tag_positions, filename):
    """
    Saves the estimated 3D tag positions to a JSON file
    
    Args:
        tag_positions (dict): Dictionary mapping tag IDs to their 3D corner positions
        filename (str): Path to save the JSON file
    
    Returns:
        None: Writes the data to the specified file
    """
    
    # Create a serializable version of tag_positions
    serializable_tag_positions = []
    tag_dict = {}
    
    for tag_id, corners in tag_positions.items():
        # Convert NumPy arrays to Python lists
        tag_dict["id"] = tag_id
        tag_dict["corners"] = np.round(corners, 2).tolist()
        serializable_tag_positions.append(tag_dict.copy())
        # serializable_tag_positions[str(tag_id)] = corners.tolist()
    
    # Write to JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_tag_positions, f, indent=4)
    
    print(f"Tag positions saved to {filename}")