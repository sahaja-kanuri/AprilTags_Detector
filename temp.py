# functions not in use:

def apply_distance_constraints(tag_positions, constraints):
    """
    Apply known distance constraints between tag centers
    
    Args:
        tag_positions: Dict mapping tag_id to 3D corner positions
        constraints: List of tuples (tag_id1, tag_id2, distance_mm)
    """
    # Make a copy to avoid modifying the input directly
    updated_positions = tag_positions.copy()

    # # Convert all tag positions to center points
    # centers = {}
    # for tag_id, corners in tag_positions.items():
    #     centers[tag_id] = np.mean(corners, axis=0)
    
    # Apply constraints iteratively
    for _ in range(5):  # Run multiple iterations for better convergence
        max_adjustment = 0
        for tag_id1, tag_id2, expected_distance in constraints:
            if tag_id1 not in updated_positions or tag_id2 not in updated_positions:
                print(f"Skipping constraint between tags {tag_id1} and {tag_id2}: One or both tags not detected")
                continue
                
            # Calculate centers of both tags
            center1 = np.mean(updated_positions[tag_id1], axis=0)
            center2 = np.mean(updated_positions[tag_id2], axis=0)
            
            # Calculate current distance
            current_vector = center2 - center1
            current_distance = np.linalg.norm(current_vector)
            print(f"Current distance between tags {tag_id1} and {tag_id2}: {current_distance:.2f} in mm")
            print(f"Expected distance between tags {tag_id1} and {tag_id2}: {expected_distance:.2f} in mm")
            
            # Skip if already close enough
            if abs(current_distance - expected_distance) < 1.:  # (All distances are in mm) 1mm tolerance
                continue
                
            # # Adjust both centers to maintain relative positions
            # adjustment = (expected_distance - current_distance) / 2
            # direction = current_vector / current_distance
            # # Apply the adjustment to each tag (moving them apart or together)
            # shift1 = - adjustment * direction
            # shift2 = adjustment * direction

            #OR:
            # Calculate adjustment factor
            adjustment_factor = (expected_distance / current_distance) - 1.0
            # Preserve direction but adjust magnitude
            adjustment_vector = current_vector * adjustment_factor
            # Apply half of the adjustment to each tag (moving them apart or together)
            shift1 = -0.5 * adjustment_vector
            shift2 = 0.5 * adjustment_vector

            # Apply adjustments to each corner
            for i in range(4):
                updated_positions[tag_id1][i] += shift1
                updated_positions[tag_id2][i] += shift2

             # Track maximum adjustment for convergence check
            adjustment_magnitude = np.linalg.norm(adjustment_vector)
            max_adjustment = max(max_adjustment, adjustment_magnitude)
            
            # # Update corner positions based on new centers
            # for tag_id in [tag_id1, tag_id2]:
            #     old_center = np.mean(tag_positions[tag_id], axis=0)
            #     translation = centers[tag_id] - old_center
            #     tag_positions[tag_id] = tag_positions[tag_id] + translation
        
        # Check for convergence
        if max_adjustment < 1e-5:
            # print(f"Converged after {iteration+1} iterations")
            break
    return updated_positions


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

def estimateAndAverage_other_tag_positions(all_observations, reference_tag_id, tag_positions,
                                           camera_matrix, dist_coeffs):
    # DELETE LATER WHEN NOT IN USE
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
    return tag_positions

    