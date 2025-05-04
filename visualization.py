# Visualization functions:

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def visualize_tags_3d(tag_positions, reference_tag_id, constraints, save_path=None):
    """
    Create an interactive 3D visualization of AprilTags
    that allows for dragging and rotating the view
    
    Args:
        tag_positions: Dict mapping tag_id to 3D corner positions
        reference_tag_id: ID of the reference tag (origin)
        constraints: List of tuples (tag_id1, tag_id2, distance) representing distance constraints
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store tag centers for constraint lines
    tag_centers = {}

    # For creating the legend
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Reference Tag'),
        Line2D([0], [0], color='blue', lw=2, label='Other Tags'),
        Line2D([0], [0], color='orange', lw=2, label='Constraints')
    ]
    
    # Plot each tag
    for tag_id, corners in tag_positions.items():
        # Extract coordinates
        x = [corner[0] for corner in corners]
        y = [corner[2] for corner in corners]  # y is depth
        z = [corner[1] for corner in corners]
        
        # Determine color based on whether this is the reference tag
        tag_color = 'red' if tag_id == reference_tag_id else 'blue'
        
        # Plot the edges (square outline)
        for i in range(4):
            next_i = (i + 1) % 4
            ax.plot([x[i], x[next_i]], [y[i], y[next_i]], [z[i], z[next_i]], 
                    color=tag_color, linewidth=2)
        
        # Plot the diagonals
        ax.plot([x[0], x[2]], [y[0], y[2]], [z[0], z[2]], color=tag_color, linewidth=1.5)
        ax.plot([x[1], x[3]], [y[1], y[3]], [z[1], z[3]], color=tag_color, linewidth=1.5)
        
        # Calculate tag center
        center_x = sum(x) / 4
        center_y = sum(y) / 4
        center_z = sum(z) / 4
        
        # Store center for constraint lines
        tag_centers[tag_id] = (center_x, center_y, center_z)
        
        # Add tag ID text to the left of the tag
        ax.text(center_x - 0.02, center_y, center_z, str(tag_id), 
                color='black', fontsize=12)
    
    # Draw constraint lines (orange L shape)
    if constraints:
        # Draw lines between constrained tags
        for tag_id1, tag_id2, distance in constraints:
            if tag_id1 in tag_centers and tag_id2 in tag_centers:
                center1 = tag_centers[tag_id1]
                center2 = tag_centers[tag_id2]
                
                # Draw orange line connecting the tag centers
                ax.plot([center1[0], center2[0]], [center1[1], center2[1]], [center1[2], center2[2]], 
                        color='orange', linewidth=2, linestyle='-')
                
                # Calculate midpoint for distance label
                mid_x = (center1[0] + center2[0]) / 2
                mid_y = (center1[1] + center2[1]) / 2
                mid_z = (center1[2] + center2[2]) / 2
                
                # Add distance text
                ax.text(mid_x, mid_y, mid_z, f"{distance}mm", 
                        color='orange', fontsize=10, ha='center')
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm) - Depth')
    ax.set_zlabel('Y (mm)')

    # Add the legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set an initial view
    ax.view_init(elev=15, azim=-100)
    
    # Turn off the grid for cleaner visualization
    ax.grid(False)
    
    # Enable tight layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Make the plot interactive - allows for dragging to rotate
    plt.show()

def visualize_tag_positions(video_path, all_observations, tag_positions, camera_matrix, dist_coeffs, output_path='tag_visualization.mp4'):
    """
    Visualize tag positions using existing observations and save the video
    
    Args:
        video_path: Path to the input video
        all_observations: Dictionary of tag observations by frame
        tag_positions: Dict mapping tag_id to 3D corner positions
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        output_path: Path to save the output video
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame has observations
        if frame_idx in all_observations:
            frame_observations = all_observations[frame_idx]
            
            # Draw each tag from the saved observations
            for tag_id, corners in frame_observations.items():
                # Draw detected corners
                for i, corner in enumerate(corners):
                    cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
                
                # Draw tag outline
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                
                # If tag has estimated 3D position
                if tag_id in tag_positions:
                    # Solve PnP to get pose
                    try:
                        success, rvec, tvec = cv2.solvePnP(
                            tag_positions[tag_id],
                            corners,
                            camera_matrix,
                            dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE
                        )
                        
                        if success:
                            # Draw axis
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                            
                            # Display tag ID
                            center = np.mean(corners, axis=0).astype(int)
                            cv2.putText(frame, str(tag_id), tuple(center), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Error visualizing tag {tag_id} in frame {frame_idx}: {e}")
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame (optional, can be commented out for faster processing)
        cv2.imshow('Tag Positions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Visualization saved to '{output_path}'")

def visualize_tag_positions_old(video_path, detector, tag_positions, camera_matrix, dist_coeffs):
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
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )
                
                if success:
                    # Draw axis
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                    
                    # Display tag ID
                    center = np.mean(r.corners, axis=0).astype(int)
                    cv2.putText(frame, str(r.tag_id), tuple(center), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # cv2.imwrite("Tag_Positions.jpg", frame)
        # print("Visualization saved as 'Tag_Positions.jpg'")

        cv2.imshow('Tag Positions', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()