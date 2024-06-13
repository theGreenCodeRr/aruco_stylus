import numpy as np

def get_rotation_matrix(rot_x, rot_y, rot_z):
    """Create a rotation matrix from Euler angles."""
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rot_x), -np.sin(rot_x)],
        [0, np.sin(rot_x), np.cos(rot_x)]
    ])
    ry = np.array([
        [np.cos(rot_y), 0, np.sin(rot_y)],
        [0, 1, 0],
        [-np.sin(rot_y), 0, np.cos(rot_y)]
    ])
    rz = np.array([
        [np.cos(rot_z), -np.sin(rot_z), 0],
        [np.sin(rot_z), np.cos(rot_z), 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx
def create_transformation_matrix(position, orientation):
    """Create a 4x4 transformation matrix from position and orientation."""
    rotation_matrix = get_rotation_matrix(*orientation)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix
def main():
    # Define the position and orientation of the camera and marker in world coordinates
    cam_position = np.array([1.0, 2.0, 3.0])
    cam_orientation = np.array([0.1, 0.2, 0.3])  # Euler angles (roll, pitch, yaw)
    marker_position = np.array([4.0, 5.0, 6.0])
    marker_orientation = np.array([0.4, 0.5, 0.6])  # Euler angles (roll, pitch, yaw)
    # Create the transformation matrices for camera and marker
    T_cam_world = create_transformation_matrix(cam_position, cam_orientation)
    T_marker_world = create_transformation_matrix(marker_position, marker_orientation)
    # Compute the transformation matrix from the camera to the marker
    T_cam_marker = np.linalg.inv(T_cam_world) @ T_marker_world
    print("Transformation matrix from camera to marker:")
    print(T_cam_marker)
if __name__ == "__main__":
    main()