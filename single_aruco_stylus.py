import cv2
from cv2 import aruco
import numpy as np
import math


def camera_setup():
    """Initialize the camera and set the intrinsic parameters."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera calibration parameters (example for a Mac webcam)
    cameraMatrix = np.array([
        [581.2490064821088, 0.0, 305.2885321972521],
        [0.0, 587.6316817762934, 288.9932758741485],
        [0.0, 0.0, 1.0]], dtype='double')

    distCoeffs = np.array([
        [-0.31329614267146066],
        [0.8386295742029726],
        [-0.0024210244191179104],
        [0.016349338905846198],
        [-1.133637004544031]], dtype='double')

    return cap, cameraMatrix, distCoeffs


def aruco_setup():
    """Setup ArUco marker dictionary and detection parameters."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.059  # Marker size in meters
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size


def pose_estimation(corner, marker_size, cameraMatrix, distCoeffs):
    """Estimate the pose of the ArUco marker."""
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    success, rvecs, tvecs = cv2.solvePnP(marker_points, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)

    if success:
        if rvecs is not None and rvecs.shape != (3, 1):
            rvecs = rvecs.reshape((3, 1))
        if tvecs is not None and tvecs.shape != (3, 1):
            tvecs = tvecs.reshape((3, 1))
    else:
        return None, None

    return rvecs, tvecs


def calculate_euler_angles(rvec):
    """Convert rotation vector to Euler angles."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def display_marker_info(frame, rvec, tvec, font):
    """Display distance, angle, and 3D position information on the frame."""
    # Calculate Euler angles (yaw, pitch, roll)
    angles = calculate_euler_angles(rvec)

    # Convert from meters to millimeters and extract x, y, z components
    x, y, z = tvec[0][0] * 1000, tvec[1][0] * 1000, tvec[2][0] * 1000  # Convert to mm
    distance_rounded = round(z, 4)  # Z-axis is often used as distance

    # Display the information on the frame
    cv2.putText(frame, f'X: {round(x, 2)} mm', (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Y: {round(y, 2)} mm', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Z (Distance): {distance_rounded} mm', (10, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Yaw: {round(angles[2], 2)} deg', (10, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Pitch: {round(angles[0], 2)} deg', (10, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Roll: {round(angles[1], 2)} deg', (10, 230), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


def draw_detected_markers(frame, corners, ids):
    """Draw markers and IDs on the frame."""
    aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    for i, corner in enumerate(corners):
        points = corner[0].astype(np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255))
        cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


def draw_marker_axes(frame, cameraMatrix, distCoeffs, rvec, tvec):
    """Draw 3D axes on each detected marker to show orientation."""
    # Project the 3D points to 2D for cube visualization
    object_points = np.array([
        [-0.02, -0.02, 0],  # 0
        [0.02, -0.02, 0],  # 1
        [0.02, 0.02, 0],  # 2
        [-0.02, 0.02, 0],  # 3
        [-0.02, -0.02, 0.02],  # 4
        [0.02, -0.02, 0.02],  # 5
        [0.02, 0.02, 0.02],  # 6
        [-0.02, 0.02, 0.02]  # 7
    ], dtype=np.float32)

    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, cameraMatrix, distCoeffs)
    image_points = image_points.reshape(-1, 2)  # Reshape the output for easier indexing

    # Draw the projected 2D points on the frame
    for point in image_points:
        point = tuple(map(int, point))  # Convert the point to a tuple of integers
        cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Draw points (red)

    # Draw lines to form a cube
    for i in range(4):  # Bottom square
        start_point = tuple(map(int, image_points[i]))
        end_point = tuple(map(int, image_points[(i + 1) % 4]))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Line 1-2, 2-3, 3-4, 4-1

    # Top square
    for i in range(4, 8):  # Top square
        start_point = tuple(map(int, image_points[i]))
        end_point = tuple(map(int, image_points[(i + 1) % 4 + 4]))  # Connecting the top points (5-6, 6-7, 7-8, 8-5)
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Connect top and bottom
    for i in range(4):  # Connecting top and bottom (vertical lines)
        start_point = tuple(map(int, image_points[i]))
        end_point = tuple(map(int, image_points[i + 4]))  # Connecting bottom to top (1-5, 2-6, 3-7, 4-8)
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Draw x, y, z axes on each detected marker to show orientation
    axis_length = 0.03  # Length of the axes in meters (e.g., 30 mm)

    # Draw the x (red), y (green), and z (blue) axes on the frame
    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, axis_length)


def main():
    # Setup camera and marker detection parameters
    cap, cameraMatrix, distCoeffs = camera_setup()
    detector, marker_size = aruco_setup()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Undistort frame to remove lens distortion
        frame_undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)

        # Detect ArUco markers in the frame
        corners, ids, _ = detector.detectMarkers(frame_undistorted)

        if len(corners) > 0:
            for i, corner in enumerate(corners):
                rvecs, tvecs = pose_estimation(corner, marker_size, cameraMatrix, distCoeffs)

                if rvecs is not None and tvecs is not None:
                    display_marker_info(frame, rvecs, tvecs, font)
                    draw_detected_markers(frame, corners, ids)
                    draw_marker_axes(frame, cameraMatrix, distCoeffs, rvecs, tvecs)  # Draw XYZ axes

        # Display the processed frame
        cv2.imshow("Frame", frame)

        # Break loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
