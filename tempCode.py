import cv2
from cv2 import aruco
import numpy as np
import math


def camera_setup():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera calibration parameters
    cameraMatrix = np.array(
        [[505.1150576, 0, 359.14439401],
         [0, 510.33530166, 230.33963591],
         [0, 0, 1]],
        dtype='double')
    distCoeffs = np.array([[0.07632527], [0.15558049], [0.00234922], [0.00500232], [-0.46829062]], dtype='double')

    return cap, cameraMatrix, distCoeffs


def aruco_setup():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.059  # Marker size in meters
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size


def pose_estimation(corner, marker_size, cameraMatrix, distCoeffs):
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


def calculate_tip_position(rvec, tvec, offset):
    """Calculate the position of the stylus tip."""
    R, _ = cv2.Rodrigues(rvec)
    tip_position = tvec.reshape(-1) + R @ offset
    return tip_position

def calculate_euler_angles(rvec):
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


def draw_tip_as_dot(frame, rvec, tvec, cameraMatrix, distCoeffs, offset):
    """Project the 3D tip position onto the image as a red dot."""
    # Compute the 3D position of the tip
    R, _ = cv2.Rodrigues(rvec)
    tip_position = tvec.reshape(-1) + R @ offset  # Compute the tip position in the marker's coordinate frame
    tip_position = tip_position.reshape(1, 1, 3)  # Reshape for `projectPoints`

    # Project the 3D position of the tip onto the 2D image plane
    image_points, _ = cv2.projectPoints(tip_position, rvec, tvec, cameraMatrix, distCoeffs)

    # Extract the first 2D point
    tip_2d = image_points[0][0]  # Extract the point from the nested array
    # print("Extracted Tip (float):", tip_2d)  # Debugging step

    # Ensure tip_2d is converted to integer
    tip_2d = tuple(map(int, tip_2d))  # Convert to integer tuple
    # print("Converted Tip (int):", tip_2d)  # Debugging step

    # Draw the red dot on the frame
    cv2.circle(frame, tip_2d, 5, (0, 0, 255), -1)
    cv2.rectangle(frame, (tip_2d[0] - 5, tip_2d[1] - 5), (tip_2d[0] + 5, tip_2d[1] + 5), (0, 0, 255), -1)


def display_marker_info(frame, rvec, tvec, font):
    # Calculate Euler angles (yaw, pitch, roll)
    angles = calculate_euler_angles(rvec)

    # Convert from meters to millimeters and extract x, y, z components
    x, y, z = tvec[0][0] * 1000, tvec[1][0] * 1000, tvec[2][0] * 1000  # Marker center in mm

    # # Display the information on the frame
    # cv2.putText(frame, f'Marker X: {round(x, 2)} mm', (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'Marker Y: {round(y, 2)} mm', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'Marker Z: {round(z, 2)} mm', (10, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'Yaw: {round(angles[2], 2)} deg', (10, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'Pitch: {round(angles[0], 2)} deg', (10, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f'Roll: {round(angles[1], 2)} deg', (10, 230), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    # Setup camera and marker detection parameters
    cap, cameraMatrix, distCoeffs = camera_setup()
    detector, marker_size = aruco_setup()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Offset for the stylus tip (100mm deep, perpendicular to marker)
    offset = np.array([0.0, 0.0, -0.1])  # In meters

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Undistorted frame to remove lens distortion
        frame_undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)

        # Detect ArUco markers in the frame
        corners, ids, _ = detector.detectMarkers(frame_undistorted)

        if len(corners) > 0:
            for i, corner in enumerate(corners):
                rvecs, tvecs = pose_estimation(corner, marker_size, cameraMatrix, distCoeffs)

                if rvecs is not None and tvecs is not None:
                    # Draw the stylus tip as a red dot
                    draw_tip_as_dot(frame, rvecs, tvecs, cameraMatrix, distCoeffs, offset)

                    # Display marker information
                    display_marker_info(frame, rvecs, tvecs, font)

        # Display the processed frame
        cv2.imshow("Frame", frame)

        # Break loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
