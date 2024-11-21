import cv2
from cv2 import aruco
import numpy as np
import math


def camera_setup():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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


def calculate_euler_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
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
    angles = calculate_euler_angles(rvec)
    x, y, z = tvec[0][0] * 1000, tvec[1][0] * 1000, tvec[2][0] * 1000
    distance_rounded = round(z, 4)

    cv2.putText(frame, f'X: {round(x, 2)} mm', (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Y: {round(y, 2)} mm', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Z (Distance): {distance_rounded} mm', (10, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Yaw: {round(angles[2], 2)} deg', (10, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Pitch: {round(angles[0], 2)} deg', (10, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Roll: {round(angles[1], 2)} deg', (10, 230), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


def draw_detected_markers(frame, corners, ids):
    aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))


def draw_marker_axes(frame, cameraMatrix, distCoeffs, rvec, tvec):
    axis_length = 0.03
    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, axis_length)


def main():
    cap, cameraMatrix, distCoeffs = camera_setup()
    detector, marker_size = aruco_setup()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)
        corners, ids, _ = detector.detectMarkers(frame_undistorted)

        if len(corners) > 0:
            for i, corner in enumerate(corners):
                rvecs, tvecs = pose_estimation(corner, marker_size, cameraMatrix, distCoeffs)

                if rvecs is not None and tvecs is not None:
                    display_marker_info(frame, rvecs, tvecs, font)
                    draw_detected_markers(frame, corners, ids)
                    draw_marker_axes(frame, cameraMatrix, distCoeffs, rvecs, tvecs)

                    # Pen tip logic
                    pen_tip_loc = np.array([[-0.01], [-0.05], [0.1]])  # Offset of the pen tip
                    rotation_matrix, _ = cv2.Rodrigues(rvecs)
                    pen_tip_world = rotation_matrix @ pen_tip_loc + tvecs

                    # Project the pen tip onto the 2D image plane
                    pen_tip_image, _ = cv2.projectPoints(pen_tip_loc.T, rvecs, tvecs, cameraMatrix, distCoeffs)
                    pen_tip_image = pen_tip_image.reshape(-1, 2)
                    pen_tip_point = tuple(map(int, pen_tip_image[0]))
                    cv2.circle(frame, pen_tip_point, 5, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
