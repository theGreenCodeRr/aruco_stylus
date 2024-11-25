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
        rvecs = rvecs.reshape((3, 1)) if rvecs is not None and rvecs.shape != (3, 1) else rvecs
        tvecs = tvecs.reshape((3, 1)) if tvecs is not None and tvecs.shape != (3, 1) else tvecs
    else:
        return None, None

    return rvecs, tvecs


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


def calculate_tip_point(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    point_local = np.array([[0, 0, -0.15]], dtype=np.float32)  # 150 mm below the marker
    point_global = R @ point_local.T + tvec  # Transform the point to the global frame
    return point_global.T[0]


def display_marker_info(frame, rvec, tvec, font, tip_point=None):
    # Marker Plane Data
    marker_angles = calculate_euler_angles(rvec)
    x, y, z = tvec[0][0] * 1000, tvec[1][0] * 1000, tvec[2][0] * 1000  # Convert to mm
    cv2.putText(frame, f"Marker Plane:", (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'X: {round(x, 2)} mm', (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Y: {round(y, 2)} mm', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Z: {round(z, 2)} mm', (10, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Yaw: {round(marker_angles[2], 2)} deg', (10, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Pitch: {round(marker_angles[0], 2)} deg', (10, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Roll: {round(marker_angles[1], 2)} deg', (10, 230), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tip Data
    if tip_point is not None:
        tip_x, tip_y, tip_z = tip_point * 1000  # Convert to mm
        tip_angles = marker_angles  # Tip shares the same orientation as the marker
        cv2.putText(frame, f"Tip [150mm(-z) ]:", (10, 270), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'X: {round(tip_x, 2)} mm', (10, 310), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Y: {round(tip_y, 2)} mm', (10, 340), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Z: {round(tip_z, 2)} mm', (10, 370), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Yaw: {round(tip_angles[2], 2)} deg', (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Pitch: {round(tip_angles[0], 2)} deg', (10, 430), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Roll: {round(tip_angles[1], 2)} deg', (10, 460), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


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
                    tip_point = calculate_tip_point(rvecs, tvecs)
                    display_marker_info(frame, rvecs, tvecs, font, tip_point=tip_point)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()