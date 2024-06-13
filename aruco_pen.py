import cv2
import math
from cv2 import aruco
import numpy as np


def decimal_round(value, digit):
    value_multiply = value * 10 ** digit
    value_float = value_multiply.astype(int) / (10 ** digit)
    return value_float


def estimatePose(corners, marker_size, mtx, distortion):
    rvecs, tvecs = [], []
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    for corner in corners:
        retval, rvec, tvec = cv2.solvePnP(marker_points, corner, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


def main():
    cap = cv2.VideoCapture(0)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    marker_size = 0.017  # in meter

    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, parameters)

    # cameraMatrix & dist for my lab camera
    cameraMatrix = np.array([
        [763.43512892, 0, 321.85994173],
        [0, 764.52495998, 187.09227291],
        [0, 0, 1]],
        dtype='double', )
    distCoeffs = np.array([[0.13158662], [0.26274676], [-0.00894502], [-0.0041256], [-0.12036324]])

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if frame capture fails

        # Detect Aruco markers in the frame
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

        # Draw the detected markers
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

        # Check if at least one marker has been detected
        if ids is not None:
            # Estimate the pose of each marker
            rvecs, tvecs = estimatePose(corners, marker_size, cameraMatrix, distCoeffs)

            # Calculate the central point among all detected markers
            central_tvec = np.mean(np.array(tvecs), axis=0)

            # Draw the 3D axis at the central point
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[0], central_tvec, marker_size / 2)

        # Display the frame with the drawn markers and 3D axis
        cv2.imshow('AR marker', frame)

        # Break the loop if the ESC key is pressed
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
