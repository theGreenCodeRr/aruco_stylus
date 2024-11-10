import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque
from utils import util_draw_custom  # Draw using PyQtGraph


def camera_setup():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # cameraMatrix & distCoeffs: global shutter camera
    cameraMatrix = np.array(
        [[505.1150576, 0, 359.14439401],
         [0, 510.33530166, 230.33963591],
         [0, 0, 1]],
        dtype='double')
    distCoeffs = np.array([[0.07632527], [0.15558049], [0.00234922], [0.00500232], [-0.46829062]])

    # # mac webcam
    # cameraMatrix = np.array(
    #     [
    #         [581.2490064821088, 0.0, 305.2885321972521],
    #         [0.0, 587.6316817762934, 288.9932758741485],
    #         [0.0, 0.0, 1.0], ],
    #     dtype='double')
    # distCoeffs = np.array(
    #     [
    #         [-0.31329614267146066],
    #         [0.8386295742029726],
    #         [-0.0024210244191179104],
    #         [0.016349338905846198],
    #         [-1.133637004544031], ],
    #     dtype='double')
    return cap, cameraMatrix, distCoeffs


def aruco_setup():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.016  # 16mm
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size

def poseEstimation(corner, marker_size, cameraMatrix, distCoeffs):  # for individual marker
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash, rvecs, tvecs = cv2.solvePnP(marker_points, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs, trash

def main():

    cap, cameraMatrix, distCoeffs = camera_setup()
    detector, marker_size = aruco_setup()

    while True:
        ret, frame = cap.read()
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

        for i, corner in enumerate( corners ):
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0,255,255))
            cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1)

        if ids is not None:
            for i in range( ids.size ):
                # Calculate pose
                rvecs, tvecs, trash = poseEstimation(corners[i], marker_size, cameraMatrix, distCoeffs)
                print(rvecs, tvecs, trash)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
