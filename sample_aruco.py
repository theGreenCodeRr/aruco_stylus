#!/usr/bin/env python
# -*- coding: utf-8 -*
import cv2
from cv2 import aruco
import numpy as np
# import math

cap = cv2.VideoCapture(0)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
marker_size = 0.05  # in meter

# CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# cameraMatrix & dist for my lab camera
cameraMatrix = np.array(
    [
        [1075.3188717662906, 0.0, 315.01708041919454],
        [0.0, 1072.0190163009527, 248.65205121389417],
        [0.0, 0.0, 1.0],
    ], dtype='double')

distCoeffs = np.array(
    [
        [2.0385351431950802],
        [-48.15234700186453],
        [-0.0028814706039823586],
        [-0.002306682662702244],
        [449.20242120647697],
    ], dtype='double')

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def decimal_round(value, digit):
    value_multiply = value * 10 ** digit
    value_float    = value_multiply.astype(int)/( 10 ** digit)
    return value_float


def estimatePose(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    i = 0
    for c in corners:
        trash, rvecs, tvecs = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return rvecs, tvecs, trash

def main():
    ret, frame = cap.read()
    while ret == True:
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
                rvecs_new, tvecs_new, _objPoints = estimatePose(corners, marker_size, cameraMatrix, distCoeffs)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rvecs_new)

                # Calculate 3D marker coordinates (m)
                marker_loc = np.zeros((3, 1), dtype=np.float64)
                marker_loc_world = rotation_matrix @ marker_loc + tvecs_new
                cv2.putText(frame, 'x : ' + str(decimal_round(marker_loc_world[0]*100,2)) + ' cm', (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame, 'y : ' + str(decimal_round(marker_loc_world[1]*100,2)) + ' cm', (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame, 'z : ' + str(decimal_round(marker_loc_world[2]*100,2)) + ' cm', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs_new, tvecs_new, 0.1)

        cv2.imshow('org', frame)
        key = cv2.waitKey(50)
        if key == 27: # ESC
            break
        # ret, frame = cap.read()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass