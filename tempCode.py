import numpy as np
import time
import cv2
from cv2 import aruco
import math

camera_matrix = np.array(
    [
        [581.2490064821088, 0.0, 305.2885321972521],
        [0.0, 587.6316817762934, 288.9932758741485],
        [0.0, 0.0, 1.0], ],
    dtype='double')
dist_matrix = np.array(
    [
        [-0.31329614267146066],
        [0.8386295742029726],
        [-0.0024210244191179104],
        [0.016349338905846198],
        [-1.133637004544031], ],
    dtype='double')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX  #font for displaying text (below)

while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    frame = dst1


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()


    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


    if ids is not None:

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        (rvec - tvec).any()  # get rid of that nasty numpy value array error


        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)


        deg = rvec[0][0][2] / math.pi * 180  #deg=rvec[0][0][2]/math.pi*180*90/104

        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec, R)
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

        rx = x * 180.0 / 3.141592653589793
        ry = y * 180.0 / 3.141592653589793
        rz = z * 180.0 / 3.141592653589793

        cv2.putText(frame, 'deg_z:' + str(ry) + str('deg'), (0, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)



        distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100


        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)



    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
