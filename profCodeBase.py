import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque
from utils import util_draw_custom  # Draw using PyQtGraph


def setup_web_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # cameraMatrix & distCoeffs: logitech camera
    cameraMatrix = np.array([
        [544.191261501094, 0.0, 938.375784412138],
        [0.0, 539.5886057166153, 490.75243759671406],
        [0.0, 0.0, 1.0],
    ], dtype='double')
    distCoeffs = np.array([
        [0.0929139556606537],
        [-0.09051659316149255],
        [-0.0026022568575366028],
        [-0.00010257374456200485],
        [0.043047517532135635],
    ], dtype='double')

    return cap, cameraMatrix, distCoeffs


def setup_aruco():
    # Set up the Aruco dictionary
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.016  # 16mm
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size


def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):  # for individual marker
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash, rvecs, tvecs = cv2.solvePnP(marker_points, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs, trash


# Estimate the global information of the dodecahedron from detected markers
def estimatePoseGlobal(model_points, image_points, cameraMatrix, distCoeffs):
    trash, rvecs, tvecs = cv2.solvePnP(model_points, image_points, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs, trash


def main():
    cap, cameraMatrix, distCoeffs = setup_web_camera()
    ret, frame = cap.read()
    detector, marker_size = setup_aruco()
    # Read Dodecahedron 3D coordinates
    data = pd.read_csv('markers/model_points_4x4.csv')  # Modified annotation (adjusted to each marker)
    row, column = data.shape  # Check the number of row & column
    # Put data into a 2D-list
    cols_to_combine = ['x', 'y', 'z']
    model_points_2d_list = data[cols_to_combine].values.tolist()
    # Convert a 2D list to a 3D list using K-slice
    # initializing K-Slicing method
    K = 4  # Number of 2D data in a group
    tmp1 = iter(model_points_2d_list)
    tmp2 = [tmp1] * K
    model_points_3d_list = [list(ele) for ele in zip(*tmp2)]

    global_pose = True
    plot_pen_tip = False

    while ret == True:
        ret, frame = cap.read()
        # #frame = filterrinhIm(frame)
        # retq, thresh1 = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
        # cv2.imshow("test", thresh1)
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

        # Use deque to append (faster)
        image_points_2d = deque()
        model_points_3d = deque()
        for i, corner in enumerate(corners):

            # Draw polylines (edge)
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 255))

            # Draw marker IDs
            cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            # Draw marker corner no.
            for j in range(1, 4):
                cv2.putText(frame, str(j), points[j], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Calculate the local pose of each AR marker
            rvecs, tvecs, _objPoints = estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs)
            if not global_pose:
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs, tvecs, 0.01)
                overlay = frame.copy()
                # A filled line
                cv2.line(overlay, (20, 20), (250, 20), (0, 255, 0), 40)
                # Transparency factor.
                alpha = 0.4
                cv2.putText(frame, 'Mode: Local Pose Estimation', (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Calculate only for #ID < 12 (avoid wrong ID detection)
            if ids[i][0] < 12:
                # Collect image points
                for j in range(4):
                    image_points_2d.append(corner[0][j].tolist())
                # Collect model points
                for j in range(4):
                    model_points_3d.append(model_points_3d_list[ids[i][0]][j])

        if ids is not None:
            # Convert to numpy array
            tmp = np.array(image_points_2d)
            image_points = tmp[np.newaxis, :]
            model_points = np.array(model_points_3d)
            if len(model_points >= 4):
                rvecs_global, tvecs_global, _objPoints = estimatePoseGlobal(model_points, image_points, cameraMatrix,
                                                                            distCoeffs)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rvecs_global)

                # Calculate the global pose of the dodecahedron
                if global_pose:
                    # Draw Axis, Line, Text
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs_global, tvecs_global, 50)
                    overlay = frame.copy()
                    # A filled line
                    cv2.line(overlay, (20, 20), (250, 20), (255, 255, 0), 40)
                    # Transparency factor.
                    alpha = 0.4
                    cv2.putText(frame, 'Mode: Global Pose Estimation', (20, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Calculate and draw the trajectory of the 3D pen tip
                    if plot_pen_tip:
                        pen_tip_loc = np.array([[-0.014943], [-65.6512], [85.2906]])  # 3x1 array
                        pen_tip_loc_world = rotation_matrix @ pen_tip_loc + tvecs_global
                        pen_tip_loc_world = pen_tip_loc_world / 25
                        new_pen = np.transpose(pen_tip_loc_world)
                        # Below is just for visualization purposes
                        x = -1.5 * new_pen[0][0] - 10
                        y = -2.0 * new_pen[0][1]
                        z = 1.5 * new_pen[0][2] - 20
                        new_pen[0] = (x, z, y)
                        util_draw_custom.plot_dodecahedron(frame, new_pen, 20)
        global_pose, plot_pen_tip = util_draw_custom.draw_image(frame)
    cap.release()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
