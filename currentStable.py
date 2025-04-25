import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque
from utils import util_draw_custom  # Draw using PyQtGraph

def load_camera_calibration():
    # Camera intrinsic parameters and distortion coefficients
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
    return cameraMatrix, distCoeffs


def setup_video_file(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video opened: {width:.0f}x{height:.0f} @ {fps:.2f} FPS")
    return cap


def setup_aruco():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    marker_size = 0.016  # meters
    return detector, marker_size


def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):
    marker_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0],
    ], dtype=np.float32)
    _, rvecs, tvecs = cv2.solvePnP(
        marker_points, corner, cameraMatrix, distCoeffs,
        False, cv2.SOLVEPNP_ITERATIVE
    )
    return rvecs, tvecs


def estimatePoseGlobal(model_points, image_points, cameraMatrix, distCoeffs):
    obj_pts = model_points.reshape(-1, 1, 3).astype(np.float32)
    img_pts = image_points.reshape(-1, 1, 2).astype(np.float32)
    _, rvecs, tvecs = cv2.solvePnP(
        obj_pts, img_pts, cameraMatrix, distCoeffs,
        False, cv2.SOLVEPNP_ITERATIVE
    )
    return rvecs, tvecs


def main():
    video_path = 'video/cam0_1080p30.mkv'
    cap = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size = setup_aruco()

    # Load model points for global pose
    data = pd.read_csv('markers/model_points_4x4.csv')
    points = data[['x', 'y', 'z']].values.tolist()
    model_points_3d_list = [points[i:i+4] for i in range(0, len(points), 4)]

    global_pose = True
    origin_pts = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        corners, ids, _ = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0,255,0))

        img_pts = []
        obj_pts = []

        if ids is not None:
            for idx, corner in zip(ids.flatten(), corners):
                if idx < len(model_points_3d_list):
                    rvec, tvec = estimatePoseLocal(corner[0], marker_size, cameraMatrix, distCoeffs)
                    if not global_pose:
                        cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.01)

                    for pt2d, pt3d in zip(corner[0], model_points_3d_list[idx]):
                        img_pts.append(pt2d)
                        obj_pts.append(pt3d)

            if len(obj_pts) >= 4:
                img_np = np.array(img_pts, dtype=np.float32)
                obj_np = np.array(obj_pts, dtype=np.float32)
                rvec_g, tvec_g = estimatePoseGlobal(obj_np, img_np, cameraMatrix, distCoeffs)

                # Project the world origin to image
                origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                proj, _ = cv2.projectPoints(origin_3d, rvec_g, tvec_g, cameraMatrix, distCoeffs)
                x, y = proj[0][0].astype(int)

                origin_pts.append((x, y))
                # Draw trajectory on video frame
                for i in range(1, len(origin_pts)):
                    p0, p1 = origin_pts[i-1], origin_pts[i]
                    cv2.line(frame, p0, p1, (0, 255, 0), 2)
                # Draw current origin marker
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(frame, 'Origin', (x + 10, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display frame and handle GUI toggles
        global_pose, _ = util_draw_custom.draw_image(frame)

    cap.release()

if __name__ == '__main__':
    main()
