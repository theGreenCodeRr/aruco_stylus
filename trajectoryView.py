import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque, defaultdict

def load_camera_calibration():
    # Camera intrinsics
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
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_size = 0.016  # meters
    return detector, marker_size


def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):
    marker_pts = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0],
    ], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(marker_pts, corner, cameraMatrix, distCoeffs,False, cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    obj = model_pts.reshape(-1,1,3).astype(np.float32)
    img = img_pts.reshape(-1,1,2).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(obj, img, cameraMatrix, distCoeffs,False, cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def main():
    # Video and ArUco setup
    video_path = 'video/cam0_1080p30.mkv'
    cap = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size = setup_aruco()

    # Load global model points (4 corners per marker)
    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values.tolist()
    model_pts_list = [pts[i:i+4] for i in range(0, len(pts), 4)]

    # Trajectory buffers
    global_origin_pts = deque()
    local_origin_pts = defaultdict(deque)  # per-marker

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw legend
        cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, 'Global trajectory: Red', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'Local trajectories: Color-coded with ID', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0,255,0))

        img_pts, obj_pts = [], []
        # Local pose drawing and tracking
        if ids is not None:
            for idx, corner in zip(ids.flatten(), corners):
                r_loc, t_loc = estimatePoseLocal(
                    corner[0], marker_size, cameraMatrix, distCoeffs
                )
                # Local axes (thin)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs,
                                   r_loc, t_loc, 0.01, 1)
                # Project local origin
                orig2d, _ = cv2.projectPoints(
                    np.array([[0.,0.,0.]]), r_loc, t_loc,
                    cameraMatrix, distCoeffs
                )
                x_l, y_l = orig2d[0][0].astype(int)
                local_origin_pts[idx].append((x_l, y_l))
                # Prepare for global pose
                if idx < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[idx]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        # Draw local trajectories
        for mid, buf in local_origin_pts.items():
            color = tuple(int(c) for c in np.random.randint(0,255,3))
            # Label the marker ID at the starting point of its trajectory
            if buf:
                x0, y0 = buf[0]
                cv2.putText(
                    frame,
                    f'ID:{mid}',
                    (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            # Draw the trajectory lines
            for i in range(1, len(buf)):
                cv2.line(frame, buf[i-1], buf[i], color, 2)

        # Global pose drawing and tracking
        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(
                np.array(obj_pts), np.array(img_pts),
                cameraMatrix, distCoeffs
            )
            # Global axes (bold)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs,
                               r_glob, t_glob, 0.05, 5)
            # Project global origin
            og2d, _ = cv2.projectPoints(
                np.array([[0.,0.,0.]]), r_glob, t_glob,
                cameraMatrix, distCoeffs
            )
            x_g, y_g = og2d[0][0].astype(int)
            global_origin_pts.append((x_g, y_g))
            for i in range(1, len(global_origin_pts)):
                cv2.line(frame, global_origin_pts[i-1],
                         global_origin_pts[i], (0,0,255), 2)

        # Display frame
        cv2.imshow('Local & Global Trajectories', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Keep window open after completion
    print("Playback finished. Press any key to close.")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
