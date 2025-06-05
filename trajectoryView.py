import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque, defaultdict


def load_camera_calibration():
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
    return cap, fps


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
    _, rvec, tvec = cv2.solvePnP(marker_pts, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    obj = model_pts.reshape(-1,1,3).astype(np.float32)
    img = img_pts.reshape(-1,1,2).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(obj, img, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def main():
    video_path = 'video/cam0_1080p30.mkv'
    cap, fps = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size = setup_aruco()

    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values.tolist()
    model_pts_list = [pts[i:i+4] for i in range(0, len(pts), 4)]

    global_origin_pts = deque()
    local_origin_pts = defaultdict(deque)

    # Playback control
    playing = True
    direction = 1  # 1 = forward, -1 = reverse
    ret, last_frame = cap.read()
    if not ret:
        print("Empty video.")
        return

    while True:
        if playing:
            if direction == 1:
                ret, frame = cap.read()
                if not ret:
                    # reached end, stay on last frame
                    playing = False
                    frame = last_frame.copy()
                else:
                    last_frame = frame.copy()
            else:
                # reverse one frame
                idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_idx = max(int(idx) - 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                ret, frame = cap.read()
                last_frame = frame.copy() if ret else last_frame.copy()
        else:
            frame = last_frame.copy()

        # --- ArUco detection and drawing ---
        cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, 'Global trajectory: Red', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'Local: Color-coded ID', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids)
        img_pts, obj_pts = [], []
        if ids is not None:
            for idx, corner in zip(ids.flatten(), corners):
                r_loc, t_loc = estimatePoseLocal(corner[0], marker_size, cameraMatrix, distCoeffs)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_loc, t_loc, 0.01, 1)
                orig2d, _ = cv2.projectPoints(np.array([[0.,0.,0.]]), r_loc, t_loc, cameraMatrix, distCoeffs)
                local_origin_pts[idx].append(tuple(orig2d[0][0].astype(int)))
                if idx < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[idx]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        for buf in local_origin_pts.values():
            for i in range(1, len(buf)):
                cv2.line(frame, buf[i-1], buf[i], (0,255,0), 2)

        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(np.array(obj_pts), np.array(img_pts), cameraMatrix, distCoeffs)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 0.05, 5)
            og2d, _ = cv2.projectPoints(np.array([[0.,0.,0.]]), r_glob, t_glob, cameraMatrix, distCoeffs)
            pt = tuple(og2d[0][0].astype(int))
            global_origin_pts.append(pt)
            for i in range(1, len(global_origin_pts)):
                cv2.line(frame, global_origin_pts[i-1], global_origin_pts[i], (0,0,255), 2)

        # Display and handle keys
        cv2.imshow('Local & Global Trajectories', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space: toggle play/pause
            playing = not playing
        elif key == ord('f'):  # f: play forward
            direction = 1
            playing = True
        elif key == ord('r'):  # r: play in reverse
            direction = -1
            playing = True
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
