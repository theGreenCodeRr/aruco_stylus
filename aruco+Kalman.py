import cv2
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video opened: {width}x{height} @ {fps:.2f} FPS")
    return cap, fps, width, height


def setup_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
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
    _, rvec, tvec = cv2.solvePnP(marker_pts, corner, cameraMatrix, distCoeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    obj = model_pts.reshape(-1,1,3).astype(np.float32)
    img = img_pts.reshape(-1,1,2).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(obj, img, cameraMatrix, distCoeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec

# Helper: Convert rvec → (pitch, yaw, roll) in degrees (ZYX convention)
def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = 0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def main():
    # Parameters
    video_path = 'video/cam0_1080p30.mkv'
    local_thickness = 3
    global_thickness = 7
    start_frame = 1
    end_frame = 262

    # Setup
    cap, fps, width, height = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size = setup_aruco()

    # Load 3D model points (one set of 4 points per marker)
    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values
    model_pts_list = np.split(pts, len(pts)//4)

    # Prepare structures for trajectories
    global_origin_pts = deque()
    local_origin_pts = defaultdict(deque)

    # Prepare lists to log every pose
    local_records = []   # per-marker, per-frame local poses
    global_records = []  # per-frame global poses

    # Initialize Kalman filter for global translation
    kf = cv2.KalmanFilter(6, 3)
    dt = 1.0 / fps if fps > 0 else 1.0
    kf.transitionMatrix = np.array([
        [1,0,0, dt,0, 0],
        [0,1,0, 0, dt,0],
        [0,0,1, 0, 0, dt],
        [0,0,0, 1, 0, 0 ],
        [0,0,0, 0, 1, 0 ],
        [0,0,0, 0, 0, 1 ],
    ], dtype=np.float32)
    kf.measurementMatrix = np.hstack([np.eye(3), np.zeros((3,3))]).astype(np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.statePost = np.zeros((6,1), dtype=np.float32)

    # Seek to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    # Playback control
    playing = True
    direction = 1

    while True:
        # Read / rewind / pause logic
        if playing:
            if direction == 1:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx-2, start_frame-1))
                ret, frame = cap.read()
                if not ret:
                    break
        else:
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx < start_frame or frame_idx > end_frame:
            break

        # Compute logged frame index starting at 1
        logged_frame = frame_idx - (start_frame - 1)

        # Overlay UI text
        cv2.rectangle(frame, (0,0), (300,60), (0,0,0), -1)
        cv2.putText(frame, 'Global: Red', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, 'Local: Green', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        img_pts, obj_pts = [], []
        r_glob, t_raw = None, None
        t_filt = None

        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                # Local pose estimation
                r_loc, t_loc = estimatePoseLocal(corner[0], marker_size, cameraMatrix, distCoeffs)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_loc, t_loc, 0.01, local_thickness)

                # Convert r_loc to Euler
                pitch_loc, yaw_loc, roll_loc = rvec_to_euler(r_loc)
                tx_loc, ty_loc, tz_loc = t_loc.ravel().tolist()

                # Log local pose using logged_frame
                local_records.append({
                    'frame': logged_frame,
                    'marker_id': int(mid),
                    'tx': float(tx_loc),
                    'ty': float(ty_loc),
                    'tz': float(tz_loc),
                    'pitch': float(pitch_loc),
                    'yaw': float(yaw_loc),
                    'roll': float(roll_loc),
                })

                # Draw Euler above marker
                top_left = tuple(corner[0][0].astype(int))
                cv2.putText(frame, f"P:{pitch_loc:.1f} Y:{yaw_loc:.1f} R:{roll_loc:.1f}",
                            (top_left[0], top_left[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

                # Draw local trajectory directly on frame
                orig2d, _ = cv2.projectPoints(np.zeros((1,3)), r_loc, t_loc, cameraMatrix, distCoeffs)
                pt2d = tuple(orig2d.ravel().astype(int))
                local_origin_pts[mid].append(pt2d)
                for p0, p1 in zip(local_origin_pts[mid], list(local_origin_pts[mid])[1:]):
                    cv2.line(frame, p0, p1, (0,255,0), local_thickness)

                # Accumulate for global pose
                if mid < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[mid]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        # Global pose estimation (Aruco + Kalman)
        if len(obj_pts) >= 4:
            r_glob, t_raw = estimatePoseGlobal(np.array(obj_pts), np.array(img_pts), cameraMatrix, distCoeffs)
            # Kalman predict + correct on raw translation
            kf.predict()
            kf.correct(t_raw.astype(np.float32))
            t_filt = kf.statePost[:3]

        # Draw global pose and trajectory on frame
        if r_glob is not None and t_filt is not None:
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_filt, 0.05, global_thickness)

            pitch_g, yaw_g, roll_g = rvec_to_euler(r_glob)
            tx_g, ty_g, tz_g = t_filt.ravel().tolist()

            # Log global pose using logged_frame
            global_records.append({
                'frame': logged_frame,
                'tx': float(tx_g),
                'ty': float(ty_g),
                'tz': float(tz_g),
                'pitch': float(pitch_g),
                'yaw': float(yaw_g),
                'roll': float(roll_g),
            })

            # Draw global Euler at top right
            cv2.putText(frame, f"G P:{pitch_g:.1f} Y:{yaw_g:.1f} R:{roll_g:.1f}",
                        (width-300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # Draw global trajectory directly on frame
            og2d, _ = cv2.projectPoints(np.zeros((1,3)), r_glob, t_filt, cameraMatrix, distCoeffs)
            ptg = tuple(og2d.ravel().astype(int))
            global_origin_pts.append(ptg)
            for p0, p1 in zip(global_origin_pts, list(global_origin_pts)[1:]):
                cv2.line(frame, p0, p1, (0,0,255), global_thickness)

        # Show frame directly (no overlay blending)
        cv2.imshow('Aruco+Kalman Trajectories', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):      # pause / unpause
            playing = not playing
        elif key == ord('f'):    # forward
            direction = 1
            playing = True
        elif key == ord('r'):    # rewind
            direction = -1
            playing = True
        elif key == ord('q'):    # quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save logs to CSV
    if len(local_records) > 0:
        df_local = pd.DataFrame(local_records)
        df_local.to_csv('aruco_kalman_local.csv', index=False)
        print(f"Saved {len(df_local)} local‐pose rows → aruco_kalman_local.csv")

    if len(global_records) > 0:
        df_global = pd.DataFrame(global_records)
        df_global.to_csv('aruco_kalman_global.csv', index=False)
        print(f"Saved {len(df_global)} global‐pose rows → aruco_kalman_global.csv")

if __name__ == '__main__':
    main()
