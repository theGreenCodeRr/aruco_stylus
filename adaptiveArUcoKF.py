import cv2
import numpy as np
import pandas as pd
from collections import deque, defaultdict

def load_camera_calibration():
    # Logitech Procam @1080p
    cameraMatrix = np.array([
        [544.191261501094,   0.0,               938.375784412138],
        [0.0,                539.5886057166153, 490.7524375321356],
        [0.0,                0.0,               1.0],
    ], dtype='double')
    distCoeffs = np.array([
        [ 0.0929139556606537],
        [-0.09051659316149255],
        [-0.0026022568575366028],
        [-0.00010257374456200485],
        [ 0.043047517532135635],
    ], dtype='double')
    return cameraMatrix, distCoeffs

def setup_video_file(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video opened: {width}x{height} @ {fps:.2f} FPS")
    return cap, fps, width, height

def setup_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params     = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector   = cv2.aruco.ArucoDetector(dictionary, params)
    marker_size = 0.016  # meters
    return detector, marker_size

def is_square(corner_pts, tol_angle=40.0):
    for i in range(4):
        p0 = corner_pts[i]
        v1 = corner_pts[(i - 1) % 4] - p0
        v2 = corner_pts[(i + 1) % 4] - p0
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle   = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        if abs(angle - 90.0) > tol_angle:
            return False
    return True

def aspect_ratio_check(corner_pts, tol=0.4):
    pts = corner_pts.reshape(4,2)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if w == 0 or h == 0:
        return False
    return min(w,h) / max(w,h) >= (1 - tol)

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

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = 0
    else:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[1,0], R[0,0])
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def create_translation_kf(dt):
    kf = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([
        [1,0,0, dt,0,0],
        [0,1,0, 0,dt,0],
        [0,0,1, 0,0,dt],
        [0,0,0, 1,0,0],
        [0,0,0, 0,1,0],
        [0,0,0, 0,0,1],
    ], np.float32)
    kf.measurementMatrix = np.hstack([np.eye(3), np.zeros((3,3))]).astype(np.float32)
    kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.errorCovPost        = np.eye(6, dtype=np.float32)
    return kf

def create_angle_kf(dt):
    kf = cv2.KalmanFilter(2, 1, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([[1, dt],
                                    [0,  1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0]], np.float32)
    kf.processNoiseCov     = np.eye(2, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(1, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(2, dtype=np.float32)
    return kf

def main():
    video_path      = 'video/cam0_1080p30.mkv'
    start_frame     = 1
    end_frame       = 262
    local_thick     = 3
    global_thick    = 7

    cap, fps, width, height = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size    = setup_aruco()

    # load model points (4 corners per marker)
    data = pd.read_csv('markers/model_points_4x4.csv')
    pts  = data[['x','y','z']].values
    model_pts_list = np.split(pts, len(pts)//4)

    dt = 1.0 / fps
    # global Kalman filters for pose
    kf_t     = create_translation_kf(dt)
    kf_pitch = create_angle_kf(dt)
    kf_yaw   = create_angle_kf(dt)
    kf_roll  = create_angle_kf(dt)
    kf_initialized = False

    # storage
    local_last     = {}
    local_segs     = defaultdict(list)
    missing_counts = defaultdict(int)
    missing_thr    = 5

    local_records  = []
    global_records = []
    global_origins = deque()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx < start_frame or frame_idx > end_frame:
            break
        logged_frame = frame_idx - (start_frame - 1)

        # UI overlay
        cv2.rectangle(frame, (0,0), (400,60), (0,0,0), -1)
        cv2.putText(frame, 'Global: Red', (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, 'Local: Green', (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        valid = []
        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                pts2d = corner[0]
                if is_square(pts2d) and aspect_ratio_check(pts2d):
                    valid.append((int(mid), pts2d))

        current_ids = {m for m,_ in valid}
        # update missing counts
        for m in list(missing_counts):
            missing_counts[m] += (m not in current_ids)
        for m in current_ids:
            missing_counts.setdefault(m, 0)
        for m, cnt in list(missing_counts.items()):
            if cnt > missing_thr:
                missing_counts.pop(m)
                local_last.pop(m, None)

        # local pose (raw)
        for mid, pts2d in valid:
            r_loc, t_loc = estimatePoseLocal(pts2d, marker_size, cameraMatrix, distCoeffs)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_loc, t_loc, 0.01, local_thick)
            tx, ty, tz = t_loc.ravel()
            p_loc, y_loc, r_loc_e = rvec_to_euler(r_loc)
            local_records.append({
                'frame': logged_frame, 'marker_id': mid,
                'tx': tx, 'ty': ty, 'tz': tz,
                'pitch': p_loc, 'yaw': y_loc, 'roll': r_loc_e
            })
            top = tuple(pts2d[0].astype(int))
            cv2.putText(frame, f"P:{p_loc:.1f} Y:{y_loc:.1f} R:{r_loc_e:.1f}",
                        (top[0], top[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            proj, _ = cv2.projectPoints(np.zeros((1,3)), r_loc, t_loc, cameraMatrix, distCoeffs)
            pt2d = tuple(proj.ravel().astype(int))
            if mid in local_last and missing_counts[mid] == 0:
                px, py, pf = local_last[mid]
                if logged_frame - pf == 1:
                    local_segs[mid].append(((px,py), pt2d))
            local_last[mid] = (pt2d[0], pt2d[1], logged_frame)

        for segs in local_segs.values():
            for (x0,y0),(x1,y1) in segs:
                cv2.line(frame, (x0,y0), (x1,y1), (0,255,0), local_thick)

        # build global PnP inputs from raw corners
        img_pts, obj_pts = [], []
        for mid, pts2d in valid:
            if mid < len(model_pts_list):
                for p2d, p3d in zip(pts2d, model_pts_list[mid]):
                    img_pts.append(p2d)
                    obj_pts.append(p3d)

        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(np.array(obj_pts), np.array(img_pts),
                                                cameraMatrix, distCoeffs)
            txg, tyg, tzg = t_glob.ravel()
            p_g, y_g, r_g  = rvec_to_euler(r_glob)

            # initialize KF state on first detection
            if not kf_initialized:
                kf_t.statePost[:3,0]   = np.array([txg, tyg, tzg], np.float32)
                kf_pitch.statePost[0,0] = np.float32(p_g)
                kf_yaw.statePost[0,0]   = np.float32(y_g)
                kf_roll.statePost[0,0]  = np.float32(r_g)
                kf_initialized = True

            # predict & correct translation
            kf_t.predict()
            kf_t.correct(np.array([[txg],[tyg],[tzg]], np.float32))
            tx_s, ty_s, tz_s = kf_t.statePost[:3,0]

            # predict & correct angles
            for kf, meas in [(kf_pitch, p_g), (kf_yaw, y_g), (kf_roll, r_g)]:
                kf.predict()
                kf.correct(np.array([[np.float32(meas)]], np.float32))
            p_s = kf_pitch.statePost[0,0]
            y_s = kf_yaw.statePost[0,0]
            r_s = kf_roll.statePost[0,0]

            # draw & record smoothed global pose
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs,
                              r_glob, np.array([[tx_s],[ty_s],[tz_s]]),
                              0.05, global_thick)
            global_records.append({
                'frame': logged_frame,
                'tx': float(tx_s), 'ty': float(ty_s), 'tz': float(tz_s),
                'pitch': float(p_s), 'yaw': float(y_s), 'roll': float(r_s)
            })
            cv2.putText(frame, f"G P:{p_s:.1f} Y:{y_s:.1f} R:{r_s:.1f}",
                        (width-300,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            proj, _ = cv2.projectPoints(np.zeros((1,3)), r_glob,
                                        np.array([[tx_s],[ty_s],[tz_s]]),
                                        cameraMatrix, distCoeffs)
            pt0 = tuple(proj.ravel().astype(int))
            global_origins.append(pt0)
            for prev, curr in zip(global_origins, list(global_origins)[1:]):
                cv2.line(frame, prev, curr, (0,0,255), global_thick)

        cv2.imshow('Aruco Trajectories (KF on global pose)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # save CSVs
    if local_records:
        pd.DataFrame(local_records).to_csv('CSVs/adaptiveArUcoKF_local.csv', index=False)
    if global_records:
        pd.DataFrame(global_records).to_csv('CSVs/adaptiveArUcoKF_global.csv', index=False)

if __name__ == '__main__':
    main()
