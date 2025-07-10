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
        [-marker_size/2, marker_size/2, 0],
        [ marker_size/2, marker_size/2, 0],
        [ marker_size/2,-marker_size/2, 0],
        [-marker_size/2,-marker_size/2, 0],
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


def create_corner_kf():
    # 4D state: [x, y, vx, vy], 2D measurement: [x, y]
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    # Increase process noise and decrease measurement noise
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


def main():
    video_path       = 'video/cam0_1080p30.mkv'
    start_frame      = 1
    end_frame        = 262
    local_thickness  = 3
    global_thickness = 7

    cap, fps, width, height = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size   = setup_aruco()

    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values
    model_pts_list = np.split(pts, len(pts)//4)

    global_origin_pts = deque()
    local_origin_pts  = defaultdict(deque)
    local_records     = []
    global_records    = []

    # Kalman filters
    corner_filters      = defaultdict(lambda: [create_corner_kf() for _ in range(4)])
    global_origin_kf    = create_corner_kf()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
    playing, direction = True, 1

    while True:
        # frame control logic
        if playing:
            if direction == 1:
                ret, frame = cap.read()
            else:
                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx-2, start_frame-1))
                ret, frame = cap.read()
        else:
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not (start_frame <= frame_idx <= end_frame):
            break
        logged_frame = frame_idx - (start_frame - 1)

        # draw legends
        cv2.rectangle(frame, (0,0), (300,60), (0,0,0), -1)
        cv2.putText(frame, 'Global: Red', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, 'Local: Green', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        img_pts, obj_pts = [], []
        r_glob, t_glob = None, None

        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                # smooth each corner in 2D
                for i in range(4):
                    raw = corner[0][i].astype(np.float32).reshape(2,1)
                    kf = corner_filters[mid][i]
                    kf.predict()
                    st = kf.correct(raw)
                    corner[0][i] = st[:2].ravel()

                # LOCAL POSE ESTIMATION
                r_loc, t_loc = estimatePoseLocal(corner[0], marker_size, cameraMatrix, distCoeffs)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_loc, t_loc, 0.01, local_thickness)
                p_loc, y_loc, r_lloc = rvec_to_euler(r_loc)
                tx, ty, tz = t_loc.ravel().tolist()
                local_records.append({'frame':logged_frame, 'marker_id':int(mid), 'tx':tx, 'ty':ty, 'tz':tz,
                                      'pitch':p_loc, 'yaw':y_loc, 'roll':r_lloc})
                # draw Euler angles
                tl = tuple(corner[0][0].astype(int))
                cv2.putText(frame, f"P:{p_loc:.1f} Y:{y_loc:.1f} R:{r_lloc:.1f}",
                            (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

                # local trajectory path
                o2d, _ = cv2.projectPoints(np.zeros((1,3)), r_loc, t_loc, cameraMatrix, distCoeffs)
                pt2d = tuple(o2d.ravel().astype(int))
                local_origin_pts[mid].append(pt2d)
                if len(local_origin_pts[mid]) > 1:
                    for a, b in zip(local_origin_pts[mid], list(local_origin_pts[mid])[1:]):
                        cv2.line(frame, a, b, (0,255,0), local_thickness)

                # prepare global PnP
                if mid < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[mid]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        # GLOBAL POSE ESTIMATION
        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(np.array(obj_pts), np.array(img_pts), cameraMatrix, distCoeffs)

        if r_glob is not None:
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 0.05, global_thickness)
            p_g, y_g, r_g = rvec_to_euler(r_glob)
            txg, tyg, tzg = t_glob.ravel().tolist()
            global_records.append({'frame':logged_frame, 'tx':txg, 'ty':tyg, 'tz':tzg,
                                   'pitch':p_g, 'yaw':y_g, 'roll':r_g})

            # filter global origin in 2D (optional)
            og2d, _ = cv2.projectPoints(np.zeros((1,3)), r_glob, t_glob, cameraMatrix, distCoeffs)
            rawg = og2d.ravel().astype(np.float32).reshape(2,1)
            global_origin_kf.predict()
            stg = global_origin_kf.correct(rawg)
            ptg = (int(stg[0,0]), int(stg[1,0]))
            global_origin_pts.append(ptg)
            if len(global_origin_pts) > 1:
                for a, b in zip(global_origin_pts, list(global_origin_pts)[1:]):
                    cv2.line(frame, a, b, (0,0,255), global_thickness)

        cv2.imshow('Kalman-smoothed Aruco', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            playing = not playing
        elif k == ord('f'):
            direction = 1; playing = True
        elif k == ord('r'):
            direction = -1; playing = True
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # save CSVs
    if local_records:
        pd.DataFrame(local_records).to_csv('CSVs/arucoKF_local.csv', index=False)
        print(f"Saved {len(local_records)} local poses → arucoKalman_local.csv")
    if global_records:
        pd.DataFrame(global_records).to_csv('CSVs/arucoKF_global.csv', index=False)
        print(f"Saved {len(global_records)} global poses → arucoKalman_global.csv")


if __name__ == '__main__':
    main()
