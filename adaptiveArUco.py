import cv2
import numpy as np
import pandas as pd
from collections import deque, defaultdict


def load_camera_calibration():
    # Logitec Procam @1080p
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


def is_square(corner_pts, tol_angle=40.0):
    for i in range(4):
        p0 = corner_pts[i]
        v1 = corner_pts[(i - 1) % 4] - p0
        v2 = corner_pts[(i + 1) % 4] - p0
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        if abs(angle - 90.0) > tol_angle:
            return False
    return True


def aspect_ratio_check(corner_pts, tol=0.4):
    pts = corner_pts.reshape(4, 2)
    rect = cv2.minAreaRect(pts)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False
    ratio = min(w, h) / max(w, h)
    return ratio >= (1 - tol)


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


def rvec_to_euler(rvec):  # Convert rvec → (pitch, yaw, roll) in degrees
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
    video_path = 'video/cam0_1080p30.mkv'
    local_thickness = 3
    global_thickness = 7
    start_frame = 1
    end_frame = 262

    cap, fps, width, height = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector, marker_size = setup_aruco()

    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values
    model_pts_list = np.split(pts, len(pts)//4)

    # Persistent storage
    global_origin_pts = deque()
    local_last = {}  # mid -> (x, y, frame)
    local_segments = defaultdict(list)  # mid -> list of ((x0,y0),(x1,y1))

    local_records = []
    global_records = []

    # Allow brief occlusions
    missing_counts = defaultdict(int)
    missing_threshold = 5

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    playing = True
    direction = 1

    while True:
        # Frame stepping
        if playing:
            if direction == 1:
                ret, frame = cap.read()
            else:
                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx-2, start_frame-1))
                ret, frame = cap.read()
            if not ret: break
        else:
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx < start_frame or frame_idx > end_frame:
            break
        logged_frame = frame_idx - (start_frame - 1)

        # Overlay UI
        cv2.rectangle(frame, (0,0), (400,60), (0,0,0), -1)
        cv2.putText(frame, 'Global: Red', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, 'Local: Green', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        corners, ids, _ = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Gather valid
        valid = []
        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                pts2d = corner[0]
                if is_square(pts2d) and aspect_ratio_check(pts2d):
                    valid.append((int(mid), pts2d))
        current_ids = {mid for mid, _ in valid}

        # Update missing-frame counters
        for mid in list(missing_counts):
            if mid not in current_ids:
                missing_counts[mid] += 1
            else:
                missing_counts[mid] = 0
        # Seed new markers
        for mid in current_ids:
            missing_counts.setdefault(mid, 0)
        # Prune old tracks
        for mid, cnt in list(missing_counts.items()):
            if cnt > missing_threshold:
                missing_counts.pop(mid)
                local_last.pop(mid, None)

        img_pts, obj_pts = [], []
        r_glob, t_raw = None, None

        # Process each valid marker
        for mid, pts2d in valid:
            r_loc, t_loc = estimatePoseLocal(pts2d, marker_size, cameraMatrix, distCoeffs)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_loc, t_loc, 0.01, local_thickness)
            pitch_loc, yaw_loc, roll_loc = rvec_to_euler(r_loc)
            tx_loc, ty_loc, tz_loc = t_loc.ravel()
            local_records.append({
                'frame': logged_frame, 'marker_id': mid,
                'tx': tx_loc, 'ty': ty_loc, 'tz': tz_loc,
                'pitch': pitch_loc, 'yaw': yaw_loc, 'roll': roll_loc
            })
            top_left = tuple(pts2d[0].astype(int))
            cv2.putText(frame, f"P:{pitch_loc:.1f} Y:{yaw_loc:.1f} R:{roll_loc:.1f}",
                        (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            # Compute origin point
            orig2d, _ = cv2.projectPoints(np.zeros((1,3)), r_loc, t_loc, cameraMatrix, distCoeffs)
            pt = tuple(orig2d.ravel().astype(int))
            # Add segment if consecutive without missing
            if mid in local_last and missing_counts[mid] == 0:
                prev_x, prev_y, prev_f = local_last[mid]
                if logged_frame - prev_f == 1:
                    local_segments[mid].append(((prev_x, prev_y), pt))
            local_last[mid] = (pt[0], pt[1], logged_frame)

            # Collect for global
            if mid < len(model_pts_list):
                for p2d, p3d in zip(pts2d, model_pts_list[mid]):
                    img_pts.append(p2d)
                    obj_pts.append(p3d)

        # Draw local trajectory segments
        for segs in local_segments.values():
            for (x0,y0), (x1,y1) in segs:
                cv2.line(frame, (x0,y0), (x1,y1), (0,255,0), local_thickness)

        # Global pose
        if len(obj_pts) >= 4:
            r_glob, t_raw = estimatePoseGlobal(np.array(obj_pts), np.array(img_pts), cameraMatrix, distCoeffs)  # noqa
        if r_glob is not None and t_raw is not None:
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_raw, 0.05, global_thickness)
            pitch_g, yaw_g, roll_g = rvec_to_euler(r_glob)
            tx_g, ty_g, tz_g = t_raw.ravel()
            global_records.append({
                'frame': logged_frame, 'tx': tx_g, 'ty': ty_g, 'tz': tz_g,
                'pitch': pitch_g, 'yaw': yaw_g, 'roll': roll_g
            })
            cv2.putText(frame, f"G P:{pitch_g:.1f} Y:{yaw_g:.1f} R:{roll_g:.1f}",
                        (width-300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            og2d, _ = cv2.projectPoints(np.zeros((1,3)), r_glob, t_raw, cameraMatrix, distCoeffs)
            global_origin_pts.append(tuple(og2d.ravel().astype(int)))
            for p0, p1 in zip(global_origin_pts, list(global_origin_pts)[1:]):
                cv2.line(frame, p0, p1, (0,0,255), global_thickness)

        # Display & controls
        cv2.imshow('QC Aruco Trajectories (No Kalman)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): playing = not playing
        elif key == ord('f'): direction=1; playing=True
        elif key == ord('r'): direction=-1; playing=True
        elif key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # Save CSVs
    if local_records:
        pd.DataFrame(local_records).to_csv('CSVs/adaptiveArUco_local.csv', index=False)
    if global_records:
        pd.DataFrame(global_records).to_csv('CSVs/adaptiveArUco_global.csv', index=False)


if __name__ == '__main__':
    main()
