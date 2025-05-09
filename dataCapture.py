import cv2
from cv2 import aruco
import numpy as np
import pandas as pd

def camera_calibration():
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

def setup_aruco():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, params)
    marker_size = 0.016  # meters
    return detector, marker_size

def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):
    obj_pts = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0],
    ], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(
        obj_pts, corner, cameraMatrix, distCoeffs,
        False, cv2.SOLVEPNP_ITERATIVE
    )
    return rvec.flatten(), tvec.flatten()

def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    obj = model_pts.reshape(-1,1,3).astype(np.float32)
    img = img_pts.reshape(-1,1,2).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(
        obj, img, cameraMatrix, distCoeffs,
        False, cv2.SOLVEPNP_ITERATIVE
    )
    return rvec.flatten(), tvec.flatten()

def main():
    # Paths to videos
    paths = {
        'cam0': 'video/cam0_1080p30.mkv',
        'cam1': 'video/cam1_720p30.mkv',
        'cam2': 'video/cam2_720p30.mkv',
    }
    # Open captures
    caps = {name: cv2.VideoCapture(p) for name,p in paths.items()}
    for name, cap in caps.items():
        if not cap.isOpened():
            raise IOError(f"Cannot open {name} at {paths[name]}")

    # Get cam0 resolution & frame count
    w0 = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_COUNT))

    # Calibration & ArUco
    cameraMatrix, distCoeffs = camera_calibration()
    detector, marker_size = setup_aruco()

    # Load 3D model points
    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x','y','z']].values.tolist()
    model_pts_list = [np.array(pts[i:i+4], dtype=np.float32)
                      for i in range(0, len(pts), 4)]

    # Select 50 frames at even intervals
    frame_indices = np.linspace(0, total_frames - 1, num=50, dtype=int)

    rows = []
    for idx in frame_indices:
        # grab each camera frame at idx
        frames = {}
        for name, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frm = cap.read()
            if not ret:
                frm = np.zeros((h0, w0, 3), np.uint8)
            elif name != 'cam0':
                frm = cv2.resize(frm, (w0, h0))
            frames[name] = frm

        # display windows
        for name, frm in frames.items():
            cv2.imshow(name, frm)
        cv2.waitKey(100)

        # manual angle
        angle = float(input(f"Frame {idx}: Enter protractor angle (deg): "))

        # detect on cam0
        cam0 = frames['cam0']
        corners, ids, _ = detector.detectMarkers(cam0)
        img_pts, obj_pts = [], []

        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                r_loc, t_loc = estimatePoseLocal(
                    corner[0], marker_size, cameraMatrix, distCoeffs)
                rows.append([
                    idx, angle,
                    np.nan, np.nan, np.nan,  # placeholders for global
                    np.nan, np.nan, np.nan,
                    mid, *t_loc
                ])
                if mid < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[mid]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        # fused global
        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(
                np.array(obj_pts), np.array(img_pts),
                cameraMatrix, distCoeffs)
            n = len(ids.flatten()) if ids is not None else 0
            for i in range(1, n+1):
                rows[-i][2:8] = [*r_glob, *t_glob]

        # close windows
        for name in frames:
            cv2.destroyWindow(name)

    # save CSV
    df = pd.DataFrame(rows, columns=[
        'frame','input_angle',
        'rglob_x','rglob_y','rglob_z','tglob_x','tglob_y','tglob_z',
        'marker_id','tx_local','ty_local','tz_local'
    ])
    df.to_csv('annotated_poses.csv', index=False)
    print("Saved annotated_poses.csv")

    # cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
