import cv2
from cv2 import aruco
import numpy as np
import pandas as pd

def camera_calibration():
    # Hard-coded camera intrinsics and distortion coefficients
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
    # Prepare ArUco detector
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, params)
    marker_size = 0.016  # Marker side length in meters
    return detector, marker_size


def estimatePoseLocal(corners, marker_size, cameraMatrix, distCoeffs):
    # 3D coordinates of the marker's corners in its local frame
    obj_pts = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0],
    ], dtype=np.float32)
    # Solve PnP for local pose
    _, rvec, tvec = cv2.solvePnP(
        obj_pts, corners, cameraMatrix, distCoeffs,
        useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rvec.flatten(), tvec.flatten()


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    # Flatten and reshape for solvePnP
    obj = model_pts.reshape(-1, 1, 3).astype(np.float32)
    img = img_pts.reshape(-1, 1, 2).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(
        obj, img, cameraMatrix, distCoeffs,
        useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rvec.flatten(), tvec.flatten()


def main():
    # Video file paths per camera
    paths = {
        'cam0': 'video/cam0_1080p30.mkv',
        'cam1': 'video/cam1_720p30.mkv',
        'cam2': 'video/cam2_720p30.mkv',
    }
    # Open VideoCaptures
    caps = {name: cv2.VideoCapture(p) for name, p in paths.items()}
    for name, cap in caps.items():
        if not cap.isOpened():
            raise IOError(f"Cannot open {name} at {paths[name]}")

    # Reference resolution and frame count from cam0
    w0 = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(caps['cam0'].get(cv2.CAP_PROP_FRAME_COUNT))

    # Camera calibration and ArUco setup
    cameraMatrix, distCoeffs = camera_calibration()
    detector, marker_size = setup_aruco()

    # Load 3D model points for each marker ID
    data = pd.read_csv('markers/model_points_4x4.csv')
    pts = data[['x', 'y', 'z']].values.tolist()
    model_pts_list = [np.array(pts[i:i+4], dtype=np.float32)
                      for i in range(0, len(pts), 4)]

    #frame_indices = np.linspace(0, total_frames - 1, num=50, dtype=int) # Sample 50 frames at even intervals

    frame_indices = range(total_frames) #all frames

    rows = []
    for idx in frame_indices:
        # Grab and resize frames from each camera
        frames = {}
        for name, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frm = cap.read()
            if not ret:
                # fallback blank frame if read fails
                frm = np.zeros((h0, w0, 3), np.uint8)
            elif name != 'cam0':
                frm = cv2.resize(frm, (w0, h0))
            frames[name] = frm

        # Show all camera windows until manual input
        for name, frm in frames.items():
            cv2.imshow(name, frm)
        cv2.waitKey(1)  # render windows

        # Get manual angle input
        angle = float(input(f"Frame {idx}: Enter protractor angle (deg): "))
        for name in frames.keys():
            cv2.destroyWindow(name)

        # Detect markers in cam0 image
        cam0 = frames['cam0']
        corners, ids, _ = detector.detectMarkers(cam0)

        img_pts, obj_pts = [], []
        if ids is not None:
            for mid, corner in zip(ids.flatten(), corners):
                # Estimate local pose
                r_loc, t_loc = estimatePoseLocal(
                    corner[0], marker_size, cameraMatrix, distCoeffs)
                # Record row: frame, angle, global placeholders, marker_id, local pose
                rows.append([
                    idx, angle,
                    np.nan, np.nan, np.nan,   # rglob_x, rglob_y, rglob_z
                    np.nan, np.nan, np.nan,   # tglob_x, tglob_y, tglob_z
                    mid,                       # marker_id
                    r_loc[0], r_loc[1], r_loc[2],  # rloc_x, rloc_y, rloc_z
                    t_loc[0], t_loc[1], t_loc[2]   # tx_local, ty_local, tz_local
                ])
                # Accumulate for global pose if model data exists
                if mid < len(model_pts_list):
                    for p2d, p3d in zip(corner[0], model_pts_list[mid]):
                        img_pts.append(p2d)
                        obj_pts.append(p3d)

        # Fuse global pose when enough correspondences
        if len(obj_pts) >= 4:
            r_glob, t_glob = estimatePoseGlobal(
                np.array(obj_pts), np.array(img_pts),
                cameraMatrix, distCoeffs)
            n = len(ids.flatten()) if ids is not None else 0
            # Backfill global pose for last n rows
            for i in range(1, n+1):
                rows[-i][2:8] = [*r_glob, *t_glob]

    # Save all annotations to CSV
    df = pd.DataFrame(rows, columns=[
        'frame', 'input_angle',
        'rglob_x', 'rglob_y', 'rglob_z',
        'tglob_x', 'tglob_y', 'tglob_z',
        'marker_id',
        'rloc_x', 'rloc_y', 'rloc_z',
        'tx_local', 'ty_local', 'tz_local'
    ])
    df.to_csv('6DoF_annotated_poses.csv', index=False)
    print("Saved >> 6DoF_annotated_poses.csv")

    # Clean up
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
