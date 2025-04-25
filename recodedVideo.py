import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque
from utils import util_draw_custom  # your PyQtGraph drawing helpers

def setup_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # your 4K camera intrinsics:
    cameraMatrix = np.array([
        [3407.377381171259,     0.0,              1669.7778384788598],
        [   0.0,            50386.562865022606,    2056.98385996533],
        [   0.0,                0.0,                 1.0]
    ], dtype=np.double)

    distCoeffs = np.array([
        [   5.074522846259332],
        [4105.311648494995   ],
        [  -0.06637971556428957],
        [  -3.164220698183084],
        [ -49.52106989042863]
    ], dtype=np.double)

    # pull FPS so we can sync display
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cap, cameraMatrix, distCoeffs, fps

def setup_aruco():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    marker_size = 0.016  # 16 mm
    return detector, marker_size

def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):
    model_pts = np.array([[-marker_size/2,  marker_size/2, 0],
                          [ marker_size/2,  marker_size/2, 0],
                          [ marker_size/2, -marker_size/2, 0],
                          [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
    _, rvecs, tvecs = cv2.solvePnP(model_pts, corner, cameraMatrix, distCoeffs,
                                   useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs

def estimatePoseGlobal(model_points, image_points, cameraMatrix, distCoeffs):
    _, rvecs, tvecs = cv2.solvePnP(model_points, image_points, cameraMatrix, distCoeffs,
                                   useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs

def main():
    video_path = 'videos/Exp1_4k.mp4'
    cap, cameraMatrix, distCoeffs, fps = setup_video(video_path)
    detector, marker_size = setup_aruco()

    # load your 3D model‑points for the dodecahedron
    df = pd.read_csv('markers/model_points_4x4.csv')
    pts3d = df[['x','y','z']].values.tolist()
    # group into per‑marker sets of 4:
    model_points_3d_list = [ pts3d[i:i+4] for i in range(0, len(pts3d), 4) ]

    global_pose = True
    plot_pen_tip = False

    delay = int(1000.0 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

        image_pts = deque()
        model_pts = deque()

        if ids is not None:
            for idx, corner in zip(ids.flatten(), corners):
                pts = corner[0].astype(int)
                # draw edges & IDs
                cv2.polylines(frame, [pts], True, (0,255,255), 2)
                cv2.putText(frame, str(idx), tuple(pts[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

                rvec, tvec = estimatePoseLocal(corner[0].astype(np.float32), marker_size,
                                               cameraMatrix, distCoeffs)
                if not global_pose:
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.01)
                    overlay = frame.copy()
                    cv2.line(overlay, (20,20), (250,20), (0,255,0), 40)
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                    cv2.putText(frame, 'Mode: Local Pose', (20,25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # only gather < 12 to avoid false positives
                if idx < len(model_points_3d_list):
                    for corner_pt, model_pt in zip(corner[0], model_points_3d_list[idx]):
                        image_pts.append(corner_pt.tolist())
                        model_pts.append(model_pt)

            if len(model_pts) >= 4:
                img_pts_np   = np.array(image_pts, dtype=np.float32)
                mdl_pts_np   = np.array(model_pts, dtype=np.float32)
                rvec_g, tvec_g = estimatePoseGlobal(mdl_pts_np, img_pts_np,
                                                    cameraMatrix, distCoeffs)
                rotm, _ = cv2.Rodrigues(rvec_g)

                if global_pose:
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec_g, tvec_g, 50)
                    overlay = frame.copy()
                    cv2.line(overlay, (20,20), (250,20), (255,255,0), 40)
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                    cv2.putText(frame, 'Mode: Global Pose', (20,25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    if plot_pen_tip:
                        pen_tip = np.array([[-0.014943],[-65.6512],[85.2906]])
                        world_pt = rotm @ pen_tip + tvec_g
                        world_pt = (world_pt / 25).flatten()
                        # swap axes for your util_draw_custom
                        util_draw_custom.plot_dodecahedron(frame,
                                           np.array([[-1.5*world_pt[0]-10,
                                                     -2.0*world_pt[1],
                                                      1.5*world_pt[2]-20]]),
                                           20)

        # toggle modes via your GUI hook
        global_pose, plot_pen_tip = util_draw_custom.draw_image(frame)

        cv2.imshow('pose-estimation', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
