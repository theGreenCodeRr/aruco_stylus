import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque


def load_camera_calibration():
    """Loads pre-calibrated camera matrix and distortion coefficients."""
    # These values are specific to the camera used to record the video.
    cameraMatrix = np.array(
        [[1611.8128831241706, 0.0, 977.2917583698485],
         [0.0, 1610.844939123866, 631.6041268141287],
         [0.0, 0.0, 1.0]],
        dtype='double')

    distCoeffs = np.array(
        [[0.1470379693082242],
         [-0.8792009505348446],
         [0.005816443190955909],
         [-0.0013223644896107078],
         [2.1420269472934077]],
        dtype='double')
    return cameraMatrix, distCoeffs


def setup_video_file(path):
    """Opens a video file and returns the capture object and FPS."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video opened: {width:.0f}x{height:.0f} @ {fps:.2f} FPS")
    return cap, fps


def setup_aruco():
    """Initializes the ArUco detector with specific parameters."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector


def setup_kalman_filter():
    """
    Initializes and configures a Kalman filter for 2D point tracking.
    The state is [x, y, vx, vy] - position and velocity.
    The measurement is [x, y] - position.
    """
    kf = cv2.KalmanFilter(4, 2)
    # Measurement Matrix (H): Maps state to measurement
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # Transition Matrix (A): Models the state evolution (constant velocity model)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # Process Noise Covariance (Q): Uncertainty in the model.
    # Allows the filter to adapt to changes in velocity.
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    # Measurement Noise Covariance (R): Uncertainty in the measurement.
    # A balanced value to provide good smoothing without excessive lag.
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8.0

    # Initial Error Covariance
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1
    # Initial State
    kf.statePost = np.zeros(4, dtype=np.float32)
    return kf


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    """
    Estimates the camera's pose relative to the global coordinate system
    defined by all visible markers using RANSAC for robustness.
    """
    if len(model_pts) < 4:
        return None, None
    try:
        _, rvec, tvec, _ = cv2.solvePnPRansac(np.array(model_pts), np.array(img_pts), cameraMatrix, distCoeffs)
        return rvec, tvec
    except cv2.error:
        return None, None


def main():
    """Main function to run the video processing and pose estimation loop."""
    video_path = 'video/c.mp4'
    cap, fps = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector = setup_aruco()
    kf = setup_kalman_filter()

    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
    except FileNotFoundError:
        print("Error: 'markers/model_points_4x4.csv' not found.")
        return

    pts = data[['x', 'y', 'z']].values.tolist()
    model_pts_by_id = [pts[i:i + 4] for i in range(0, len(pts), 4)]

    # --- Trajectory and Pen Tip Setup ---
    global_origin_pts = deque(maxlen=1000)
    pen_tip_path_filtered = deque(maxlen=1000)
    first_detection = True
    marker_was_visible = False

    # --- Define Pen Tip Location ---
    pen_tip_loc_mm = np.array([[-0.02327], [-102.2512], [132.8306]])
    pen_tip_3d = pen_tip_loc_mm.reshape(1, 1, 3)

    print(f"Using final extended pen tip location (mm):")
    print(f"  X={pen_tip_loc_mm[0][0]:.4f}, Y={pen_tip_loc_mm[1][0]:.4f}, Z={pen_tip_loc_mm[2][0]:.4f}")
    print(f"  Total length from center: {np.linalg.norm(pen_tip_loc_mm):.2f} mm")

    # --- Playback and Mode Control ---
    playing = True
    direction = 1
    plot_pen_tip = False
    ret, last_frame = cap.read()
    if not ret:
        print("Could not read the first frame.")
        return

    # --- Create a persistent canvas for drawing trajectories ---
    trajectory_canvas = np.zeros_like(last_frame)

    while True:
        if playing:
            if direction == 1:
                ret, frame = cap.read()
                if not ret:
                    playing = False
                    frame = last_frame.copy()
                else:
                    last_frame = frame.copy()
            else:
                current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_idx = max(int(current_frame_idx) - 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                ret, frame = cap.read()
                if ret:
                    last_frame = frame.copy()
                else:
                    playing = False
                    frame = last_frame.copy()
        else:
            frame = last_frame.copy()

        # --- ArUco Detection and Pose Estimation ---
        corners, ids, _ = detector.detectMarkers(frame)
        kf.predict()
        pose_found_this_frame = False

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            image_points_collected = []
            model_points_collected = []

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                if marker_id < len(model_pts_by_id):
                    image_points_collected.extend(corner[0])
                    model_points_collected.extend(model_pts_by_id[marker_id])

            r_glob, t_glob = estimatePoseGlobal(model_points_collected, image_points_collected, cameraMatrix,
                                                distCoeffs)

            if r_glob is not None and t_glob is not None:
                pose_found_this_frame = True
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 30, 4)

                origin_2d, _ = cv2.projectPoints(np.array([[0., 0., 0.]]), r_glob, t_glob, cameraMatrix, distCoeffs)
                global_origin_pts.append(tuple(origin_2d[0][0].astype(int)))
                if len(global_origin_pts) > 1:
                    cv2.line(trajectory_canvas, global_origin_pts[-2], global_origin_pts[-1], (0, 0, 255), 5)

                if plot_pen_tip:
                    pen_tip_2d, _ = cv2.projectPoints(pen_tip_3d, r_glob, t_glob, cameraMatrix, distCoeffs)
                    raw_pt = tuple(pen_tip_2d[0][0].astype(int))

                    # If marker was lost, treat this as a new first detection to avoid connecting paths
                    if not marker_was_visible:
                        first_detection = True

                    if first_detection:
                        kf.statePost = np.array([raw_pt[0], raw_pt[1], 0, 0], dtype=np.float32)
                        filtered_pt = raw_pt
                        first_detection = False
                    else:
                        measurement = np.array(raw_pt, dtype=np.float32).reshape(2, 1)
                        corrected_state = kf.correct(measurement)
                        filtered_pt = (int(corrected_state[0]), int(corrected_state[1]))

                    pen_tip_path_filtered.append(filtered_pt)
                    cv2.circle(frame, filtered_pt, 6, (255, 0, 0), -1)
                    if len(pen_tip_path_filtered) > 1 and marker_was_visible:
                        cv2.line(trajectory_canvas, pen_tip_path_filtered[-2], pen_tip_path_filtered[-1], (0, 255, 0),
                                 5)

        # Update the visibility status for the next frame
        marker_was_visible = pose_found_this_frame

        # --- Combine frame with the trajectory canvas ---
        img2gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        lines_fg = cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask)
        display_frame = cv2.add(frame_bg, lines_fg)

        # --- UI Text Overlay ---
        cv2.rectangle(display_frame, (0, 0), (450, 90), (0, 0, 0), -1)
        cv2.putText(display_frame, 'Global Origin Trajectory: Red', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        tip_color = (0, 255, 0) if plot_pen_tip else (128, 128, 128)
        cv2.putText(display_frame, f"Pen Tip: {'ON' if plot_pen_tip else 'OFF'} ('p' to toggle)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tip_color, 2)
        cv2.putText(display_frame, "Filtered Trajectory (Blue)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    2)

        # --- Display and Handle Keyboard Input ---
        cv2.imshow('Global Pose Estimation with Pen Tip', display_frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('f'):
            direction = 1
            playing = True
        elif key == ord('r'):
            direction = -1
            playing = True
        elif key == ord('p'):
            plot_pen_tip = not plot_pen_tip
            if not plot_pen_tip:
                pen_tip_path_filtered.clear()
                trajectory_canvas[:] = 0
                for i in range(1, len(global_origin_pts)):
                    cv2.line(trajectory_canvas, global_origin_pts[i - 1], global_origin_pts[i], (0, 0, 255), 5)
                kf = setup_kalman_filter()
                first_detection = True
                marker_was_visible = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
