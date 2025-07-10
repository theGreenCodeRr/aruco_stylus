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

    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
    except FileNotFoundError:
        print("Error: 'markers/model_points_4x4.csv' not found.")
        return

    pts = data[['x', 'y', 'z']].values.tolist()
    model_pts_by_id = [pts[i:i + 4] for i in range(0, len(pts), 4)]

    # --- Trajectory and Pen Tip Setup ---
    global_origin_pts = deque(maxlen=1000)
    pen_tip_path = deque(maxlen=1000)

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

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            image_points_collected = []
            model_points_collected = []

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                if marker_id < len(model_pts_by_id):
                    for point_2d in corner[0]:
                        image_points_collected.append(point_2d)
                    for point_3d in model_pts_by_id[marker_id]:
                        model_points_collected.append(point_3d)

            r_glob, t_glob = estimatePoseGlobal(model_points_collected, image_points_collected, cameraMatrix,
                                                distCoeffs)

            if r_glob is not None and t_glob is not None:
                # Draw the global coordinate system axes on the current frame
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 30, 4)

                # Project global origin and add to path
                origin_2d, _ = cv2.projectPoints(np.array([[0., 0., 0.]]), r_glob, t_glob, cameraMatrix, distCoeffs)
                global_origin_pts.append(tuple(origin_2d[0][0].astype(int)))
                # Draw the new segment of the origin path on the persistent canvas
                if len(global_origin_pts) > 1:
                    cv2.line(trajectory_canvas, global_origin_pts[-2], global_origin_pts[-1], (0, 0, 255), 5)

                # If enabled, project the pen tip and trace its path
                if plot_pen_tip:
                    pen_tip_2d, _ = cv2.projectPoints(pen_tip_3d, r_glob, t_glob, cameraMatrix, distCoeffs)
                    pen_pt = tuple(pen_tip_2d[0][0].astype(int))
                    pen_tip_path.append(pen_pt)
                    # Draw the current pen tip location as a circle on the frame
                    cv2.circle(frame, pen_pt, 5, (0, 255, 0), -1)
                    # Draw the new segment of the pen tip path on the persistent canvas
                    if len(pen_tip_path) > 1:
                        cv2.line(trajectory_canvas, pen_tip_path[-2], pen_tip_path[-1], (0, 255, 0), 5)

        # --- Combine frame with the trajectory canvas using a mask for full opacity ---
        # Create a mask where the trajectory lines are (non-black pixels)
        img2gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the lines in the video frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Get only the trajectory lines from the canvas
        lines_fg = cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask)

        # Add the trajectory lines to the blacked-out frame to get the final result
        display_frame = cv2.add(frame_bg, lines_fg)

        # --- UI Text Overlay ---
        # Draw this on top of the combined image so it's always visible
        cv2.rectangle(display_frame, (0, 0), (350, 65), (0, 0, 0), -1)
        cv2.putText(display_frame, 'Global Origin Trajectory: Red', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        tip_color = (0, 255, 0) if plot_pen_tip else (128, 128, 128)
        cv2.putText(display_frame, f"Pen Tip Trajectory: {'ON' if plot_pen_tip else 'OFF'} ('p' to toggle)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tip_color, 2)

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
            # If turning off, clear the path and redraw the canvas without the green line
            if not plot_pen_tip:
                pen_tip_path.clear()
                # Redraw the canvas with only the red trajectory
                trajectory_canvas[:] = 0
                for i in range(1, len(global_origin_pts)):
                    cv2.line(trajectory_canvas, global_origin_pts[i - 1], global_origin_pts[i], (0, 0, 255), 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
