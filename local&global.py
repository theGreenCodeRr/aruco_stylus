import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque


def load_camera_calibration():
    # Logitech Procam @1080p
    cameraMatrix = np.array([
        [544.191261501094, 0.0, 938.375784412138],
        [0.0, 539.5886057166153, 490.7524375321356],
        # Note: I corrected a typo from your snippet (original had 490.7524375321356)
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


def setup_video_capture(video_file_path, start_frame_index):
    """
    Opens a video file and loads camera calibration.
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file_path}")
        return None, None, None

    # Set the video to start at the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

    # Load calibration data from the dedicated function
    cameraMatrix, distCoeffs = load_camera_calibration()

    return cap, cameraMatrix, distCoeffs


def setup_aruco():
    """
    Sets up the Aruco detector.
    Assumes units are in millimeters (mm).
    """
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    # Set marker size in mm. Assumes model_points.csv is also in mm.
    marker_size = 16.0  # 16mm
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size


def estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs):  # for individual marker
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash, rvecs, tvecs = cv2.solvePnP(marker_points, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs, trash


# Estimate the global information of the dodecahedron from detected markers
def estimatePoseGlobal(model_points, image_points, cameraMatrix, distCoeffs):
    trash, rvecs, tvecs = cv2.solvePnP(model_points, image_points, cameraMatrix, distCoeffs, False,
                                       cv2.SOLVEPNP_ITERATIVE)
    return rvecs, tvecs, trash


def main():
    # --- Parameters from User ---
    video_path = 'video/cam0_1080p30.mkv'
    start_frame = 1
    end_frame = 262
    local_thick = 3
    global_thick = 7
    # ----------------------------

    frame_count = start_frame  # Initialize frame counter

    cap, cameraMatrix, distCoeffs = setup_video_capture(video_path, start_frame)
    if cap is None:
        return  # Exit if video file couldn't be opened

    detector, marker_size = setup_aruco()

    # Read Dodecahedron 3D coordinates
    # Assumes 'model_points_4x4.csv' coordinates are in millimeters (mm)
    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
    except FileNotFoundError:
        print("Error: 'markers/model_points_4x4.csv' not found.")
        print("Please make sure the CSV file is in the correct directory.")
        cap.release()
        return

    row, column = data.shape
    cols_to_combine = ['x', 'y', 'z']
    model_points_2d_list = data[cols_to_combine].values.tolist()

    K = 4  # Number of 2D data in a group
    tmp1 = iter(model_points_2d_list)
    tmp2 = [tmp1] * K
    model_points_3d_list = [list(ele) for ele in zip(*tmp2)]

    global_pose = True
    plot_pen_tip = False

    while True:
        ret, frame = cap.read()

        # If 'ret' is False (video ends) or we pass the end_frame, stop.
        if not ret or frame_count > end_frame:
            if not ret:
                print("End of video reached.")
            else:
                print(f"Reached specified end frame ({end_frame}).")
            break

        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

        image_points_2d = deque()
        model_points_3d = deque()

        if ids is not None:
            for i, corner in enumerate(corners):
                points = corner[0].astype(np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 255))
                cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                for j in range(1, 4):
                    cv2.putText(frame, str(j), points[j], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                rvecs, tvecs, _objPoints = estimatePoseLocal(corner, marker_size, cameraMatrix, distCoeffs)

                if not global_pose:
                    # Draw local axis (10mm length) with specified local_thick
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs, tvecs, 10.0, local_thick)
                    overlay = frame.copy()
                    cv2.line(overlay, (20, 20), (250, 20), (0, 255, 0), 40)  # Green bar
                    alpha = 0.4
                    cv2.putText(frame, 'Mode: Local Pose Estimation (L/G to toggle)', (20, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                if ids[i][0] < 12:
                    for j in range(4):
                        image_points_2d.append(corner[0][j].tolist())
                    for j in range(4):
                        model_points_3d.append(model_points_3d_list[ids[i][0]][j])

            tmp = np.array(image_points_2d)
            image_points = tmp[np.newaxis, :]
            model_points = np.array(model_points_3d)

            if len(model_points) >= 4:
                rvecs_global, tvecs_global, _objPoints = estimatePoseGlobal(model_points, image_points, cameraMatrix,
                                                                            distCoeffs)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rvecs_global)

                if global_pose:
                    # Draw global axis (50mm length) with specified global_thick
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs_global, tvecs_global, 50.0, global_thick)
                    overlay = frame.copy()
                    cv2.line(overlay, (20, 20), (250, 20), (255, 255, 0), 40)  # Cyan bar
                    alpha = 0.4
                    cv2.putText(frame, 'Mode: Global Pose Estimation (L/G to toggle)', (20, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    if plot_pen_tip:
                        pen_tip_loc = np.array([[-0.014943], [-65.6512], [85.2906]])
                        pen_tip_object_point = pen_tip_loc.reshape(1, 1, 3)
                        image_point, _ = cv2.projectPoints(pen_tip_object_point, rvecs_global, tvecs_global,
                                                           cameraMatrix, distCoeffs)

                        if image_point is not None:
                            pt = image_point[0][0].astype(int)
                            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
                            cv2.putText(frame, "Pen Tip", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)

        # Display current frame number
        cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('g'):
            global_pose = True
            print("Mode: Global Pose Estimation")
        elif key == ord('l'):
            global_pose = False
            print("Mode: Local Pose Estimation")
        elif key == ord('p'):
            plot_pen_tip = not plot_pen_tip
            print(f"Plot Pen Tip: {plot_pen_tip}")

        # Increment frame counter
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processing finished. Stopped at frame {frame_count - 1}.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass