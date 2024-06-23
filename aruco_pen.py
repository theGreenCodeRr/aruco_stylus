import cv2
import numpy as np
import cv2.aruco as aruco

# Defining
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
marker_size = 0.017  # 17 mm

# # cameraMatrix & distCoeff: lab camera
# cameraMatrix = np.array([
#     [763.43512892, 0, 321.85994173],
#     [0, 764.52495998, 187.09227291],
#     [0, 0, 1]],
#     dtype='double', )
# distCoeffs = np.array([[0.13158662], [0.26274676], [-0.00894502], [-0.0041256], [-0.12036324]])

# Camera matrix and distortion coefficients: Mac camera
cameraMatrix = np.array([
    [826.68182975, 0, 614.71137477],
    [0, 823.29094599, 355.24928406],
    [0, 0, 1]],
    dtype='double', )
distCoeffs = np.array([[-0.43258492], [3.71129672], [-0.01377461], [0.00989978], [-9.44694337]])


# Define the 3D points for the corners
marker_3d_points = np.array([
    [-marker_size / 2, marker_size / 2, 0],  # Top-left corner
    [marker_size / 2, marker_size / 2, 0],   # Top-right corner
    [marker_size / 2, -marker_size / 2, 0],  # Bottom-right corner
    [-marker_size / 2, -marker_size / 2, 0]  # Bottom-left corner
])


# Start the camera capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set of marker IDs we want to find (from 1 to 12)
desired_marker_ids = set(range(1, 13))

# Dictionary to store the corners of each detected marker [marker_id: 2d corners, 3d points]
marker_corners_dict_2d = {}
marker_corners_dict_3d = {}

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the grayscale frame
        corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        # Variables to store the sum of all marker centers
        sum_center_x = 0
        sum_center_y = 0
        num_markers = 0

        # If at least one marker detected
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            # print(f"Detected marker IDs: {ids}")

            for i, marker_id in enumerate(ids):
                # Convert marker_id to an integer if it's a NumPy array
                if isinstance(marker_id, np.ndarray):
                    marker_id = marker_id.item()  # Converts to integer

                # Check if the marker_id is within the desired range
                if marker_id in desired_marker_ids:
                    # Add the detected marker's corners to the dictionary
                    marker_corners_dict_2d[marker_id] = corners[i].tolist()
                    marker_corners_dict_3d[marker_id] = marker_3d_points.tolist()

                    # # Calculate the center of the current marker
                    # center_x = np.mean(corners[i][:, 0])
                    # center_y = np.mean(corners[i][:, 1])
                    #
                    # # Add to the sum of centers
                    # sum_center_x += center_x
                    # sum_center_y += center_y
                    # num_markers += 1
                    #
                    # # Calculate the average center of all markers
                    #
                    # avg_center_x = sum_center_x / num_markers
                    # avg_center_y = sum_center_y / num_markers
                    #
                    # # Define the average center in 2D and 3D space
                    # avg_center_2d = np.array([[avg_center_x, avg_center_y]], dtype=np.float32)
                    # avg_center_3d = np.array([[0, 0, 0]],
                    #                          dtype=np.float32)  # Assuming the center is at the origin in 3D space
                    #
                    # # Use solvePnP to estimate the pose at the average center
                    # success, rvec, tvec = cv2.solvePnP(avg_center_3d, avg_center_2d, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)

                    # Proceed with pose estimation using solvePnP
                    corners_2d = np.squeeze(marker_corners_dict_2d[marker_id])
                    corners_3d = np.array(marker_corners_dict_3d[marker_id], dtype=np.float32)

                    # Use solvePnP to estimate pose
                    success, rvec, tvec = cv2.solvePnP(corners_3d, corners_2d, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)

                    if success:
                        # Draw the axes on the marker
                        aruco.drawDetectedMarkers(frame, corners, ids) # Draw the detected markers
                        cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                    else:
                        print("Pose estimation failed for the average center.")

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

