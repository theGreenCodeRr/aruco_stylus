import cv2
import numpy as np
import cv2.aruco as aruco

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 0.017  # 17 mm

# Define the 3D points for the corners
marker_3d_points = np.array([
    [-marker_size / 2, marker_size / 2, 0],  # Top-left corner
    [marker_size / 2, marker_size / 2, 0],   # Top-right corner
    [marker_size / 2, -marker_size / 2, 0],  # Bottom-right corner
    [-marker_size / 2, -marker_size / 2, 0]  # Bottom-left corner
])

# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters()

# Start the camera capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set of marker IDs we want to find (from 1 to 12)
desired_marker_ids = set(range(1, 13))

# Dictionary to store the corners of each detected marker
marker_corners_dict = {}

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

        # If at least one marker detected
        if ids is not None and len(ids) > 0:
            # Flatten the ids array to iterate over it
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                # Check if the detected marker is one of the desired
                if marker_id in desired_marker_ids:
                    # Store the corners for the detected marker
                    marker_corners_dict[marker_id] = corners[i].tolist()
                    # Remove the found marker ID from the set of desired IDs
                    desired_marker_ids.remove(marker_id)

            # Visualize the detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Check if all desired markers are found
        if not desired_marker_ids:
            print("All desired markers have been detected.")
            break

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Print the stored corners for each detected marker
for marker_id, corners in marker_corners_dict.items():
    print(f"Marker ID {marker_id} corners:")
    print(corners)

# Open a text file to write the corners
with open('aruco_corners.txt', 'w') as file:
    for marker_id, corners in marker_corners_dict.items():
        # Write the marker ID and its corners to the file
        file.write(f"Marker ID {marker_id}:\n")
        for i, corner in enumerate(corners[0]):
            # Write each 2D corner and its corresponding 3D point
            file.write(f"[2D Corner] {i+1}: {corner[0]}, {corner[1]}   [3D Point: {marker_3d_points[i][0]}, {marker_3d_points[i][1]}, {marker_3d_points[i][2]}]\n")
        file.write("\n")

print("The corner points and corresponding 3D points have been saved")