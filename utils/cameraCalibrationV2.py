import cv2
import numpy as np

# Chessboard size
square_size = 21.5  # Mesh size (in mm)
pattern_size = (7, 7)  # Number of interception points
reference_img = 40  # Number of images to be captured

# Chessboard coordinates (X, Y, Z) with Z=0
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Set camera resolution
width = 640
height = 480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while len(objpoints) < reference_img:  # Capture specified number of images
    ret, img = capture.read()
    if not ret:
        print("Failed to capture image")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect corners of the chessboard
    ret, corner = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        print(f"Detected corner {len(objpoints) + 1}/{reference_img}")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
        imgpoints.append(corner.reshape(-1, 2))
        objpoints.append(pattern_points)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corner, ret)

    cv2.imshow('image', img)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # Wait for 200 ms for the next image
        break

capture.release()
cv2.destroyAllWindows()

if len(objpoints) < reference_img:
    print("Not enough corner points detected for calibration")
else:
    print("Calculating camera parameters...")

    # Intrinsic parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the formatted camera matrix and distortion coefficients
    cameraMatrix = np.array([
        [mtx[0, 0], 0, mtx[0, 2]],
        [0, mtx[1, 1], mtx[1, 2]],
        [0, 0, 1]
    ], dtype='double')

    distCoeffs = np.array([dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3], dist[0, 4]], dtype='double').reshape(-1, 1)

    print("cameraMatrix = np.array(")
    print("    [")
    for row in cameraMatrix:
        print(f"        [{', '.join(map(str, row))}],")
    print("    ],")
    print("    dtype='double'")
    print(")")

    print("distCoeffs = np.array(")
    print("    [")
    for coeff in distCoeffs:
        print(f"        [{coeff[0]}],")
    print("    ],")
    print("    dtype='double'")
    print(")")

# globalShutterCam


#  # mac webcam
# cameraMatrix = np.array(
#     [
#         [581.2490064821088, 0.0, 305.2885321972521],
#         [0.0, 587.6316817762934, 288.9932758741485],
#         [0.0, 0.0, 1.0],],
#     dtype='double')
# distCoeffs = np.array(
#     [
#         [-0.31329614267146066],
#         [0.8386295742029726],
#         [-0.0024210244191179104],
#         [0.016349338905846198],
#         [-1.133637004544031],],
#     dtype='double')