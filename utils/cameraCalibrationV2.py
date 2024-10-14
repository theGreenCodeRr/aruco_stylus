import cv2
import numpy as np

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a 7x7 chessboard pattern with 20mm squares
square_size = 20  # 20mm squares
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Start capturing video from the first camera
cap = cv2.VideoCapture(0)

while len(objpoints) < 50:  # Capture at least 15 valid frames
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        objpoints.append(objp)

        # Refine corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (7, 7), corners2, ret)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration to get the camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the formatted camera matrix and distortion coefficients
cameraMatrix = np.array([
    [1726.08307, 0, 623.477513],
    [0, 1737.78697, 379.475766],
    [0, 0, 1]],
    dtype='double'
)

distCoeffs = np.array([[-1.36772144], [27.480739], [-0.0307059996], [0.00539878234], [-263.256716]],
                      dtype='double'
                      )

print("cameraMatrix =", cameraMatrix)
print("distCoeffs =", distCoeffs)

# #mac webcam
# cameraMatrix = np.array([
#     [1726.08307, 0, 623.477513],
#     [0, 1737.78697, 379.475766],
#     [0, 0, 1]],
#     dtype='double'
# )
#
# distCoeffs = np.array([[-1.36772144], [27.480739], [-0.0307059996], [0.00539878234], [-263.256716]],
#     dtype='double'
# )

# lab shutterCam


