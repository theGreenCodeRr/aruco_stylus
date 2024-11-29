import cv2
import numpy as np

# Chessboard size
square_size = 21.5      # Mesh size (in mm)
pattern_size = (7, 7)  # Number of interception points
reference_img = 40     # Number of images to be captured

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)  # Chessboard coordinate (X,Y,Z) with (Z=0)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

# Capture --- set camera resolution
width = 640
height = 480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while len(objpoints) < reference_img:
    ret, img = capture.read()
    height = img.shape[0]
    width = img.shape[1]
    print(height, width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners of the chessboard
    ret, corner = cv2.findChessboardCorners(gray, pattern_size)

    # If corners were found
    if ret:
        print("detected corner!")
        print(str(len(objpoints) + 1) + "/" + str(reference_img))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
        imgpoints.append(corner.reshape(-1, 2))
        objpoints.append(pattern_points)

    cv2.imshow('image', img)

    # Wait for 200 ms for the next image
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

print("calculating camera parameter...")

# Intrinsic parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save to .npy files
np.save("Camera Matrix", mtx)             # Camera Matrix
np.save("Distortion Coefficient", dist.ravel())   # Distortion Matrix

# Save to text file
with open("camera_parameters.txt", "w") as f:
    f.write("Camera Matrix:\n")
    np.savetxt(f, mtx, fmt='%f')
    f.write("\nDistortion Coefficients:\n")
    np.savetxt(f, dist.ravel(), fmt='%f')

# Show the results
print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())