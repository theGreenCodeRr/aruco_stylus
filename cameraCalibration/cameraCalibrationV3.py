import cv2
import threading
import numpy as np

# -------------------------------------------------------------------------
# Threaded Video Capture Class
# -------------------------------------------------------------------------
class VideoCaptureThread:
    def __init__(self,
                 src=0,
                 width=640,
                 height=480,
                 backend=None,
                 use_grayscale=False):
        """
        :param src: Camera index or video file path.
        :param width: Desired capture width.
        :param height: Desired capture height.
        :param backend: Optional OpenCV backend (e.g., cv2.CAP_DSHOW or cv2.CAP_V4L2).
        :param use_grayscale: If True, attempt to capture grayscale frames directly.
                              (May or may not be supported by your camera/driver.)
        """
        if backend is not None:
            self.capture = cv2.VideoCapture(src, backend)
        else:
            self.capture = cv2.VideoCapture(src)

        # Attempt to set resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Try to disable color conversion if grayscale is desired
        # (Many cameras won't support this in practice.)
        if use_grayscale:
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self.ret = False
        self.frame = None
        self.stop_flag = False

        # Start a background thread to read frames
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously read frames from the camera on a background thread."""
        while not self.stop_flag:
            ret, frame = self.capture.read()
            if ret:
                self.ret = True
                self.frame = frame

    def read(self):
        """Return the most recent frame."""
        return self.ret, self.frame

    def release(self):
        """Release the camera and stop the thread."""
        self.stop_flag = True
        self.thread.join()
        self.capture.release()


# -------------------------------------------------------------------------
# Calibration Script
# -------------------------------------------------------------------------
def main():
    # -- Calibration Parameters --
    square_size = 20        # size of each chessboard square (mm)
    pattern_size = (7, 10)  # (rows, cols) of internal corners
    reference_img = 40      # number of successful detections required

    # Prepare 3D coordinate grid for the chessboard
    pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # Containers for 3D and 2D points
    objpoints = []
    imgpoints = []

    # Initialize capture (Windows example uses CAP_DSHOW)
    cap = VideoCaptureThread(
        src=0,
        width=640,
        height=480,
        backend=cv2.CAP_DSHOW,
        use_grayscale=True
    )

    print("Starting chessboard detection. Press 'q' to quit early.")
    while len(objpoints) < reference_img:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera.")
            continue

        # If the camera truly outputs single-channel frames, shape is (H, W).
        # Otherwise, convert from BGR to grayscale.
        if len(frame.shape) == 2:
            gray = frame  # Already single-channel
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the corners
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        if found:
            # Refine corner locations
            term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term_crit)

            # Store points
            objpoints.append(pattern_points)
            # Convert corners to (N,2) float32
            imgpoints.append(corners.reshape(-1, 2).astype(np.float32))

            print(f"[INFO] Detected corner pattern {len(objpoints)}/{reference_img}")

            # Draw corners
            cv2.drawChessboardCorners(frame, pattern_size, corners, found)

        # Show feed
        cv2.imshow("Calibration Feed", frame)

        # Quit early with 'q'
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Check if enough images were captured
    if len(objpoints) < reference_img:
        print(f"Only found corners in {len(objpoints)} images; need {reference_img}.")
        print("Not enough corner points detected for calibration.")
        return

    # Perform calibration
    print("[INFO] Calculating camera parameters...")
    image_size = gray.shape[::-1]  # (width, height)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Format camera matrix
    cameraMatrix = np.array([
        [mtx[0, 0], 0.0,       mtx[0, 2]],
        [0.0,       mtx[1, 1], mtx[1, 2]],
        [0.0,       0.0,       1.0     ]
    ], dtype='double')

    # Distortion coefficients (k1, k2, p1, p2, k3)
    distCoeffs = dist[0, :5].reshape(-1, 1).astype('double')

    print(f"[INFO] RMS error = {ret:.4f}")

    # Print camera matrix
    print("\ncameraMatrix = np.array(")
    print("    [")
    for row in cameraMatrix:
        print(f"        [{', '.join(map(str, row))}],")
    print("    ], dtype='double')")

    # Print distortion coefficients
    print("\ndistCoeffs = np.array(")
    print("    [")
    for coeff in distCoeffs:
        print(f"        [{coeff[0]}],")
    print("    ], dtype='double')")

    # ---------------------------------------------------------------------
    # Calculate mean reprojection error (with shape & type fixes)
    # ---------------------------------------------------------------------
    total_error = 0
    for i in range(len(objpoints)):
        # Project the 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )

        # Convert from (N,1,2) to (N,2) float32
        projected_points = projected_points.reshape(-1, 2).astype(np.float32)
        # Ensure original image points are also float32
        original_points = imgpoints[i].astype(np.float32)

        error = cv2.norm(original_points, projected_points, cv2.NORM_L2) \
                / len(projected_points)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"\n[INFO] Mean reprojection error: {mean_error:.4f}")


# -------------------------------------------------------------------------
# Run the main function
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
