import sys
import numpy as np
import cv2
from cv2 import aruco
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets  # Correct import for QtWidgets
import math

# Camera setup function
def camera_setup():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera matrix and distortion coefficients (calibrated values)
    cameraMatrix = np.array(
        [[505.1150576, 0, 359.14439401],
         [0, 510.33530166, 230.33963591],
         [0, 0, 1]],
        dtype='double')
    distCoeffs = np.array([[0.07632527], [0.15558049], [0.00234922], [0.00500232], [-0.46829062]], dtype='double')

    return cap, cameraMatrix, distCoeffs

# ArUco setup function
def aruco_setup():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.059  # Marker size in meters
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, marker_size

# Pose estimation for the marker (computing rotation and translation vectors)
def pose_estimation(corner, marker_size, cameraMatrix, distCoeffs):
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    success, rvecs, tvecs = cv2.solvePnP(marker_points, corner, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)
    if success:
        rvecs = rvecs.reshape((3, 1)) if rvecs.shape != (3, 1) else rvecs
        tvecs = tvecs.reshape((3, 1)) if tvecs.shape != (3, 1) else tvecs
    return rvecs, tvecs

# PyQt window and 3D plot setup
class Window(QtWidgets.QWidget):  # Correctly inherit from QWidget
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time 3D Visualization")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QtWidgets.QVBoxLayout(self)

        # Create a 3D plot using PyQtGraph
        self.plotWidget = pg.PlotWidget()
        self.layout.addWidget(self.plotWidget)

        # Create a 3D scatter plot
        self.scatter = pg.ScatterPlotItem(size=10, symbol='o', brush=(255, 0, 0))  # Red pen tip
        self.plotWidget.addItem(self.scatter)

        # Set up the camera and ArUco detection
        self.cap, self.cameraMatrix, self.distCoeffs = camera_setup()
        self.detector, self.marker_size = aruco_setup()

        # Start a timer to update the plot
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 50ms

    def update_plot(self):
        ret, img = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        img_undistorted = cv2.undistort(img, self.cameraMatrix, self.distCoeffs)
        corners, ids, _ = self.detector.detectMarkers(img_undistorted)

        if len(corners) > 0:
            for corner in corners:
                rvecs, tvecs = pose_estimation(corner, self.marker_size, self.cameraMatrix, self.distCoeffs)

                if rvecs is not None and tvecs is not None:
                    # Calculate the position of the pen tip and marker axes
                    rotation_matrix, _ = cv2.Rodrigues(rvecs)
                    origin = tvecs.flatten() * 1000  # Convert to mm
                    pen_tip_local = np.array([[-0.01], [-0.05], [0.1]])  # Offset in meters (pen tip location)
                    pen_tip_world = rotation_matrix @ pen_tip_local + tvecs
                    pen_tip_mm = pen_tip_world.flatten() * 1000  # Convert to mm

                    # Update the scatter plot with the new pen tip position
                    x, y, z = pen_tip_mm
                    self.scatter.setData([x], [y], [z])  # Correct call to setData with separate lists for x, y, z

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# Main function to start the PyQt application
def main():
    app = QtWidgets.QApplication(sys.argv)  # Use QtWidgets for PyQt5
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
