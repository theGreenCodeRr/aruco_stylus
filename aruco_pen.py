import cv2
import numpy as np
from cv2 import aruco
from math import sqrt


def get_dodecahedron_vertices(side_length):
    phi = (1 + sqrt(5)) / 2  # The golden ratio
    a = side_length
    R = a / 2 * sqrt(3) * phi  # Radius of the circumscribed sphere

    # Define the vertices of dodecahedron
    vertices = [
        (R, R, R), (R, R, -R), (R, -R, R), (R, -R, -R),
        (-R, R, R), (-R, R, -R), (-R, -R, R), (-R, -R, -R),
        (0, a / 2 * phi, a / 2 / sqrt(phi)), (0, a / 2 * phi, -a / 2 / sqrt(phi)),
        (0, -a / 2 * phi, a / 2 / sqrt(phi)), (0, -a / 2 * phi, -a / 2 / sqrt(phi)),
        (a / 2 / sqrt(phi), 0, a / 2 * phi), (a / 2 / sqrt(phi), 0, -a / 2 * phi),
        (-a / 2 / sqrt(phi), 0, a / 2 * phi), (-a / 2 / sqrt(phi), 0, -a / 2 * phi),
        (a / 2 * phi, a / 2 / sqrt(phi), 0), (a / 2 * phi, -a / 2 / sqrt(phi), 0),
        (-a / 2 * phi, a / 2 / sqrt(phi), 0), (-a / 2 * phi, -a / 2 / sqrt(phi), 0)
    ]

    return np.array(vertices, dtype=np.float32)


def get_dodecahedron_center(vertices):
    # geometric center of a regular dodecahedron is the average of all its vertices
    center = np.mean(vertices, axis=0)
    return center


def estimate_pose(corners, marker_size, mtx, distortion):
    rvecs, tvecs = [], []
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    for corner in corners:
        retval, rvec, tvec = cv2.solvePnP(marker_points, corner, mtx, distortion)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


def average_rotation_vectors(rvecs):
    # Convert rotation vectors to rotation matrices
    rotation_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    # Average the rotation matrices
    avg_rotation_matrix = np.mean(rotation_matrices, axis=0)
    # Convert the average rotation matrix back to a rotation vector
    avg_rvec, _ = cv2.Rodrigues(avg_rotation_matrix)
    return avg_rvec


def main():
    cap = cv2.VideoCapture(0)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    marker_size = 0.017  # Marker size in meters

    # # cameraMatrix & distCoeff: lab camera
    # cameraMatrix = np.array([
    #     [763.43512892, 0, 321.85994173],
    #     [0, 764.52495998, 187.09227291],
    #     [0, 0, 1]],
    #     dtype='double', )
    # distCoeffs = np.array([[0.13158662], [0.26274676], [-0.00894502], [-0.0041256], [-0.12036324]])

    # Camera matrix and distortion coefficients: mac camera
    cameraMatrix = np.array([
        [826.68182975, 0, 614.71137477],
        [0, 823.29094599, 355.24928406],
        [0, 0, 1]],
        dtype='double', )
    distCoeffs = np.array([[-0.43258492], [3.71129672], [-0.01377461], [0.00989978], [-9.44694337]])

    # Calculate the geometric center of the dodecahedron
    side_length = 20  # Side length of the dodecahedron in mm
    dodecahedron_vertices = get_dodecahedron_vertices(side_length)
    dodecahedron_center = get_dodecahedron_center(dodecahedron_vertices)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect Aruco markers in the frame
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)

        # If at least one marker detected
        if ids is not None and len(ids) > 0:
            # Draw the detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose of each marker
            rvecs, tvecs = estimate_pose(corners, marker_size, cameraMatrix, distCoeffs)

            # Calculate the average rotation vector
            avg_rvec = average_rotation_vectors(rvecs)

            # The geometric center is used directly for the translation vector
            central_tvec = dodecahedron_center.reshape(-1, 1)

            # # Project the geometric center onto the 2D image plane
            # projected_center, _ = cv2.projectPoints(central_tvec, avg_rvec, np.zeros((3, 1)), cameraMatrix, distCoeffs)

            # After calculating the average rotation vector and central translation vector
            avg_rmat, _ = cv2.Rodrigues(avg_rvec)
            transformed_center = np.dot(avg_rmat, dodecahedron_center.reshape(-1, 1)) + central_tvec

            # Now use tvec for the translation vector in drawFrameAxes
            frame = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, avg_rvec, transformed_center, 0.01)
        else:
            print("No markers detected")

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# known issue >>>>
# frame frezze
# center of dodecahedron is in the center of the frame don't move with detected marker