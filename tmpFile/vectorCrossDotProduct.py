import cv2
import numpy as np
import cv2.aruco as aruco


def calculate_angle(marker_corners, camera_position):
    vector1 = np.array([marker_corners[1][0], marker_corners[1][1], 0]) - np.array(
        [marker_corners[0][0], marker_corners[0][1], 0])
    vector2 = np.array([marker_corners[2][0], marker_corners[2][1], 0]) - np.array(
        [marker_corners[0][0], marker_corners[0][1], 0])
    normal_vector = np.cross(vector1, vector2)
    marker_center = np.mean(marker_corners, axis=0)
    marker_center_3d = np.array([marker_center[0], marker_center[1], 0])
    viewing_direction = camera_position - marker_center_3d

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)

    dot_product = np.dot(normal_vector, viewing_direction)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)

    return angle_degrees, normal_vector, viewing_direction, marker_center


def setup_web_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cameraMatrix = np.array(
        [
            [512.4608748213299, 0.0, 328.4200385656418],
            [0.0, 515.8693358160277, 239.43174579198057],
            [0.0, 0.0, 1.0],
        ],
        dtype='double'
    )
    distCoeffs = np.array(
        [
            [0.027471395463218463],
            [0.6930024069075884],
            [0.0057744705053643704],
            [-0.0051510738746365515],
            [-2.6107000812747216],
        ],
        dtype='double'
    )
    return cap, cameraMatrix, distCoeffs


def main():
    cap, cameraMatrix, distCoeffs = setup_web_camera()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    marker_size = 0.016  # 16mm
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    camera_position = np.array([0, 0, 1])  # Adjust based on your setup

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if len(corners) > 0:
            for corner in corners:
                marker_corners = corner[0]
                angle_degrees, normal_vector, viewing_direction, marker_center = calculate_angle(marker_corners,
                                                                                                 camera_position)

                print(f"Angle: {angle_degrees:.2f} degrees")
                print(f"Normal Vector: {normal_vector}")
                print(f"Viewing Direction: {viewing_direction}")
                print(f"Marker Center: {marker_center}")

                marker_center_int = tuple(marker_center.astype(int))
                normal_end = (
                marker_center_int[0] + int(normal_vector[0] * 100), marker_center_int[1] + int(normal_vector[1] * 100))
                viewing_end = (marker_center_int[0] + int(viewing_direction[0] * 100),
                               marker_center_int[1] + int(viewing_direction[1] * 100))

                # Draw normal vector
                cv2.arrowedLine(frame, marker_center_int, normal_end, (0, 0, 255), 2, tipLength=0.3)
                # Draw viewing direction
                cv2.arrowedLine(frame, marker_center_int, viewing_end, (255, 0, 0), 2, tipLength=0.3)
                # Draw camera position
                cv2.circle(frame, (int(camera_position[0]), int(camera_position[1])), 5, (0, 255, 255), -1)

                # Display angle
                cv2.putText(frame, f"Angle: {angle_degrees:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
