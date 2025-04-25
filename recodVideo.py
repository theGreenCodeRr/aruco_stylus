import cv2
import datetime
import platform

def open_camera(idx, fps):
    system = platform.system()
    if system == "Windows":
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif system == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for api in backends:
        cap = cv2.VideoCapture(idx, api)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if cap.isOpened():
            print(f"[Camera {idx}] opened with backend {api}")
            return cap
        cap.release()
    print(f"Warning: Could not open camera index {idx}")
    return None

def main(cam_configs):
    cams = []
    writers = []

    # Open cameras & writers
    for cfg in cam_configs:
        cap = open_camera(cfg['idx'], cfg['fps'])
        if not cap:
            raise RuntimeError(f"Failed to open camera {cfg['idx']}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])

        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        writer = cv2.VideoWriter(
            cfg['output'],
            fourcc,
            cfg['fps'],
            (cfg['width'], cfg['height']),
            True
        )
        cams.append((cap, cfg))
        writers.append(writer)

    recording = False
    print("Press 's' to start recording, 'q' to quit.")

    try:
        while True:
            # Grab & display each stream
            for (cap, cfg), writer in zip(cams, writers):
                ret, frame = cap.read()
                if not ret:
                    print(f"[Camera {cfg['idx']}] stream ended or failed.")
                    continue

                # Timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, timestamp,
                            (10, cfg['height'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Status overlay
                status_text = "REC" if recording else "PAUSE"
                status_color = (0,0,255) if recording else (255,255,255)
                cv2.putText(frame, status_text,
                            (cfg['width'] - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Show window
                win = f"Cam{cfg['idx']} {cfg['width']}x{cfg['height']}@{cfg['fps']}FPS"
                cv2.imshow(win, frame)

                # Write if recording
                if recording:
                    writer.write(frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if not recording:
                    recording = True
                    print("Recording started...")
            elif key == ord('q'):
                break

    finally:
        for (cap, _), writer in zip(cams, writers):
            cap.release()
            writer.release()
        cv2.destroyAllWindows()
        if recording:
            print("All streams saved.")
        else:
            print("Exited without recording.")

if __name__ == '__main__':
    configs = [
        {'idx': 0, 'width': 1920, 'height': 1080, 'fps': 60, 'output': 'videos/cam0_1080p60.mkv'},
        {'idx': 1, 'width': 1280, 'height': 720,  'fps': 30, 'output': 'videos/cam1_720p30.mkv'},
        {'idx': 2, 'width': 1280, 'height': 720,  'fps': 30, 'output': 'videos/cam2_720p30.mkv'},
    ]
    main(configs)
