import cv2
import datetime
import platform
import threading
import time
import queue
import os


class CameraThread(threading.Thread):
    """
    A dedicated thread for continuously capturing frames from a single camera.
    This prevents the main processing loop from being blocked by a slow camera.
    """

    def __init__(self, cam_config, frame_queue):
        super().__init__()
        self.cam_config = cam_config
        self.frame_queue = frame_queue
        self.cap = None
        self.running = False
        self.daemon = True  # Thread will exit when main program exits

    def open_camera(self):
        """Tries different backends to open the camera, based on the OS."""
        idx = self.cam_config['idx']
        system = platform.system()
        # Define preferred backends for each OS
        if system == "Windows":
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif system == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:  # macOS or other
            backends = [cv2.CAP_ANY]

        for api in backends:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                # Request specific resolution and FPS from the camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_config['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_config['height'])
                cap.set(cv2.CAP_PROP_FPS, self.cam_config['fps'])

                # Verify the settings
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"[Camera {idx}] Opened with backend {api}.")
                print(
                    f"  -> Requested: {self.cam_config['width']}x{self.cam_config['height']} @ {self.cam_config['fps']} FPS")
                print(f"  -> Actual:    {w}x{h} @ {fps:.2f} FPS")
                return cap
            cap.release()
        print(f"[ERROR] Could not open camera index {idx} with any backend.")
        return None

    def run(self):
        """The main loop for the camera thread."""
        self.cap = self.open_camera()
        if not self.cap:
            return  # Exit thread if camera failed to open

        self.running = True
        while self.running:
            # If the queue is full, it means the main thread is lagging.
            # We discard the oldest frame and put the new one to keep the feed live.
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass

            ret, frame = self.cap.read()
            if not ret:
                print(f"[Warning] Camera {self.cam_config['idx']} stream ended or failed.")
                self.running = False
                continue

            # Put the captured frame and its capture time into the queue
            self.frame_queue.put((self.cam_config['idx'], frame))

        # Cleanup
        self.cap.release()
        print(f"[Camera {self.cam_config['idx']}] Thread stopped and camera released.")

    def stop(self):
        """Signals the thread to stop."""
        self.running = False


def main(cam_configs):
    """
    Main function to manage camera threads, display feeds, and handle recording.
    """
    # Create a directory for video output if it doesn't exist
    output_dir = 'video'
    os.makedirs(output_dir, exist_ok=True)

    # --- Initialization ---
    threads = []
    frame_queues = {}
    writers = {}

    # Use a more common compressed codec. FFV1 is lossless and creates huge files.
    # 'mp4v' is widely supported for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for cfg in cam_configs:
        # A queue for each camera to hold frames. maxsize=2 means we keep one frame
        # in buffer and one being processed, reducing latency.
        frame_queues[cfg['idx']] = queue.Queue(maxsize=2)

        thread = CameraThread(cfg, frame_queues[cfg['idx']])
        threads.append(thread)
        thread.start()

    # Give threads a moment to initialize and open cameras
    time.sleep(3)  # Increased sleep time slightly for camera init

    # --- Main Loop ---
    recording = False
    print("\n" + "=" * 30)
    print("Controls:")
    print("  's' - Start/Stop Recording")
    print("  'q' - Quit")
    print("=" * 30 + "\n")

    try:
        while True:
            # Check if all threads have stopped (e.g., all cameras disconnected)
            if not any(t.is_alive() for t in threads):
                print("All camera threads have stopped. Exiting.")
                break

            # Process frames from all active cameras
            for cfg in cam_configs:
                try:
                    # Get the latest frame from the queue without blocking
                    idx, frame = frame_queues[cfg['idx']].get_nowait()
                except queue.Empty:
                    # No new frame available for this camera yet
                    continue

                # --- Overlays ---
                # Timestamp with milliseconds
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                cv2.putText(frame, timestamp, (10, cfg['height'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Status overlay
                if recording:
                    status_text = "REC"
                    status_color = (0, 0, 255)  # Red
                    # Draw a red circle indicator
                    cv2.circle(frame, (cfg['width'] - 100, 25), 10, status_color, -1)
                else:
                    status_text = "LIVE"
                    status_color = (0, 255, 0)  # Green

                cv2.putText(frame, status_text, (cfg['width'] - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # --- Display ---
                win_name = f"Cam {cfg['idx']}"
                cv2.imshow(win_name, frame)

                # --- Writing ---
                if recording and idx in writers:
                    writers[idx].write(frame)

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit command received. Shutting down...")
                break

            elif key == ord('s'):
                recording = not recording
                if recording:
                    print("Recording STARTED...")
                    # Create new VideoWriter objects for each camera
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    for cfg in cam_configs:
                        # Check if the corresponding thread is alive before creating a writer
                        if any(t.is_alive() and t.cam_config['idx'] == cfg['idx'] for t in threads):
                            output_filename = os.path.join(output_dir, f"cam{cfg['idx']}_{ts}.mp4")
                            writers[cfg['idx']] = cv2.VideoWriter(
                                output_filename, fourcc, cfg['fps'],
                                (cfg['width'], cfg['height'])
                            )
                            print(f"  -> Saving Cam {cfg['idx']} to {output_filename}")
                else:
                    print("Recording STOPPED.")
                    # Release all writer objects
                    for writer in writers.values():
                        writer.release()
                    writers.clear()  # Clear the dictionary for the next recording session

    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        # Stop all running threads
        for t in threads:
            t.stop()
            t.join()  # Wait for thread to finish

        # Release any remaining writers
        if recording:
            print("Finalizing video files...")
            for writer in writers.values():
                writer.release()

        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    # Configuration for a single camera.
    # Even if the camera is 4K capable, we are requesting 1080p @ 30fps.
    # The script will attempt to set these properties on the camera.
    camera_configurations = [
        {'idx': 0, 'width': 1920, 'height': 1080, 'fps': 30},
    ]
    main(camera_configurations)
