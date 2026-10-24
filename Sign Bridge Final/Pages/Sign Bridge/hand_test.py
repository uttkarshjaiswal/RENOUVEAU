
import cv2
import mediapipe as mp
import os
import time
import urllib.request

# Download the MediaPipe hand landmark model if needed
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Connections between the 21 hand landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

try:
    with HandLandmarker.create_from_options(options) as hands:
        print("Hand tracking started! Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the webcam for a natural preview
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Detect hands in video mode
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = hands.detect_for_video(
                mp_image, timestamp_ms
            )

            height, width, _ = frame.shape

            # Draw detected hand landmarks
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connections
                    for start, end in connections:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Draw landmark points
                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Show your hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("SignBridge - Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()