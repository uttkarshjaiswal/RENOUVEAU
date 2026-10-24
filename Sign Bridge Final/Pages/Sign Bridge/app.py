
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import joblib
import text_to_speech as tts


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_FILE = "sign_model.pkl"
TASK_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

CONFIDENCE_THRESHOLD = 0.65
PREDICTION_WINDOW = 7
MIN_AGREEMENT = 5
MAX_HISTORY = 8

CAMERA_INDEX = 0

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ==========================================
# 2. DOWNLOAD MEDIAPIPE TASK MODEL IF NEEDED
# ==========================================

if not os.path.exists(TASK_FILE):
    print("Downloading MediaPipe hand landmark model...")

    try:
        from urllib.request import urlretrieve

        urlretrieve(MODEL_URL, TASK_FILE)
        print("Hand landmark model downloaded.")

    except Exception as exc:
        raise RuntimeError(
            "Could not download hand_landmarker.task. "
            "Check your internet connection and try again."
        ) from exc


# ==========================================
# 3. LOAD THE TRAINED CLASSIFIER
# ==========================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} was not found. "
        "Run 'python train_model.py' first."
    )

saved_data = joblib.load(MODEL_FILE)

if not isinstance(saved_data, dict) or "model" not in saved_data:
    raise ValueError(
        "Unexpected model file format. "
        "Retrain using the SignBridge train_model.py script."
    )

model = saved_data["model"]
labels = saved_data.get("labels", model.classes_)

if getattr(model, "n_features_in_", None) != 63:
    raise ValueError(
        "The trained model must accept 63 features "
        "(21 hand landmarks x 3 coordinates). "
        "Please check train_model.py and data/signs.csv."
    )

if not hasattr(model, "predict_proba"):
    raise ValueError(
        "The trained model does not support predict_proba(). "
        "Check that your training script uses a compatible "
        "classifier, such as RandomForestClassifier."
    )

print("===================================")
print("       SIGNBRIDGE STARTING")
print("===================================")
print("Model loaded successfully.")
print("Recognized classes:", list(labels))


# ==========================================
# 4. INITIALIZE MEDIAPIPE TASKS API
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=TASK_FILE
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def extract_features(hand_landmarks):
    """
    Convert 21 hand landmarks into 63 normalized features.

    This must match the normalization used in train_model.py
    and the dataset collection script.
    """
    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32,
    )

    # Make coordinates relative to the wrist.
    points = points - points[0]

    # Normalize scale.
    scale = float(np.max(np.abs(points)))

    if scale > 0:
        points = points / scale

    return points.flatten().reshape(1, 63)


def draw_hand(frame, hand_landmarks):
    """Draw hand landmarks and connections on the camera frame."""
    height, width = frame.shape[:2]
    pixel_points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        pixel_points.append((x, y))

    # Draw connections.
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_point = pixel_points[start_idx]
        end_point = pixel_points[end_idx]

        cv2.line(
            frame,
            start_point,
            end_point,
            (80, 220, 100),
            2,
            cv2.LINE_AA,
        )

    # Draw landmark points.
    for point in pixel_points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 180, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_text(
    frame,
    text,
    position,
    scale=0.6,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ==========================================
# 6. APPLICATION STATE
# ==========================================

recent_predictions = deque(maxlen=PREDICTION_WINDOW)
sign_history = deque(maxlen=MAX_HISTORY)

last_saved_sign = None
subtitle = "Show a sign to begin"
confidence = 0.0


# ==========================================
# 7. START WEBCAM AND HAND LANDMARKER
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not open webcam. Check camera permissions "
        "and close other applications using the camera."
    )

if not cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):
    print(
        "Camera backend does not support a one-frame buffer; "
        "live-video lag may persist."
    )

print("\nWebcam started!")
print("Press Q to quit.")
print("Press C to clear sign history.")

speaker = None

try:
    # Keep one speech engine alive instead of creating a worker per label.
    speaker = tts.SpeechOutput()

    with HandLandmarker.create_from_options(options) as hands:

        while True:
            speaker.raise_if_failed()

            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            # Selfie-style mirror view.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # MediaPipe Tasks requires an SRGB image.
            rgb_frame = np.ascontiguousarray(rgb_frame)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # VIDEO mode requires increasing timestamps.
            timestamp_ms = time.monotonic_ns() // 1_000_000

            results = hands.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # --------------------------------------
            # DETECT HAND AND PREDICT SIGN
            # --------------------------------------

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                draw_hand(frame, hand_landmarks)

                features = extract_features(hand_landmarks)

                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))

                prediction = str(model.classes_[best_index])
                confidence = float(probabilities[best_index])

                if confidence >= CONFIDENCE_THRESHOLD:
                    recent_predictions.append(prediction)

                    if len(recent_predictions) >= MIN_AGREEMENT:
                        label, count = Counter(
                            recent_predictions
                        ).most_common(1)[0]

                        if count >= MIN_AGREEMENT:
                            subtitle = label

                            # Save only when the stable label changes.
                            if label != last_saved_sign:
                                sign_history.append(label)
                                speaker.speak(label)
                                last_saved_sign = label
                        else:
                            subtitle = "Recognizing..."
                    else:
                        subtitle = "Recognizing..."

                else:
                    recent_predictions.clear()
                    last_saved_sign = None
                    subtitle = "Uncertain sign"

            else:
                recent_predictions.clear()
                confidence = 0.0
                subtitle = "No hand detected"

                # Allow the same sign to be recorded again
                # after the hand disappears.
                last_saved_sign = None

            # --------------------------------------
            # DRAW USER INTERFACE
            # --------------------------------------

            height, width = frame.shape[:2]

            # Header bar.
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 55),
                (35, 35, 35),
                -1,
            )

            draw_text(
                frame,
                "SignBridge | Live Sign Recognition",
                (15, 35),
                scale=0.7,
            )

            # Sign history.
            history_items = list(sign_history)

            history_text = (
                "Signs: " + " -> ".join(history_items)
                if history_items
                else "Signs: None yet"
            )

            max_chars = max(20, (width - 30) // 10)

            if len(history_text) > max_chars:
                history_text = "..." + history_text[-(max_chars - 3):]

            draw_text(
                frame,
                history_text,
                (15, 85),
                scale=0.55,
                color=(0, 255, 255),
            )

            # Subtitle background.
            cv2.rectangle(
                frame,
                (0, height - 80),
                (width, height),
                (35, 35, 35),
                -1,
            )

            draw_text(
                frame,
                "Sign: " + subtitle,
                (15, height - 42),
                scale=0.8,
            )

            draw_text(
                frame,
                f"Confidence: {confidence:.0%}",
                (15, height - 12),
                scale=0.5,
                color=(180, 255, 180),
                thickness=1,
            )

            draw_text(
                frame,
                "C: Clear history | Q: Quit",
                (max(10, width - 260), height - 12),
                scale=0.45,
                color=(220, 220, 220),
                thickness=1,
            )

            cv2.imshow(
                "SignBridge - Live Recognition",
                frame,
            )

            # --------------------------------------
            # KEYBOARD CONTROLS
            # --------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                sign_history.clear()
                recent_predictions.clear()
                last_saved_sign = None
                subtitle = "History cleared"
                confidence = 0.0

                print("Sign history cleared.")

finally:
    if speaker is not None:
        speaker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("SignBridge closed.")