
import os
import time
from collections import Counter, deque
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import joblib
import importlib.util
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer
import threading
if "gesture_display" not in st.session_state:
    st.session_state["gesture_display"] = {
        "lock": threading.Lock(),
        "current": "Show a sign to begin",
        "text": "",
        "confidence": 0.0,
    }

gesture_display = st.session_state["gesture_display"]


tts_path = Path(__file__).parent / "Sign Bridge" / "text_to_speech.py"
spec = importlib.util.spec_from_file_location("signbridge_tts", tts_path)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {tts_path}")

tts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts)
st.html("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.35)),
        url("https://i.pinimg.com/736x/a8/99/a7/a899a7489d4cf51f5db54738d802bf31.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""")


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = (
    PROJECT_ROOT / "sign_model.pkl",
    Path(__file__).resolve().parent / "sign_model.pkl",
    Path(__file__).resolve().parent / "Sign Bridge" / "sign_model.pkl",
)
MODEL_FILE = next(
    (path for path in MODEL_CANDIDATES if path.is_file()),
    MODEL_CANDIDATES[0],
)
TASK_FILE = PROJECT_ROOT / "hand_landmarker.task"

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

if not MODEL_FILE.is_file():
    raise FileNotFoundError(
        "No trained sign model was found. Checked: "
        f"{', '.join(str(path) for path in MODEL_CANDIDATES)}. "
        "Run 'python \"Pages\\Sign Bridge\\train_model.py\"' first."
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
        model_asset_path=str(TASK_FILE)
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
@st.cache_resource
def get_hand_landmarker():
    return HandLandmarker.create_from_options(options)

@st.cache_resource
def get_speaker():
    return tts.SpeechOutput()

hands = get_hand_landmarker()
speaker = get_speaker()

recent_predictions = deque(maxlen=PREDICTION_WINDOW)
sign_history = deque(maxlen=MAX_HISTORY)

last_saved_sign = None
subtitle = "Show a sign to begin"
confidence = 0.0


def process_video_frame(video_frame):
    global last_saved_sign, subtitle, confidence

    # WebRTC supplies the camera frames.
    frame = video_frame.to_ndarray(format="bgr24")
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = time.monotonic_ns() // 1_000_000
    results = hands.detect_for_video(mp_image, timestamp_ms)

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
                label, count = Counter(recent_predictions).most_common(1)[0]

                if count >= MIN_AGREEMENT:
                    subtitle = label

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
            subtitle = "Uncertain sign"
    else:
        recent_predictions.clear()
        confidence = 0.0
        subtitle = "No hand detected"
        last_saved_sign = None
    with gesture_display["lock"]:
        gesture_display["current"] = subtitle
        gesture_display["text"] = " ".join(sign_history)
        gesture_display["confidence"] = confidence
    # Draw the recognized text on the video shown in Streamlit.
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (width, 55), (35, 35, 35), -1)
    draw_text(frame, "SignBridge | Live Sign Recognition", (15, 35), scale=0.7)

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

    cv2.rectangle(frame, (0, height - 80), (width, height), (35, 35, 35), -1)
    draw_text(frame, "Sign: " + subtitle, (15, height - 42), scale=0.8)
    draw_text(
        frame,
        f"Confidence: {confidence:.0%}",
        (15, height - 12),
        scale=0.5,
        color=(180, 255, 180),
        thickness=1,
    )

    # Return the processed frame to the Streamlit page.
    return av.VideoFrame.from_ndarray(frame, format="bgr24")


st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 34px;
        color:white;
    ">
       RENOUVOEU.
    </p>
    """,
    unsafe_allow_html=True)
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 24px;
        color:white;
    ">
      Press START below and allow your browser to use the camera.
    </p>
    """,
    unsafe_allow_html=True)


webrtc_streamer(
    key="RENOUVOEU",
    video_frame_callback=process_video_frame,
    media_stream_constraints={"video": True, "audio": False},
)
@st.fragment(run_every=1)
def show_recognized_text():
    with gesture_display["lock"]:
        current = gesture_display["current"]
        recognized_text = gesture_display["text"]
        score = gesture_display["confidence"]



    if recognized_text:
        st.write(recognized_text)
        st.caption(f"Latest gesture: {current} · Confidence: {score:.0%}")

        if st.button("🔊 Speak recognized text", key="speak_camera_text"):
            speaker.speak(recognized_text)
    else:
        st.write(current)

show_recognized_text()
