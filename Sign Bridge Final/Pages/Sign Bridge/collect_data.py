"""
SignBridge - Custom Sign Recorder

Add any user-defined sign label without editing Python code.
Each run records one label into data/signs.csv.
"""
import csv
import os
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

DATA_FOLDER = "data"
CSV_FILE = os.path.join(DATA_FOLDER, "signs.csv")
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FEATURE_COLUMNS = [f"feature_{i}" for i in range(63)]
HEADER = ["label"] + FEATURE_COLUMNS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_files(csv_file=CSV_FILE, model_path=MODEL_PATH):
    csv_file = Path(csv_file)
    model_path = Path(model_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    if not csv_file.exists() or csv_file.stat().st_size == 0:
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
    else:
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != HEADER:
            raise ValueError(
                f"{csv_file} has an unexpected CSV header. Back it up and "
                "make sure it has a 'label' column plus feature_0 ... feature_62."
            )

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Hand landmark model downloaded.")


def extract_features(hand_landmarks):
    points = [
        [float(p.x), float(p.y), float(p.z)]
        for p in hand_landmarks
    ]
    if len(points) != 21:
        return None
    wrist = points[0]
    centered = [
        [point[i] - wrist[i] for i in range(3)]
        for point in points
    ]
    scale = max(abs(value) for point in centered for value in point)
    if scale < 1e-6:
        return None
    return [value / scale for point in centered for value in point]


def draw_hand(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = [
        (max(0, min(width - 1, int(p.x * width))),
         max(0, min(height - 1, int(p.y * height))))
        for p in hand_landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 140, 255), -1)


def record_sign(label, target_samples, csv_file=CSV_FILE, model_path=MODEL_PATH):
    label = label.strip()
    if not label:
        raise ValueError("Enter a label for the gesture.")
    if label.casefold() == "unknown":
        raise ValueError("'UNKNOWN' is reserved for uncertain predictions.")
    if target_samples < 10:
        raise ValueError("Collect at least 10 samples; 100-300 is recommended.")

    csv_file = Path(csv_file)
    model_path = Path(model_path)
    ensure_files(csv_file, model_path)
    print(f"\nLabel: {label}")
    print(f"Target: {target_samples} samples")
    print("In the camera window: S = start/pause, Q = save and quit.")
    print("Keep the sign clear; vary distance and position slightly.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    collecting = False
    saved_count = 0
    frame_count = 0
    try:
        with HandLandmarker.create_from_options(options) as hands:
            with csv_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError("Could not read a frame from the webcam.")

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = time.monotonic_ns() // 1_000_000
                    result = hands.detect_for_video(image, timestamp_ms)
                    frame_count += 1

                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        draw_hand(frame, landmarks)
                        if collecting and frame_count % 3 == 0:
                            features = extract_features(landmarks)
                            if features is not None:
                                writer.writerow([label] + features)
                                f.flush()
                                saved_count += 1
                                if saved_count >= target_samples:
                                    collecting = False

                    if collecting:
                        status = "RECORDING"
                    elif saved_count >= target_samples:
                        status = "TARGET REACHED - press Q"
                    else:
                        status = "PAUSED - press S"
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (35, 35, 35), -1)
                    cv2.putText(frame, f"Label: {label}", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, status, (12, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 220, 0) if collecting else (0, 200, 255), 2)
                    cv2.putText(frame, f"Samples: {saved_count}/{target_samples}",
                                (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    cv2.imshow("SignBridge - Record Custom Sign", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        if saved_count >= target_samples:
                            print("Target already reached. Press Q to finish.")
                            collecting = False
                        else:
                            collecting = not collecting
                            print("Recording started." if collecting else "Recording paused.")
                    if saved_count >= target_samples:
                        collecting = False
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nFinished recording label: {label}")
    print(f"Samples saved this session: {saved_count}")
    print(f"Dataset: {csv_file.resolve()}")
    return saved_count


def main():
    label = input(
        "\nEnter the new sign label (for example, GOOD MORNING): "
    ).strip()
    if not label:
        print("No label entered. Nothing was recorded.")
        return

    try:
        target = input("How many samples to collect? [150]: ").strip()
        target_samples = int(target) if target else 150
        record_sign(label, target_samples)
    except ValueError as exc:
        print(exc)


if __name__ == "__main__":
    main()
