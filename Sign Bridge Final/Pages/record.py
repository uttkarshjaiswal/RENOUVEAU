import importlib.util
from pathlib import Path

import streamlit as st

left, center, right = st.columns([1, 7, 1])
with right:
    if st.button("HOME", width="content"):
        st.success("loading")
        st.switch_page("1.py")

st.set_page_config(page_title="Record a gesture", layout="wide")

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

project_root = Path(__file__).resolve().parent.parent
recorder_path = project_root / "Pages" / "Sign Bridge" / "collect_data.py"
trainer_path = project_root / "Pages" / "Sign Bridge" / "train_model.py"
dataset_path = project_root / "data" / "signs.csv"
landmarker_path = project_root / "hand_landmarker.task"
model_path = project_root / "sign_model.pkl"

st.title("Record a custom gesture")
st.write(
    "Enter the word or phrase for this gesture, then record examples with "
    "your webcam. The samples are added to the dataset used by RENOUVEAU."
)

with st.form("record_gesture_form"):
    label = st.text_input("Gesture label", placeholder="For example: GOOD MORNING")
    target_samples = st.number_input(
        "Samples to record",
        min_value=10,
        max_value=1000,
        value=150,
        step=10,
        help="100-300 samples per gesture is a useful starting point.",
    )
    submitted = st.form_submit_button("Start recording")

st.info(
    "A separate webcam window will open. Press S to start or pause recording "
    "and Q to finish. Keep the gesture visible and vary its position slightly."
)

if submitted:
    normalized_label = label.strip()
    if not normalized_label:
        st.error("Enter a label before starting the recording.")
    elif normalized_label.casefold() == "unknown":
        st.error("'UNKNOWN' is reserved for uncertain predictions. Choose another label.")
    else:
        spec = importlib.util.spec_from_file_location(
            "signbridge_collect_data", recorder_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load the gesture recorder at {recorder_path}")
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)

        with st.spinner("Recording gesture samples from your webcam..."):
            saved_count = recorder.record_sign(
                normalized_label,
                int(target_samples),
                csv_file=dataset_path,
                model_path=landmarker_path,
            )

        if saved_count:
            trainer_spec = importlib.util.spec_from_file_location(
                "signbridge_train_model", trainer_path
            )
            if trainer_spec is None or trainer_spec.loader is None:
                raise ImportError(f"Could not load the model trainer at {trainer_path}")
            trainer = importlib.util.module_from_spec(trainer_spec)
            trainer_spec.loader.exec_module(trainer)

            with st.spinner("Training the recognizer with the updated gesture data..."):
                try:
                    labels = trainer.train_model(
                        data_file=dataset_path,
                        model_file=model_path,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    st.error(
                        f"Recorded {saved_count} samples for '{normalized_label}', "
                        f"but the recognizer was not updated: {exc}"
                    )
                else:
                    st.success(
                        f"Recorded {saved_count} samples and updated the recognizer. "
                        f"Recognized labels: {', '.join(labels)}"
                    )
                    st.info(
                        "The new gesture will be available when you open or restart "
                        "the live recognizer."
                    )
        else:
            st.warning("No samples were recorded.")

if st.button("Back to instructions"):
    st.switch_page("Pages/reader.py")
