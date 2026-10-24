# SignBridge — Record Your Own Signs

SignBridge recognizes hand poses from webcam landmarks and displays/speaks the predicted label.
You can now record **any custom label** without editing the Python code.

## Setup

Use Python 3.10 or 3.11 for best compatibility with the MediaPipe Tasks API, then install:

```bash
python -m pip install -r requirements.txt
```

## Add a new label and record its sign

You can use the RENOUVEAU GUI by selecting **START**, then **RECORD**. Enter
the gesture label and sample count in the form; a webcam window opens for
recording. The GUI adds samples to the project-level `data/signs.csv` used by
the live recognizer and automatically retrains the project-level model. Open
or restart the live recognizer to load the updated labels.

From this folder, the command-line workflow is also available:

1. Open a terminal in this folder.
2. Run:

   ```bash
   python collect_data.py
   ```

3. Type any label, such as `GOOD MORNING`, `PLEASE HELP`, or a word of your choice.
4. Enter the target number of samples (default 150; 100–300 is a useful starting point).
5. In the webcam window, press **S** to start recording. Press **S** again to pause; press **Q** to save and close.
6. Repeat `python collect_data.py` for each additional label. There is no hard-coded list of labels.
7. When you have collected examples for at least two labels, run:

   ```bash
   python train_model.py
   ```

8. Run live recognition:

   ```bash
   python app.py
   ```

## Important controls

- `S`: start/pause recording
- `Q`: save and quit
- Data is appended to `data/signs.csv`; existing examples are preserved.
- Training creates/updates `sign_model.pkl`. Restart `app.py` after training so it loads the updated model.

## Tips for better results

- Collect at least 100 samples per label as a starting point.
- Keep the hand visible and the sign consistent; slightly vary hand position and distance.
- Record every label you expect the model to distinguish, including confusingly similar signs.
- Test using recordings or people not used during collection; training accuracy alone is not enough.
- This prototype uses one hand and 63 normalized hand-landmark features. It is designed for static hand poses, not complete sign-language translation or motion-based signs.
- Only add labels for signs you have verified. The model learns the labels you assign to the examples; it does not validate whether a gesture is an official sign-language sign.

## Files

- `collect_data.py` — dynamic custom-label recorder
- `data/signs.csv` — stored samples
- `train_model.py` — trains the classifier from the CSV
- `app.py` — live recognition, subtitles, and speech
- `text_to_speech.py` — speech output
- `hand_landmarker.task` — MediaPipe hand landmark model
