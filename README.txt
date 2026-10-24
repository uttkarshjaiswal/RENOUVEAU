# Hand Gesture to Text and Speech

This project converts hand gestures into text and speech. It also allows users to record and add new gestures.

## How to Run the Project

1. Open a terminal in the project folder containing `1.py`.
2. Run the following command:

   ```bash
   streamlit run 1.py
   ```

3. A new browser window or tab will open with the application.
4. Click the **Start** button.
5. Follow the on-screen guide for instructions on how to use the application.

## Adding New Gestures

The application also allows users to record new gestures. Follow the instructions provided in the application to record and add them.

## Used python Modules 
   
   Module imported in Python	Package to install	Purpose
   
   1) streamlit	                streamlit	        Website and app interface
   2) cv2	                opencv-python	        Webcam access and image processing
   3) mediapipe	                mediapipe	        Hand landmark detection
   4) numpy	                numpy	                Landmark and numerical calculations
   5) joblib	                joblib            	Loading the trained recognition model
   6) pandas	                pandas          	Reading and processing the training dataset
   7) sklearn            	scikit-learn	        Training and evaluating the classifier
   8) av	                av	                Handling video frames for Streamlit WebRTC
   9) streamlit_webrtc       	streamlit-webrtc	Streaming webcam video in the browser
   10)pyttsx3	                pyttsx3	                Text-to-speech
   11)comtypes	                comtypes	        Windows speech-engine integration