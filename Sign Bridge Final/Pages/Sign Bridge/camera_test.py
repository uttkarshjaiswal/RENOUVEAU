
import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    # Capture one frame from the camera
    success, frame = camera.read()

    if not success:
        print("Could not read a frame.")
        break

    # Show the frame in a window
    cv2.imshow("SignBridge - Camera Test", frame)

    # Quit when the user presses q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()