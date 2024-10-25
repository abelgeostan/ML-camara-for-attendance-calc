# ML-camara-for-attendance-calc
Hive Night Hackathon project

to do live detection:
import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('C:\\yolodogcat\\face4.pt')  # Replace with the path to the downloaded 'best.pt'

# Open the webcam (0 is the default camera index; change if necessary)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame from the webcam

    if not ret:
        break

    # Use the trained YOLOv8 model to predict objects in the frame
    results = model.predict(source=frame, conf=0.25)  # Set confidence threshold to 0.25

    # Annotate the frame with YOLOv8 predictions
    annotated_frame = results[0].plot()  # YOLOv8 provides a plot() method to visualize predictions

    # Show the annotated frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
