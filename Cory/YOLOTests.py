import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Function to process each frame from the webcam
def process_frame(frame):
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame
    results = model(frame_rgb)

    # Ensure that results is not empty
    if results:
        # Get the first result object
        result = results[0]

        # Get annotated image with bounding boxes
        annotated_image = result.plot()

        # Convert annotated image from numpy array to BGR format
        annotated_image_bgr = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

        # Display the frame with bounding boxes
        cv2.imshow('YOLO Object Detection', annotated_image_bgr)
    else:
        # Display the original frame if no results found
        cv2.imshow('YOLO Object Detection', frame)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

# Process frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process the frame
    process_frame(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
