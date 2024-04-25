import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Create a directory to save the cropped images
save_dir = 'cropped_images'
os.makedirs(save_dir, exist_ok=True)

# Function to process each frame from the webcam
def process_frame(frame, frame_count):
    global save_dir
    
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

        # Save cropped images every 50 frames
        if frame_count % 50 == 0:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cropped_img = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(save_dir, f'cropped_{frame_count}_{i}.jpg'), cropped_img)

    else:
        # Display the original frame if no results found
        cv2.imshow('YOLO Object Detection', frame)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

frame_count = 0

# Process frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process the frame
    process_frame(frame, frame_count)

    frame_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
