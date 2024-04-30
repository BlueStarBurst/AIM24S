from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('yolov8n.pt')

annotations = []
classes = []
def process_frame(frame):
    global annotations
    global classes
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame
    results = model(frame_rgb)

    # Ensure that results is not empty
    if results:
        # Get the first result object
        result = results[0]
        
        annotations = result.boxes.xyxy.tolist()
        classes = [model.names[int(i)] for i in result.boxes.cls]
        
        # print("\n\n\n")
        # print("ANNOTATIONS",annotations)
        # print("\n\n\n")
        # print("CLASSES",classes)
        # print("\n\n\n")

        # Get annotated image with bounding boxes
        annotated_image = result.plot()

        # Convert annotated image from numpy array to BGR format
        annotated_image_bgr = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

        # Display the frame with bounding boxes
        cv2.imshow('YOLO Object Detection', annotated_image_bgr)
    else:
        # Display the original frame if no results found
        cv2.imshow('YOLO Object Detection', frame)
        
        annotations = {}

cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    w = frame.shape[1]
    h = frame.shape[0]
    s = min(w, h)
    # crop width to center s 
    frame = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2] 
    # resize to 512x512
    frame = cv2.resize(frame, (512, 512))

    # Convert frame to JPEG format
    _, frame_data = cv2.imencode('.jpg', frame)
    
    process_frame(frame)

cv2.destroyAllWindows()