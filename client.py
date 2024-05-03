import socket
import cv2
import sys
import struct
import numpy as np
import threading
import json
from ultralytics import YOLO
from time import sleep

# Load the YOLO model


address = '127.0.0.1'
ports = [40195, 40490]
# ports = [3000,3001]

annotations = []
classes = []

stop = False

frame = None
annotated_image_bgr = None

def yolo_thread():
    model = YOLO('yolov8n.pt')
    global annotations
    global classes
    global annotated_image_bgr
    print("Yolo thread has started")
    while not stop:
        # Ensure that frame is not None
        global frame
        if frame is None:
            # print("Frame is None")
            continue
    
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection on the frame
        results = model(frame_rgb, verbose=False)
        # results = model(frame_rgb)

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
            # cv2.imshow('YOLO Object Detection', annotated_image_bgr)
        else:
            print("No results found")
            # Display the original frame if no results found
            # cv2.imshow('YOLO Object Detection', frame)
            annotated_image_bgr = frame
            
            annotations = []
            
    print("Yolo thread has stopped")

def display_frames(original_frame, modified_frame):
    global stop
    cv2.imshow("Original Frame", original_frame)
    cv2.imshow("Modified Frame", modified_frame)
    if annotated_image_bgr is not None:
        cv2.imshow('YOLO Object Detection', annotated_image_bgr)
    key = cv2.waitKey(1)
    if key == ord('q') or stop:
        cv2.destroyAllWindows()
        stop = True
        return False
    return True

modified_frame = None

def sendAndReceiveFrames():
    global stop
    
    webcamSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    webcamServerAddress = (address, ports[0])
    webcamSocket.connect(webcamServerAddress)

    cap = cv2.VideoCapture(0)

    while not stop:
        # Capture frame from webcam
        ret, tframe = cap.read()
        if not ret:
            break
        
        w = tframe.shape[1]
        h = tframe.shape[0]
        s = min(w, h)
        # crop width to center s 
        tframe = tframe[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2] 
        # resize to 512x512
        global frame
        frame = cv2.resize(tframe, (512, 512))
        
        # create yolo thread

        # Convert frame to JPEG format
        _, frame_data = cv2.imencode('.jpg', frame)

        # Get the size of the frame data
        frame_size = len(frame_data)

        # Pack the size of the frame data as a 4-byte integer
        size_data = struct.pack("I", frame_size)

        # Send the size of the frame data to the server
        webcamSocket.sendall(size_data)

        # Send the frame data to the server
        webcamSocket.sendall(frame_data)
        
        if stop:
            break

        # Receive the size of the modified frame data from the server
        size_data = webcamSocket.recv(4)
        if not size_data:
            break

        # Unpack the size of the modified frame data
        frame_size = struct.unpack("I", size_data)[0]
        
        if frame_size == 0:
            continue

        # Receive the modified frame data from the server
        frame_data = b''
        while len(frame_data) < frame_size:
            data = webcamSocket.recv(frame_size - len(frame_data))
            if not data:
                break
            frame_data += data

        # Convert frame data to numpy array
        global modified_frame
        modified_frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Display both original and modified frames
        # should_continue = display_frames(frame, modified_frame)
        # if not should_continue:
        #     break
        
        if stop:
            break

    print("Closing webcam socket")
    webcamSocket.close()
    cv2.destroyAllWindows()

textPrompt = "realistic, batman, mask"

def sendText():
    textSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    textServerAddress = (address, ports[1])
    textSocket.connect(textServerAddress)
    
    # print(annotations)

    while not stop:
        text = textPrompt.replace("\n", "")

        annotation = []
        cp = annotations
        if len(annotations) > 0:
            annotation = cp[0]

            # print("Sending annotation", json.dumps(annotation))

            # Send text to server
            textSocket.sendall((text + "<split>" + json.dumps(annotation) + "<end>").encode())
        
        if stop:
            break
        
        sleep(0.1)

    textSocket.close()
    
def changePrompt():
    global stop
    while not stop:        
        tmpTextPrompt = input("Enter text to send to server: ")
        global textPrompt
        print("Current prompt: ", tmpTextPrompt)
        textPrompt = tmpTextPrompt
        if textPrompt == "q":
            stop = True
            break
        

def loop_display():
    global stop
    while not stop:
        if annotated_image_bgr is not None:
            cv2.imshow('YOLO Object Detection', annotated_image_bgr)
            if modified_frame is not None:
                cv2.imshow("Modified Frame", modified_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or stop:
            cv2.destroyAllWindows()
            stop = True
            break

def main():
    
    args = sys.argv
    
    # get the address from the args
    global address
    if len(args) > 1:
        print("Address provided in args", args[1])
        address = args[1]
    
    yoloThread = threading.Thread(target=yolo_thread)
    webcamThread = threading.Thread(target=sendAndReceiveFrames)
    textThread = threading.Thread(target=sendText)
    changePromptThread = threading.Thread(target=changePrompt)
    displayThread = threading.Thread(target=loop_display)

    # Starting the threads
    yoloThread.start()
    textThread.start()
    webcamThread.start()
    changePromptThread.start()
    displayThread.start()

    # Waiting for both threads to finish
    webcamThread.join()
    textThread.join()
    yoloThread.join()
    changePromptThread.join()
    displayThread.join()

    print("All functions have finished executing")

if __name__ == "__main__":
    main()