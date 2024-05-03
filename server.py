import socket
import cv2
import struct
import numpy as np
import threading
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import os
import json
from streamdiff import streamdiffusion, setprompt

textPrompt = "Normal"


# get the local address of the server
address = "0.0.0.0"


# MobileSAM initialization
model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

abspath = os.path.abspath(__file__) # sets directory of inference.py

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

stop = False

predictor = SamPredictor(mobile_sam)

annotation = []

def modify_frame(frame):
    global annotation
    # print("Annotation:", annotation)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ndarray-ify
    predictor.set_image(frame)
    
    if len(annotation) > 0:
        masks, _, _ = predictor.predict(box=annotation)
        print("Annotation:", annotation)
    else:
        masks, _, _ = predictor.predict()
        
    # save mask
    # print(masks)
    # turn bool to int
    masks = masks.astype(np.uint8)
    # turn 0 and 1 to 0 and 255
    masks = masks * 255
    modified_frame = masks[0]
    return modified_frame

image_new = None
sam_new = None
diffusion_new = None

def sam_thread():
    global image_new
    global sam_new
    global diffusion_new
    while not stop:
        if image_new is not None:
            sam_start = cv2.getTickCount()
            sam_new = modify_frame(image_new)
            sam_end = cv2.getTickCount()
            sam_fps = cv2.getTickFrequency() / (sam_end - sam_start)
            # print("SAM FPS:", sam_fps)/
            # image_new = None
            
# TODO: potentially make sepearte streamdiffusion instances to cycle through + simplify streamdiffusion overhead (it said 90fps! why is it only 2fps on a H100!?!?!) D;
def diffusion_thread():
    global image_new
    global sam_new
    global diffusion_new
    while not stop:
        if sam_new is not None:
            diff_start = cv2.getTickCount()
            # try:
            diffusion_new = np.array(streamdiffusion(image_new, sam_new))
            # except Exception as e:
                # print("Error in streamdiffusion", e)
            diff_end = cv2.getTickCount()
            diff_fps = cv2.getTickFrequency() / (diff_end - diff_start)
            # print("DIFF FPS:", diff_fps)
            # sam_new = None

def send_receive_webcam_frames():
    global image_new
    global sam_new
    global diffusion_new
    webcamSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    webcamServerAddress = (address, 3000)
    webcamSocket.bind(webcamServerAddress)
    webcamSocket.listen(1)
    print("Server is listening...")
    connection, client_address = webcamSocket.accept()
    print("Client connected")

    while not stop:
        
        start_time = cv2.getTickCount()
        
        # Receive the size of the frame data from the client
        size_data = connection.recv(4)
        if not size_data:
            break

        # Unpack the size of the frame data
        frame_size = struct.unpack("I", size_data)[0]

        # Receive the frame data from the client
        frame_data = b''
        while len(frame_data) < frame_size:
            data = connection.recv(frame_size - len(frame_data))
            if not data:
                break
            frame_data += data
            
        

        # Convert frame data to numpy array
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        image_new = frame
        
        # send back diffusion new if not None
        if sam_new is not None:
            # send the mask back to the client 
            maskArr = np.array(sam_new)
            maskArr = np.stack((maskArr,)*3, axis=-1)
            blurred_mask_image = cv2.blur(maskArr, (35,35))
            
            _, frame_data = cv2.imencode('.jpg', blurred_mask_image)
            frame_size = len(frame_data)
            connection.send(struct.pack("I", frame_size))
            connection.send(frame_data)   
            
            
            
        # if diffusion_new is not None:
        #     # Convert frame to JPEG format
        #     _, frame_data = cv2.imencode('.jpg', diffusion_new)
        #     frame_size = len(frame_data)
        #     connection.send(struct.pack("I", frame_size))
        #     connection.send(frame_data)   
        else:
            # send frame size 0
            connection.send(struct.pack("I", 0))
            
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        # print("FPS:", fps)

    connection.close()
    webcamSocket.close()

def receiveText():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (address, 3001)

    server_socket.bind(server_address)
    server_socket.listen(1)

    connection, client_address = server_socket.accept()

    global textPrompt
    global annotation
    global stop
    tempData = ""
    text = ""
    annotations = []
    nextTempData = ""
    
    while not stop and text != "q":
        # read for text and annotations (ex: <start>text<split>annotation<end>)
        data = connection.recv(1024).decode()
        if not data:
            break
        tempData += data
        
        # print("TempData:", tempData)
        # print("Data:", data)
        
        if "<end>" in tempData:
            print("TempData:", tempData)
            text = tempData.split("<end>")[0]
            # print("Text:", text)
            
            
            # get very last end
            nextTempData = tempData.split("<end>")[-1]
            print("NextTempData:", nextTempData)
            
            if "<split>" in text:
                newTextPrompt = text.split("<split>")[0]
                tmp_annotation = text.split("<split>")[1]
                print("TMP Annotation:", tmp_annotation)
                floatArray = tmp_annotation.replace("[", "").replace("]", "").replace(" ", "").split(",")
                fake = False
                if len(floatArray) == 4:
                    for i in range(len(floatArray)):
                        if floatArray[i] == "":
                            fake = True
                            break
                        floatArray[i] = float(floatArray[i])
                    if not fake:
                        annotation = np.array(floatArray)
                        
                print("Annotation1:", annotation)
            else:
                newTextPrompt = text
                annotation = []
                print("Annotation2:", annotation)
                
            if newTextPrompt != textPrompt:
                setprompt(textPrompt, "bad quality, fake, not realistic")
                
            tempData = nextTempData
            

    stop = True
    connection.close()
    server_socket.close()

def main():
    webcamThread = threading.Thread(target=send_receive_webcam_frames)
    textThread = threading.Thread(target=receiveText)
    samThread = threading.Thread(target=sam_thread)
    diffusionThread = threading.Thread(target=diffusion_thread)

    webcamThread.daemon = True
    textThread.daemon = True
    samThread.daemon = True
    diffusionThread.daemon = True

    # Starting the threads
    textThread.start()
    webcamThread.start()
    samThread.start()
    diffusionThread.start()

    # Waiting for both threads to finish
    webcamThread.join()
    textThread.join()
    samThread.join()
    diffusionThread.join()

    print("All functions have finished executing")

if __name__ == "__main__":
    main()