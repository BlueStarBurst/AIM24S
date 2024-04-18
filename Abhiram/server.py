from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import socket
import cv2
import os
from time import sleep
import struct
import numpy as np

# MobileSAM initialization
model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

abspath = os.path.abspath(__file__) # sets directory of inference.py

predictor = SamPredictor(mobile_sam)

def modify_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ndarray-ify
    print("predicting...")
    predictor.set_image(frame)
    masks, _, _ = predictor.predict()
    # save mask
    # print(masks)
    # turn bool to int
    masks = masks.astype(int)
    # turn 0 and 1 to 0 and 255
    masks = masks * 255
    print("done")
    modified_frame = masks[0]
    return modified_frame

def send_receive_webcam_frames(connection):

    while True:
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

        # Modify the frame
        modified_frame = modify_frame(frame)

        # Convert modified frame to JPEG format
        _, modified_frame_data = cv2.imencode('.jpg', modified_frame)

        # Get the size of the modified frame data
        modified_frame_size = len(modified_frame_data)

        # Pack the size of the modified frame data as a 4-byte integer
        modified_size_data = struct.pack("I", modified_frame_size)

        # Send the size of the modified frame data to the client
        connection.sendall(modified_size_data)

        # Send the modified frame data to the client
        connection.sendall(modified_frame_data)

    connection.close()

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12345)

    server_socket.bind(server_address)
    server_socket.listen(1)

    print("Server is listening...")

    connection, client_address = server_socket.accept()

    send_receive_webcam_frames(connection)

    server_socket.close()

if __name__ == "__main__":
    main()