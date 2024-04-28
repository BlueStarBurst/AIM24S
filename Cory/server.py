import socket
import cv2
import struct
import numpy as np
import threading
import json

textPrompt = "Normal"

# Function to handle YOLO annotations
def handle_annotations(annotations):
    # Process YOLO annotations here
    print("Received YOLO annotations:", annotations)

# Function to handle user input
def handle_user_input(user_input):
    global textPrompt
    # Check if the user input is tagged as text or annotation
    if user_input.startswith("T:"):
        # It's user text input
        text = user_input[2:]  # Remove the "T:" prefix
        print("Received text from client:", text)
        textPrompt = str(text)
    else:
        # It's YOLO annotations
        annotations = json.loads(user_input)
        handle_annotations(annotations)

def receive_data(client_socket):
    data = b""
    while True:
        packet = client_socket.recv(1024)
        if not packet:
            break
        data += packet
    return data

def receiveText(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        user_input = data.decode()
        handle_user_input(user_input)

def send_receive_webcam_frames():
    webcamSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    webcamServerAddress = ('127.0.0.1', 12345)
    webcamSocket.bind(webcamServerAddress)
    webcamSocket.listen(1)
    print("Server is listening for webcam frames...")
    connection, client_address = webcamSocket.accept()

    while True:
        # Receive the size of the frame data from the client
        size_data = connection.recv(4)
        if not size_data:
            break

        # Unpack the size of the frame data
        frame_size = struct.unpack("I", size_data)[0]

        # Receive the frame data from the client
        frame_data = receive_data(connection)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Dummy modification for demonstration
        modified_frame = cv2.flip(frame, 1)

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
    webcamSocket.close()

def main():
    textSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    textServerAddress = ('127.0.0.1', 54321)
    textSocket.bind(textServerAddress)
    textSocket.listen(1)
    print("Server is listening for text input...")

    webcamThread = threading.Thread(target=send_receive_webcam_frames)

    while True:
        client_socket, client_address = textSocket.accept()
        print("Received connection from:", client_address)
        textThread = threading.Thread(target=receiveText, args=(client_socket,))
        textThread.start()

    textSocket.close()

if __name__ == "__main__":
    main()
