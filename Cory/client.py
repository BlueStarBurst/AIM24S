import socket
import cv2
import struct
import numpy as np

def display_frames(original_frame, modified_frame):
    cv2.imshow("Original Frame", original_frame)
    cv2.imshow("Modified Frame", modified_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        return False
    return True

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12345)

    client_socket.connect(server_address)

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to JPEG format
        _, frame_data = cv2.imencode('.jpg', frame)

        # Get the size of the frame data
        frame_size = len(frame_data)

        # Pack the size of the frame data as a 4-byte integer
        size_data = struct.pack("I", frame_size)

        # Send the size of the frame data to the server
        client_socket.sendall(size_data)

        # Send the frame data to the server
        client_socket.sendall(frame_data)

        # Receive the size of the modified frame data from the server
        size_data = client_socket.recv(4)
        if not size_data:
            break

        # Unpack the size of the modified frame data
        frame_size = struct.unpack("I", size_data)[0]

        # Receive the modified frame data from the server
        frame_data = b''
        while len(frame_data) < frame_size:
            data = client_socket.recv(frame_size - len(frame_data))
            if not data:
                break
            frame_data += data

        # Convert frame data to numpy array
        modified_frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Display both original and modified frames
        should_continue = display_frames(frame, modified_frame)
        if not should_continue:
            break

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()