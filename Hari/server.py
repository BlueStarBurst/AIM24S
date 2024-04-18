import socket
import os

# Function to receive image from client
def receiveimage():
    # Create UDP socket
    server_socket = socket.socket(socket.AFINET, socket.SOCKDGRAM)

    # Bind socket to localhost and port 12345
    server_socket.bind(('127.0.0.1', 12345))

    print("Server is listening...")

    # Receive image size
    size, client_addr = server_socket.recvfrom(16000000)
    size = int(size.decode())

    # Receive image data
    data, client_addr = server_socket.recvfrom(16000000)
    image_data = data

    while len(image_data) < size:
        data, client_addr = server_socket.recvfrom(16000000)
        image_data += data

    print("Image received from client.")

    # Write image data to file
    with open("received_image.jpg", "wb") as f:
        f.write(image_data)

    print("Image saved as 'received_image.jpg'")

    # Close socket
    server_socket.close()

if __name__ == "__main__":
    receiveimage()