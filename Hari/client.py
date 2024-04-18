import socket

#Function to send image to server
def sendimage():
    # Create UDP socket
    client_socket = socket.socket(socket.AFINET, socket.SOCKDGRAM)

    # Specify server address and port
    server_address = ('127.0.0.1', 12345)
 
    # Read image file
    with open(r"C:\Users\creaz\source\repos\AIM24S\Cory\Letter C Profile Picture.jpg", "rb") as f:
        image_data = f.read()

    # Send image size
    client_socket.sendto(str(len(image_data)).encode(), server_address)

    # Send image data
    client_socket.sendto(image_data, server_address)

    print("Image sent to server.")

    # Close socket
    client_socket.close()

if __name__ == "__main__":
    sendimage()