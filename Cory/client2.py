import socket

def send_text():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12346)

    client_socket.connect(server_address)

    # Get text input from user
    text = input("Enter text to send to server: ")

    # Send text to server
    client_socket.sendall(text.encode())

    client_socket.close()

if __name__ == "__main__":
    send_text()