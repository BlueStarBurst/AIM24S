import socket

def receive_text():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12346)

    server_socket.bind(server_address)
    server_socket.listen(1)

    print("Server is listening...")

    connection, client_address = server_socket.accept()

    # Receive text from client
    text = connection.recv(1024).decode()

    print("Received text from client:", text)

    connection.close()
    server_socket.close()

if __name__ == "__main__":
    receive_text()