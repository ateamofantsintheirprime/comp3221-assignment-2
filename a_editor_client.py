import socket

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
client_address = ('localhost', 6104)
sock.bind(client_address)

try:
    # Receive message from the server
    print("Waiting to receive message from server...")
    data, server_address = sock.recvfrom(4096)
    print(f"Received message from server on port 6100: {data.decode()}")

finally:
    # Close the socket
    sock.close()