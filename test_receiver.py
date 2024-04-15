import socket
ip = "127.0.0.1"
port = 6000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

packages_received = 0
ends_received = 0

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print("package received", packages_received, data)
    if data == b"end":
        ends_received += 1
    if ends_received >= 3:
        break
    packages_received += 1
print("received packages: ", packages_received)
