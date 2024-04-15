import socket, time
ip = "127.0.0.1"
port = 6000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

current_time = time.time()

start_time = current_time - (current_time%5) + 10

time_to_wait = start_time - current_time

print("current time", time.time())
print("starting at ", start_time)
print("sleeping for ", time_to_wait)
time.sleep(time_to_wait)
print("starting...")

for i in range(10000):
    sock.sendto(b".", (ip, port))
sock.sendto(b"end", (ip, port))
print("sent 10000 packages")
