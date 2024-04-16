class Client:
    def __init__(self, client_id, recv_socket, send_socket, address, datasize):
        self.client_id = client_id
        self.datasize = datasize
        self.recv_socket = recv_socket
        self.send_socket = send_socket
        self.address = address

    def get_client_id(self):
        return self.client_id
    
    def get_send_socket(self):
        return self.send_socket
    
    def get_recv_socket(self):
        return self.recv_socket
