class Client:
    def __init__(self, id, port):
        self.id = id
        self.data_size = 0
        self.port = port
        self.active = False
        self.latest_model = None
