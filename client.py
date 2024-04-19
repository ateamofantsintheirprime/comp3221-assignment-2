
class Client:
    def __init__(self, port):
        self.data_size = 0
        self.port = port
        self.active = False
        self.in_queue = False
        self.latest_model = None
        self.latest_round = -1
        self.train_MSE = -1
        self.test_MSE = -1
