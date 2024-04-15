import pickle

class Packet():
    def __init__(self):
        self.source = None
        self.data = None
        self.id = 0

    def to_bits(self):
        return pickle.dumps({"source" : self.source,
                             "data" : self.data,
                             "id": self.id})
    def from_bits(self, bits):
        data = pickle.loads(bits)
        self.source = data["source"]
        self.data = data["data"]
        self.id = data["id"]
    def get_source(self):       return self.source
    def get_data(self):      return self.data
