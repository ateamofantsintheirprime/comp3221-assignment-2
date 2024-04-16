import sys, os, torch, sklearn, socket, time, pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LinearRegressionModel import *

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import os
import json
import copy
import random
import numpy as np


client_id = sys.argv[1]
client_port = int(sys.argv[2])
ADDRESS = '127.0.0.1'
SERVER_PORT = 6000
LOCAL_EPOCHS = 4
opt_method = sys.argv[3]
data_size = 10

client_ports = {"client1": 6001, 
               "client2": 6002,
               "client3": 6003,
               "client4": 6004,
               "client5": 6005
}

def update_model(model):
    pass


client_port = client_ports.get(client_id)


send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# send_socket.connect((ADDRESS, SERVER_PORT))
client_message = {"type": "handshake",
                "id" : client_id,
                "data_size" : data_size}
message_bytes = pickle.dumps(client_message)

send_socket.sendto(message_bytes, (ADDRESS,SERVER_PORT)) 
print("sent handshake")
time.sleep(1)
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((ADDRESS,client_port))
# receive_socket.connect((ADDRESS, client_port))

while True:
    message, server_address = receive_socket.recvfrom(1024)
    message = pickle.loads(message)
    print("Received:", message)
    if not message or message["type"] == "finish":
        break
    if message["type"] == "model":
        print("received model from server")
        update_model(message['id'], message['data_size'])


time.sleep(1)

# send_socket.close()
# receive_socket.close()







# class Client():
#     def __init__(self, id, port, opt_method):
#         assert id in ['client1','client2','client3','client4','client5']
#         assert port in range(6001,6006)
#         assert opt_method in [0,1]
#         self.id = id
#         self.port = port
#         self.opt_method = opt_method

#         self.loss = nn.MSELoss()
#         self.model = self.await_global_model()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


#     def set_parameters(self, model):
#         for old_param, new_param in zip(self.model.parameters(), model.parameters()):
#             old_param.data = new_param.data.clone()

#     def await_global_model(self) -> LinearRegressionModel:
#         pass

#     def train(self):
#         pass
        


# # config_file =  os.path.join(os.getcwd(), "configs", sys.argv[3])
# n = Client(id=sys.argv[1], port=sys.argv[2], opt_method=sys.argv[3])
