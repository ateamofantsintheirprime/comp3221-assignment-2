import sys, os, torch, sklearn, socket, time, pickle, io, binascii
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
LEARNING_RATE = .01
INPUT_FEATURES = 8
opt_method = bool(sys.argv[3])
batch_size = 64 # ?

client_ports = {"client1": 6001, 
               "client2": 6002,
               "client3": 6003,
               "client4": 6004,
               "client5": 6005
}


def model_to_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

def model_from_bytes(bytes):

    buffer = io.BytesIO(bytes)
    state_dict = torch.load(buffer)

    model = nn.Linear(state_dict['weight'].shape[1], state_dict['weight'].shape[0])
    model.load_state_dict(state_dict)
    return model

def update_model(model):
    # Copy new model data to local model
    for old, new in zip(local_model.parameters(), model.parameters()):
        old.data = new.data.clone()
    print(f"updating local model to {model}")

def train(model):
    pass

def test(model):
    pass

def send_model(model):
    model_bytes = model_to_bytes(model)
    client_message = {"type": "model",
                "model" : model_bytes,
                "round" : round_number}
    send_socket.sendto(client_message, (ADDRESS,SERVER_PORT))

def get_training_data():
    pass

def get_testing_data():
    pass

def get_X():
    pass

def get_Y():
    pass

client_port = client_ports.get(client_id)


send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# send_socket.connect((ADDRESS, SERVER_PORT))
client_message = {"type": "handshake",
                "id" : client_id,
                "data_size" : batch_size}
message_bytes = pickle.dumps(client_message)

send_socket.sendto(message_bytes, (ADDRESS,SERVER_PORT)) 
print("sent handshake")
time.sleep(1)
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((ADDRESS,client_port))
# receive_socket.connect((ADDRESS, client_port))

# Set up data for training and testing.

train_data = get_training_data()
train_X = get_X(train_data)
train_Y = get_Y(train_data)
test_data = get_testing_data()
test_X = get_X(test_data)
test_y = get_Y(test_data)

train_data = [(x, y) for x, y in zip(train_X, train_Y)]
test_data = [(x, y) for x, y in zip(test_X, test_y)]

if not opt_method: # Set batch size to full training data size for regular GD
    batch_size = len(train_Y)

trainloader = DataLoader(train_data, batch_size=batch_size)
testloader = DataLoader(train_data, batch_size=len(test_y))

# Define loss function

loss = nn.MSELoss()

local_model = nn.Linear(INPUT_FEATURES, 1)


optimiser = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)

round_number = 0
while True:
    message, server_address = receive_socket.recvfrom(2048)
    message = pickle.loads(message)
    print("Received:", message)
    if not message or message["type"] == "finish":
        break
    if message["type"] == "model":
        print("received model from server")
        model = model_from_bytes(message['model'])
        update_model(model)
        round_number = message["round"]
        train(model)
        send_model(model)


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
