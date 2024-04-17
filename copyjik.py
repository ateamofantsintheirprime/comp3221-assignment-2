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
opt_method = sys.argv[3]
data_size = 10

##########################
x_train = None
x_test = None
y_test = None
y_train = None
train_sample = 0
test_sample = 0
train_data = None
test_data = None
##########################

client_ports = {"client1": 6001, 
               "client2": 6002,
               "client3": 6003,
               "client4": 6004,
               "client5": 6005
}

##########################
##########################
def get_data(client):
    train_MedInc = []
    train_HouseAge = []
    train_AveRooms = []
    train_AveBedrms = []
    train_Population = []
    train_AveOccup = []
    train_Latitude = []
    train_Longitude = []
    train_MedHouseVal = []

    test_MedInc = []
    test_HouseAge = []
    test_AveRooms = []
    test_AveBedrms = []
    test_Population = []
    test_AveOccup = []
    test_Latitude = []
    test_Longitude = []
    test_MedHouseVal = []

    filename = "FLData/calhousing_train_" + client + ".csv"
    with open(filename, "r") as file:
        #ignores the first line of names of dataset
        next(file)
        for line in file:
            split_data = line.split(",")
            train_MedInc.append(float(split_data[0]))
            train_HouseAge.append(float(split_data[1]))
            train_AveRooms.append(float(split_data[2]))
            train_AveBedrms.append(float(split_data[3]))
            train_Population.append(float(split_data[4]))
            train_AveOccup.append(float(split_data[5]))
            train_Latitude.append(float(split_data[6]))
            train_Longitude.append(float(split_data[7]))
            train_MedHouseVal.append(float(split_data[8]))

    filename = "FLData/calhousing_test_" + client + ".csv"
    with open(filename, "r") as file:
        #ignores the first line of names of dataset
        next(file)
        for line in file:
            split_data = line.split(",")
            test_MedInc.append(float(split_data[0]))
            test_HouseAge.append(float(split_data[1]))
            test_AveRooms.append(float(split_data[2]))
            test_AveBedrms.append(float(split_data[3]))
            test_Population.append(float(split_data[4]))
            test_AveOccup.append(float(split_data[5]))
            test_Latitude.append(float(split_data[6]))
            test_Longitude.append(float(split_data[7]))
            test_MedHouseVal.append(float(split_data[8]))

    x_train_np = np.column_stack((train_MedInc, train_HouseAge, train_AveRooms, train_AveBedrms, train_Population, train_AveOccup, train_Latitude, train_Longitude))
    x_test_np = np.column_stack((test_MedInc, test_HouseAge, test_AveRooms, test_AveBedrms, test_Population, test_AveOccup, test_Latitude, test_Longitude))
    
    x_train = torch.Tensor(x_train_np).view(-1,8).type(torch.float32)
    x_test = torch.Tensor(x_test_np).view(-1,8).type(torch.float32)
    
    y_train = torch.Tensor(train_MedHouseVal).type(torch.float32)
    y_test = torch.Tensor(test_MedHouseVal).type(torch.float32)

    return x_train, x_test, y_train, y_test, len(train_MedHouseVal), len(test_MedHouseVal)
##########################
##########################

def update_model(model):
    pass

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

##########################
x_train, x_test, y_train, y_test, train_sample, test_sample = get_data(client_id)



"""
while True:
    message, server_address = receive_socket.recvfrom(2048)
    message = pickle.loads(message)
    print("Received:", message)
    if not message or message["type"] == "finish":
        break
    if message["type"] == "model":
        print("received model from server")
        update_model(message['id'], message['data_size'])

"""


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
