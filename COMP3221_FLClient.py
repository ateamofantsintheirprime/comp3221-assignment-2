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

def train(model, epochs, optimiser, loss, trainloader):
    model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (X, y) in enumerate(trainloader):
            optimiser.zero_grad()
            output = model(X)
            y = y.unsqueeze(1)
            loss_val = loss(output, y)
            loss_val.backward()
            optimiser.step()
    return model, loss_val.data, optimiser, loss, trainloader
    

def test(model, testloader):
    model.eval()
    mse = 0
    for x, y in testloader:
        y_pred = model(x)
        y = y.unsqueeze(1)

        mse += loss(y_pred, y)
    return model, mse, testloader

def send_model(model, round_number):
    
    model_bytes = model_to_bytes(model)
    
    client_message = pickle.dumps({"type": "model",
            "id" : client_id,
            "model" : model_bytes,
            "round" : round_number})

    """
    client_message = {"type": "model",
                "model" : model_bytes,
                "round" : round_number}
    """
    send_socket.sendto(client_message, (ADDRESS,SERVER_PORT))
    

"""
def get_training_data():
    pass

def get_testing_data():
    pass

def get_X():
    pass

def get_Y():
    pass
    
"""
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

client_port = client_ports.get(client_id)
train_X, test_X, train_Y, test_Y, train_samples, test_samples = get_data(client_id)

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# send_socket.connect((ADDRESS, SERVER_PORT))
client_message = {"type": "handshake",
                "id" : client_id,
                "data_size" : train_samples}
message_bytes = pickle.dumps(client_message)

send_socket.sendto(message_bytes, (ADDRESS,SERVER_PORT)) 
print("sent handshake")
time.sleep(1)
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((ADDRESS,client_port))
# receive_socket.connect((ADDRESS, client_port))

# Set up data for training and testing.
#train_X, test_X, train_Y, test_Y, train_samples, test_samples = get_data(client_id)

train_data = [(x, y) for x, y in zip(train_X, train_Y)]
test_data = [(x, y) for x, y in zip(test_X, test_Y)]

if not opt_method: # Set batch size to full training data size for regular GD
    batch_size = len(train_Y)

trainloader = DataLoader(train_data, batch_size=batch_size)
testloader = DataLoader(train_data, batch_size=len(test_Y))

# Define loss function

loss = nn.MSELoss()

local_model = nn.Linear(INPUT_FEATURES, 1)


optimiser = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)

round_number = 0
while True:
    print("I am {}".format(client_id))
    message, server_address = receive_socket.recvfrom(2048)
    message = pickle.loads(message)
    print("Received new global model")
    print("Received:", message)
    if not message or message["type"] == "finish":
        break
    
    if message["type"] == "model":
        print("Received new global model")
        model = model_from_bytes(message['model'])
        update_model(model)
        round_number = message["round"]

        model, test_MSE, testloader = test(model, testloader)
        print("Testing MSE: {}".format(test_MSE))

        model, train_MSE, optimiser, loss, trainloader = train(model, LOCAL_EPOCHS, optimiser, loss, trainloader)
        print("Training MSE: {}".format(train_MSE))

        log_input = "TestMSE: {}, TrainMSE: {}\n".format(test_MSE, train_MSE)
        file_name = "Logs/" + client_id + ".txt"
        with open(file_name, "a") as myfile:
            myfile.write(log_input)
        
        print("Sending new local model")
        send_model(model, round_number)
        round_number+=1

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