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
LOCAL_EPOCHS = 50
LEARNING_RATE = 0.0000001
INPUT_FEATURES = 8
opt_method = bool(sys.argv[3])
batch_size = 64 # ?
client_ports = {"client1": 6001, 
               "client2": 6002,
               "client3": 6003,
               "client4": 6004,
               "client5": 6005
}


# Define loss function

loss = nn.MSELoss(reduction='mean')

local_model = nn.Linear(INPUT_FEATURES, 1)


optimiser = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)

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

def train(epochs):
    local_model.train()
    for epoch in range(1, epochs + 1):
        local_model.train()
        for batch_idx, (X, y) in enumerate(trainloader):
            # print("BEGINNING NEW BATCH!!!!! =================")
            # print("model parameter gradients before zero_grad(): ")
            # for param in local_model.parameters():
            #     print(param.grad)
            # model.zero_grad()
            optimiser.zero_grad()
            output = local_model(X).reshape(-1)
            # print("output: ", output)
            # # y = y.unsqueeze(1)
            # print("y: ", y)
            loss_val = loss(output, y)
            # print("loss_val:", loss_val)
            loss_val.backward()
            # print("model parameter gradients before clipping: ")
            # for param in local_model.parameters():
            #     print(param.grad)
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 10)# print("model parameter gradients: ")
            # print("model parameter gradients after clipping: ")
            # for param in local_model.parameters():
            #     print(param.grad)
            # for param in local_model.parameters():
            #     print(param.grad)
            # print("\tmodel parameters BEFORE optimiser.step: ")
            # for param in local_model.parameters():
            #     print(param)
            optimiser.step()
            # print("\tmodel parameters AFTER optimiser.step: ")
            # for param in local_model.parameters():
            #     print(param)
            # print("model statedict after updating grads: ", model.state_dict())
    return loss_val.data
    

def test():
    local_model.eval()
    mse = 0
    for x, y in testloader:
        y_pred = local_model(x)
        y = y.unsqueeze(1)

        mse += loss(y_pred, y)
    return mse.tolist()

def send_model(model, round_number, test_MSE, train_MSE):
    
    model_bytes = model_to_bytes(model)
    
    client_message = pickle.dumps({"type": "model",
            "id" : client_id,
            "model" : model_bytes,
            "round" : round_number,
            "test_MSE": test_MSE,
            "train_MSE": train_MSE})

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

time.sleep(5)
send_socket.sendto(message_bytes, (ADDRESS,SERVER_PORT)) 
print("sent handshake")
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

round_number = 0
while True:
    # print("I am {}".format(client_id))
    message, server_address = receive_socket.recvfrom(2048)
    message = pickle.loads(message)
    # print("Received:", message)
    if not message or message["type"] == "finish":
        break
    
    if message["type"] == "model":
        # print("Received new global model")
        model = model_from_bytes(message['model'])
        update_model(model)
        round_number = message["round"]

        # print("model parameter gradients: ")
        # for param in local_model.parameters():
        #     print(param.grad)
        # # print("model parameters before training function: ================")
        # for param in local_model.parameters():
        #     print(param)
        train_MSE = train(LOCAL_EPOCHS)
        # print("Training MSE: {}".format(train_MSE))
        test_MSE = test()
        # print("Testing MSE: {}".format(test_MSE))

        # print("model parameters after training function: ================")
        # for param in local_model.parameters():
        #     print(param)

        log_input = "TestMSE: {}, TrainMSE: {}\n".format(test_MSE, train_MSE)
        file_name = "Logs/" + client_id + ".txt"
        with open(file_name, "a") as myfile:
            myfile.write(log_input)
        
        # print("Sending new local model")
        send_model(local_model, round_number, test_MSE, train_MSE)