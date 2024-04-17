import sys, os, torch, socket, threading, copy, random, time, pickle, io, binascii
import torch.nn as nn 

import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from LinearRegressionModel import *
from client import Client


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


COMMUNICATION_ROUNDS = 100
LEARNING_RATE = .01
BATCH_SIZE = 64
CLIENT_COUNT = 5 # dont change this
ADDRESS = '127.0.0.1'
INPUT_FEATURES = 8

port = int(sys.argv[1])
sub_client = sys.argv[2]


clients = {
    'client1' : Client(6001),
    'client2' : Client(6002),
    'client3' : Client(6003),
    'client4' : Client(6004),
    'client5' : Client(6005)
}


# thirty_seconds_elapsed = False

# def update_client_list():
#     for i in range(0,5):
#         if client_list[i] == None:
#             if temp_client_list[i] != None:
#                 client_list[i] = temp_client_list[i]

# def print_client_list():
#     for c in client_list:
#         if c != None:
#             print(c.get_client_id())
#         else:
#             print("None")


def send_finish_message():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = pickle.dumps({"type": "finish"})
    print("Sending finish message")
    for id in clients.keys():
        client = clients[id]
        sock.sendto(message, (ADDRESS,client.port))

def add_client(id, data_size):
    if clients[id] == True:
        raise Exception
    clients[id].active = True
    clients[id].in_queue = True
    clients[id].data_size = data_size
    print(f"Received handshake from client {id}")

def receive_packet(socket):
    message, client_address = socket.recvfrom(1024)
    message = pickle.loads(message)
    if message["type"] == "handshake":
        add_client(message['id'], message['data_size'])
    if message["type"] == "model":
        add_model(message["model"])

def listening_loop(socket):
    while True:
        receive_packet(socket)

def add_model(message):
    client_id = message['id']
    model = message['model']
    model_round = message['round']
    print(f"getting local model from client {client_id}")
    clients[client_id].latest_model = model
    clients[client_id].latest_round = model_round

def aggregate_models(round_number) -> LinearRegressionModel:
    models = []
    for id in clients.keys():
        client = clients[id]
        if client.active and client.latest_round == round_number:
            models.append(client.latest_model)
    
    ### do some maths to aggregate the models

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

def distribute_global_model(model):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    model_bytes = model_to_bytes(model)
    # model_hex = binascii.hexlify(model_bytes).decode('utf-8')

    message = pickle.dumps({
        "type" : "model",
        "model" : model_bytes
    })
    print(message)
    for id in clients.keys():
        client = clients[id]
        sock.sendto(message, (ADDRESS, client.port))

if port != 6000:
    print("Port Server Must be 6000")


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ADDRESS, port))

#Waits until first connection is made and adds it to client list.

message, client_address = s.recvfrom(1024)
message = pickle.loads(message)
if message["type"] == "handshake":
    add_client(message['id'], message['data_size'])
if message["type"] == "model":
    add_model(message)

#Once the first connection is made, the program waits 30 seconds before moving on.
#5 is just the placeholder time

s.settimeout(5)

# thirty_seconds_elapsed = True
s.settimeout(None)

listen_thread = threading.Thread(target=listening_loop, args=(s,))
listen_thread.start()

print("Starting Federated Learning Now")
#do federated learning here
time.sleep(1)

# global_model = LinearRegressionModel(INPUT_FEATURES)
global_model = nn.Linear(INPUT_FEATURES, 1)
distribute_global_model(global_model)
# Load the tensor from the byte array
# w = torch.from_buffer(weight_bytes, dtype=torch.float32)
# b = torch.from_buffer(bias_bytes, dtype=torch.float32)

# print("Loaded weights:")
# print(w)
# print("Loaded biases:")
# print(b)


#After T training rounds are completed, send finish messages to clients and close sockets
send_finish_message()


# s.close()
