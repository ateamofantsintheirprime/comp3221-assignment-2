import sys, os, torch, socket, threading, copy, random, time, pickle
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

class Client:
    def __init__(self, port):
        self.data_size = 0
        self.port = port
        self.active = False
        self.in_queue = False
        self.latest_model = None
        self.latest_round = 0


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

def distribute_global_model(model):
    message = pickle.dumps({
        "type" : "model",
        "model" : model
    })
    for id in clients.keys():
        client = clients[id]
        pass #TODO

if port != 6000:
    print("Port Server Must be 6000")


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ADDRESS, port))
# s.listen()

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
#gets w0
print(f"global model: {global_model.weight}")
print(f"global model: {global_model.bias}")


#After T training rounds are completed, send finish messages to clients and close sockets
send_finish_message()


# s.close()
