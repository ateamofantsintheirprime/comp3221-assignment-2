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
    print(f"getting local model from client {client_id}")
    if client_id not in clients.keys():
        print(f"error, client {client_id} attempted to send a model before handshaking")
        return
    clients[client_id].latest_model = message['model']
    clients[client_id].latest_round = message['round']

def wait_for_client_training(round_number):
    all_clients_completed = False
    while not all_clients_completed:
        all_clients_completed = True
        for c in clients.values():
            if c.latest_round != round_number:
                all_clients_completed = False
                time.sleep(1) # dont want to spam too hard
                break

def get_total_data_size() -> int:
    sum([c.data_size for c in clients.keys()])


def aggregate_models(round_number) -> LinearRegressionModel:
    # Zero out global model
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)
    
    # Not all clients will be active
    usable_clients = [c for c in clients.value() if c.active]
    
    # Get selection of clients for aggregating based on subsampling specifications
    aggregate_clients = usable_clients
    if sub_client != 0:
        # This line will cause issues if < K clients have provided a model this round. Consult specs
        aggregate_clients = random.sample(usable_clients, sub_client)

    # Aggregate sampled client models into global model
    for client in aggregate_clients():
        sample_size_ratio = client.data_size / get_total_data_size()
        global_model.data += client.latest_model.data.clone() * sample_size_ratio
    
    return global_model

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

def distribute_global_model(model, round_number):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    model_bytes = model_to_bytes(model)
    # model_hex = binascii.hexlify(model_bytes).decode('utf-8')

    message = pickle.dumps({
        "type" : "model",
        "model" : model_bytes,
        "round" : round_number
    })
    # print(message)
    for id in clients.keys():
        client = clients[id]
        sock.sendto(message, (ADDRESS, client.port))

if port != 6000:
    print("Port Server Must be 6000")


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ADDRESS, port))

#Waits until first connection is made and adds it to client list.

message, client_address = s.recvfrom(2048)
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

global_model = nn.Linear(INPUT_FEATURES, 1)

for i in range(COMMUNICATION_ROUNDS):
    distribute_global_model(global_model, i)
    wait_for_client_training(i)
    aggregate_models(i)

#After T training rounds are completed, send finish messages to clients and close sockets
send_finish_message()


# s.close()
