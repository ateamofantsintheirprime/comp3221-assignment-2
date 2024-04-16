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


client_list = [None, None, None, None, None]
temp_client_list = [None, None, None, None, None]

client_num = {"client1": 1, 
               "client2": 2,
               "client3": 3,
               "client4": 4,
               "client5": 5
}

client_ports = {"client1": 6001, 
               "client2": 6002,
               "client3": 6003,
               "client4": 6004,
               "client5": 6005
}

thirty_seconds_elapsed = False

#Adds the clients into a list 
def handle_client(packet, recv_socket, client_address):
    client_id = packet['id']
    client_data_size = packet['data_size']
    client_port = client_ports.get(client_id)
    print(f"getting handshake from client {client_id}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ADDRESS, client_port))
    s.listen()
    send_socket, client_address = s.accept()
    new_client = Client(client_id, recv_socket,send_socket, client_address, client_data_size)
    client_list[client_num.get(client_id)-1] = new_client
    print(packet)

def handle_late_client(packet, recv_socket, client_address):
    client_id = packet['id']
    client_data_size = packet['data_size']
    client_port = client_ports.get(client_id)
    print(f"getting handshake from client {client_id}")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ADDRESS, client_port))
    s.listen()
    send_socket, client_address = s.accept()
    new_client = Client(client_id, recv_socket,send_socket, client_address, client_data_size)
    temp_client_list[client_num.get(client_id)-1] = new_client

def update_client_list():
    for i in range(0,5):
        if client_list[i] == None:
            if temp_client_list[i] != None:
                client_list[i] = temp_client_list[i]

def print_client_list():
    for c in client_list:
        if c != None:
            print(c.get_client_id())
        else:
            print("None")


#Runs while the initil 30 seconds is over in a different thread to add more clients.
def add_additional_clients(message, recv_socket, client_address):
    while True:
        if thirty_seconds_elapsed:
            client_thread = threading.Thread(target=handle_late_client, args=(message, recv_socket, client_address))
            client_thread.start()

        else:
            client_thread = threading.Thread(target=handle_client, args=(message, recv_socket, client_address))
            client_thread.start()

def send_finish_message(client_list):
    message = "Training Completed"
    for c in client_list:
        if c:
            c.get_send_socket().sendall(message.encode())

def close_sockets(client_list):
    for c in client_list:
        if c:
            c.get_send_socket().close()
            c.get_recv_socket().close()

def receive_packet(socket):
    recv_socket, client_address = s.accept()
    message = pickle.loads(recv_socket.recv(1024))
    if message["type"] == "handshake":
        add_additional_clients(message, recv_socket, client_address)
    if message["type"] == "model":
        add_model(message["model"])

def listening_loop(socket):
    while True:
        receive_packet(socket)

def add_model(message):
    client_id = message['id']
    model = message['model']
    print(f"getting local model from client {client_id}")

def distribute_global_model(model):
    message = pickle.dumps({
        "type" : "model",
        "model" : model
    })
    for client in client_list:
        pass #TODO

if port != 6000:
    print("Port Server Must be 6000")

global_model = LinearRegressionModel(INPUT_FEATURES)
#gets w0

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ADDRESS, port))
s.listen()

#Waits until first connection is made and adds it to client list.
recv_socket, client_address = s.accept()
client_thread = threading.Thread(target=receive_packet, args=(s))
client_thread.start()

#Once the first connection is made, the program waits 30 seconds before moving on.
#5 is just the placeholder time

s.settimeout(5)
try:
    while True:
        recv_socket, client_address = s.accept()
        client_thread = threading.Thread(target=receive_packet, args=(s))
        client_thread.start()
except socket.timeout:
    print("30 Seconds Elapsed Since First Connection")


thirty_seconds_elapsed = True
s.settimeout(None)
listen_thread = threading.Thread(target=listening_loop, args=(s,))
listen_thread.start()


print("Starting Federated Learning Now")
#do federated learning here
time.sleep(1)


#After T training rounds are completed, send finish messages to clients and close sockets
send_finish_message(client_list)
close_sockets(client_list)


s.close()
