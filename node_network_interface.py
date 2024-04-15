from node import Node
from packet import Packet
import threading, time, socket
ip = 'localhost'

CORRECT_NODE_PORTS = {
    'A' : 6000,
    'B' : 6001,
    'C' : 6002,
    'D' : 6003,
    'E' : 6004,
    'F' : 6005,
    'G' : 6006,
    'H' : 6007,
    'I' : 6008,
    'J' : 6009
}

class NodeNetworkInterface():
    def __init__(self, id, listening_port, config_file):
        assert listening_port == CORRECT_NODE_PORTS[id]
        self.config_file = config_file
        self.listening_port = listening_port
        self.sending_ports = {}
        self.packet_count = 0
        self.timeouts = {}
        self.node = Node(id)

        self.sending_lock = threading.Lock()
        self.listening_lock = threading.Lock()

        self.load_from_config()
        self.start_threads()

    def load_from_config(self):
        with open(self.config_file, 'r') as f:
            lines = f.readlines() # it should be open for the minimum amount of time
        for line in lines[1:]: # ignore the first line
            line = line.strip().split(" ")
            n_id = line[0]
            n_cost = float(line[1])
            n_port = int(line[2])
            # may add a line into the config to
            # specify if the node is up or down
            self.sending_ports[n_id] = n_port
            self.timeouts[n_id] = time.time()
            self.node.set_neighbour_costs(n_id, n_cost)

    def start_threads(self):
        listening_thread = threading.Thread(target=self.listen)
        sending_thread = threading.Thread(target=self.broadcast)
        routing_calculations_thread = threading.Thread(target=self.routing_calculations)
        cli_listening_thread = threading.Thread(target=self.cli_listen)

        listening_thread.start()
        sending_thread.start()
        routing_calculations_thread.start()
        cli_listening_thread.start()

        listening_thread.join()
        sending_thread.join()
        routing_calculations_thread.join()
        cli_listening_thread.join()


    def listen(self):
        listening_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listening_socket.bind((ip, self.listening_port))
        self.listening_loop(listening_socket)

    def listening_loop(self, listening_socket):
        while True:
            self.listening_lock.acquire() # Block until the lock is free
            self.listening_lock.release()
            data, addr = listening_socket.recvfrom(1024)
            packet = Packet()
            packet.from_bits(data)
            t = time.strftime("%H:%M:%S")
            self.timeouts[packet.source] = time.time()
            print("received packet: ", packet.id, " at time: ", t)
            self.node.read_packet(packet)

    def broadcast(self):
        sending_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sending_loop(sending_socket)

    def sending_loop(self, sending_socket):
        # packet_send_time = time.time() + 10
        while True:
            self.sending_lock.acquire() # Block until the lock is free
            self.sending_lock.release()
            print("broadcasting...")
            self.send_packets(sending_socket)
            time.sleep(10)
            # current_time = time.time()
            # if current_time >= packet_send_time:
            #     packet_send_time = current_time + 10
            #     print("broadcasting...")
            #     self.send_packets(sending_socket)

            # elif current_time + 9 <= packet_send_time:
            #     time.sleep(8)
            #     # i dont trust that sleep(8) wont have some imprecision
                # that compounds as the program runs over time
                # this makes it so the loop isnt spamming so hard
                # when the packet send time is still several seconds away

    def send_packets(self, sending_socket):
        for n_id in self.node.neighbour_costs.keys():
            destination_port = self.sending_ports[n_id]
            packet = Packet()
            packet.data = self.node.reachability_matrix
            packet.source = self.node.id
            packet.id = self.packet_count

            t = time.strftime("%H:%M:%S")
            print("sending: ", packet.id, " to port :",destination_port, " time: ",t )
            sending_socket.sendto(packet.to_bits(), (ip, destination_port))
            self.packet_count += 1

    def routing_calculations(self):
        time.sleep(60)
        self.routing_calculations_loop()

    def routing_calculations_loop(self):
        reachability_matrix = self.node.reachability_matrix
        while True: # gotta be a better way to do this than while true
            # check if any of our neighbours have taken a suspiciously long amount of time to send us a packet
            suspiciously_long_time = 15 # 15 seconds
            for neighbour in self.timeouts.keys():
                print("time: ", time.time())
                print("timeouts: ", self.timeouts)
                if time.time() - self.timeouts[neighbour] > suspiciously_long_time:
                    print("node: ", neighbour, "has been detected as failed!")
                    self.node.neighbours_up[neighbour] = False
                else:
                    if not self.node.neighbours_up[neighbour]:
                        print("node", neighbour, "has been detected as recovered")
                    self.node.neighbours_up[neighbour] = True
            self.node.update_reachability_matrix()
            # if reachability_matrix != self.node.reachability_matrix: # only recalculate it if it's changed
            self.node.calculate_shortest_paths() 
            time.sleep(10)
            reachability_matrix = self.node.reachability_matrix


    def cli_listen(self):
        cli_port = self.listening_port + 100
        cli_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cli_socket.bind((ip, cli_port))
        self.cli_listen_loop(cli_socket)
    
    def cli_listen_loop(self, cli_socket):
        while True:
            data, addr = cli_socket.recvfrom(1024)
            command = data.decode()
            command = command.split()
            
            if command[0] == "FAIL":
                self.down = True
                self.sending_lock.acquire()
                self.listening_lock.acquire()
                print("Node deactivated by command line interface")

            elif command[0] == "RECOVER":
                self.down = False
                self.sending_lock.release()
                self.listening_lock.release()
                print("Node revived by command line interface")
            else:
                node_id = command[0]
                node_cost = int(command[1])
                node_port = command[2]
                if node_id in self.node.neighbours.keys():
                    if node_port in self.sending_ports.keys():
                        self.node.neighbour_costs[node_id] = node_cost
                        print("Successfully adjusted link cost to ", node_id, ". New cost: ", node_cost)
                    else:
                        print("Error: link cost adjustment command used incorrect port (", node_port,") for neighbour ", node_id)
                else:
                    print("Error: tried to adjust link cost with neighbour ", node_port, "but no such neighbour exists.")