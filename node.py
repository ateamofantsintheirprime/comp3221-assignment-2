from packet import Packet

class Node():
    def __init__(self, id):
        self.up = True # https://edstem.org/au/courses/15078/discussion/1775924
            # we assume that all nodes start up successfully
        self.id = id
        self.neighbour_costs = {}
        self.neighbours_up = {}
        self.reachability_matrix = {self.id : {}}
        pass

    def set_neighbour_costs(self, key, cost):
        self.neighbour_costs[key] = cost
        self.neighbours_up[key] = True
        self.reachability_matrix = {self.id : self.neighbour_costs}
        for neighbour in self.neighbour_costs.keys():
            if neighbour in self.reachability_matrix.keys():
                self.reachability_matrix[neighbour][self.id] = self.neighbour_costs[neighbour]
            else:
                self.reachability_matrix[neighbour] = {self.id : self.neighbour_costs[neighbour]}

    def calculate_shortest_paths(self):
        # Dijkstra's algorithm
        dist = {}
        path = {}
        q = []
        for v in self.reachability_matrix.keys():
            dist[v] = 99999999
            path[v] = []
            q.append(v)

        dist[self.id] = 0
        path[self.id] = [self.id]

        while len(q) > 0:
            u = min(q, key=lambda x: dist[x])
            q.remove(u)

            for v in self.reachability_matrix[u]:
                alt = dist[u] + self.reachability_matrix[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    path[v] = path[u] + [v]

        print("This is node ", self.id)
        for node in dist.keys():
            if node != self.id:
                print("Least cost path from ", self.id, " to ", node, ": ", end = "")
                print("".join(path[node]), ", link cost: ", dist[node])

    def get_active_neighbour_costs(self):
        # get the dictionary of neighbour costs, but only those neighbours that are up
        active_neighbour_costs = {}
        for neighbour in self.neighbour_costs.keys():
            if self.neighbours_up[neighbour]:
                active_neighbour_costs[neighbour] = self.neighbour_costs[neighbour]
        return active_neighbour_costs

    def update_reachability_matrix(self, new_matrix = None):
        # Receive the matrix update from neighbour.
        # We want to incorperate this info into our own matrix.
        # However we want to ignore what they say about our own neighbour link costs
        if new_matrix == None:
            new_matrix = self.reachability_matrix
        self.reachability_matrix = new_matrix  # Incorperate what they say
        active_neighbour_costs = self.get_active_neighbour_costs()
        self.reachability_matrix[self.id] = active_neighbour_costs # Overwrite what they say about us
        for neighbour in active_neighbour_costs.keys():
            if neighbour in self.reachability_matrix.keys():
                self.reachability_matrix[neighbour][self.id] = active_neighbour_costs[neighbour]
            else:
                self.reachability_matrix[neighbour] = {self.id : active_neighbour_costs[neighbour]}
        print("reachability matrix:", self.reachability_matrix)
        print("neighbour costs:", active_neighbour_costs)

    def read_packet(self, packet):
        d = packet.get_data()
        self.update_reachability_matrix(d)
