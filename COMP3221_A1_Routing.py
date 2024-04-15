import sys, os
from node_network_interface import NodeNetworkInterface


"""
    TODO:
        - Listening thread, always waiting to recieve packets
            and stores the info somewhere
        - Sending thread, periodically (every 10 seconds)
            broadcasts information packets to all neighbours.
            (this could be the link costs to neighbours or the
            whole routing table or both)
        - Routing Calculations thread, whenever a change happens
            (link cost updates, node goes down, new node is added)
            the routing table should be recalculated. Maybe this
            can trigger a rebroadcast of the routing table.
        - Network startup behaviours, each node only initially
            has info about it's connection to its neighbours.
            so the routing table is largely a big question mark.
            it must start up all its threads and broadcast its
            incomplete routing table and wait for their neighbours
            to do the same. Once its receives the routing table of
            its neighbours, it should update its with the new info

            Note: from the assignment specifications:
                ' The routing table must be constructed AFTER
                receiving update packets from the entire network '

                also:

                ' Wait 60 seconds after startup for the routing
                algorithm to run, turning the reachability matrix/
                routing table, into a list of shortest paths to be
                printed to terminal '

        - CLI, let users edit network topological properties and
            have this be reflected in config files and the relevant
            nodes

"""


config_file =  os.path.join(os.getcwd(), "configs", sys.argv[3])
n = NodeNetworkInterface(sys.argv[1], int(sys.argv[2]), config_file)
