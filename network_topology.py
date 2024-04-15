import random, os

def generate_network_topology(node_count = 10, link_count = 15):
    assert node_count <= link_count
    node_ids = ["B", "C", "D", "E", "F", "G", "H", "I", "J"]
    nodes_in_network = ["A"]
    links = []
    while len(node_ids) > 0:
        n1 = node_ids.pop(0)
        links.append({n1, random.choice(nodes_in_network)})
        nodes_in_network.append(n1)
    while len(links) < link_count:
        n1 = random.choice(nodes_in_network)
        n2 = random.choice(nodes_in_network)
        if n1 != n2 and not {n1,n2} in links:
            links.append({n1,n2}) 
    node_links = {node : [] for node in nodes_in_network}
    for l in [list(li) for li in links]:
        cost = random.randint(5,100)/10.0
        node_links[l[0]].append([l[1], cost])
        node_links[l[1]].append([l[0], cost])
    return node_links

def config_files(links, directory = os.path.join(os.getcwd(), "configs")):
    node_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    node_ports = {node_ids[i]: 6000 + i for i in range(len(node_ids))}

    if not os.path.exists(directory):
        os.mkdir(directory)

    for node in node_ids:
        config_string = str(len(links[node])) + "\n"
        for link_recipient in links[node]:
            config_string += f"{link_recipient[0]} {link_recipient[1]} {str(node_ports[link_recipient[0]])}\n"
        filename = node+"config.txt"
        with open(os.path.join(directory, filename), 'w') as file:
            file.write(config_string)

config_files(generate_network_topology())
