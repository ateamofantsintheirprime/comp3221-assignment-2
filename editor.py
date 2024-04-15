import socket, time

#checks if input is a decimal number
def is_decimal(input_str):
    try:
        float(input_str)
        return True
    except ValueError:
        return False
    
#function not used, was used before to remove node from config file. 
#just incase it needs to be used again
def remove_connection(config_node, remove_node):
    if config_node == remove_node:
        return
    else:
        c_filename = "configs/" + config_node + "config.txt"
        
        try:
            c_file = open(c_filename, "r+")
            c_file_lines = c_file.readlines()
            c_file.seek(0)
            c_file.truncate()

            for line in c_file_lines:
                if line[0] == remove_node:
                    pass
                else:
                    c_file.writelines(line)
            c_file.close()

        except FileNotFoundError:
            return
    

#while (1): 
#program only works for one edit for now 
user_command = input()
user_commands = user_command.split()

start_port = 6100
valid_nodeid = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

#the syntax for cost change is CHANGE NODE_1 NODE_2 NEW_COST
if user_commands[0] == "CHANGE":
    
    #ensures that both nodes are valid and the new cost is a valid decimal number
    if user_commands[1] in valid_nodeid and user_commands[2] in valid_nodeid and is_decimal(user_commands[3]):

        c_file1_name = "configs/" + user_commands[1] + "config.txt"
        c_file2_name = "configs/" + user_commands[2] + "config.txt"
        
        #Check if the node connection exists in the config file. 
        #No need to check both files as they should have the same connections. 
        connection_exists = False
        try:
            with open(c_file1_name, "r") as c_file1:
                c_file1_lines = c_file1.readlines()
                for line in c_file1_lines:
                    if line[0] == user_commands[2]:
                        connection_exists = True
        except FileNotFoundError:
            print("FILE NOT FOUND")

        #If connection exists, both config files are searched for the line detailing the costs, and is changed 
        if connection_exists == True:
            c_file1_input = user_commands[2] + " " +  user_commands[3]
            c_file1 = open(c_file1_name, "w")
            
            for line in c_file1_lines:
                if line[0] == user_commands[2]:
                    split_line = line.split()
                    c_file1_port = split_line[2]
                    c_file1_input = c_file1_input + " " + c_file1_port + "\n"
                    c_file1.writelines(c_file1_input)
                else:
                    c_file1.writelines(line)
            c_file1.close()

            
            c_file2 = open(c_file2_name, "r+")
            c_file2_lines = c_file2.readlines()
            c_file2.seek(0)
            c_file2_input = user_commands[1] + " " + user_commands[3]
            for line in c_file2_lines:
                if line[0] == user_commands[1]:
                    split_line = line.split()
                    c_file2_port = split_line[2]
                    c_file2_input = c_file2_input + " " + c_file2_port + "\n"
                    c_file2.writelines(c_file2_input)
                else:
                    c_file2.writelines(line)
            c_file2.close()

            #sends message to NODE_1 and NODE_2 ports. 
            #ports are calculated by 6100 + node index
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            node1_index = valid_nodeid.index(user_commands[1])
            node2_index = valid_nodeid.index(user_commands[2])
            node1_address = ('localhost', 6100+node1_index)
            node2_address = ('localhost', 6100+node2_index)
            node1_message = c_file1_input
            node2_message = c_file2_input
            
            try:
            # Send the message to the client
                sock.sendto(node1_message.encode(), node1_address)
                sock.sendto(node2_message.encode(), node2_address)

            finally:
                # Close the socket
                sock.close()
    
        else:
            print("INVALID CONNECTION")


    else:
        print("Invalid Change Command")

#sends fail or recover message to port. 
elif user_commands[0] == "FAIL" or user_commands[0] == "RECOVER":

    target_node = user_commands[1]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    kill_node_index = valid_nodeid.index(target_node)
    print(start_port + kill_node_index)
    node_address = ('localhost', start_port + kill_node_index)

    node_status_message = user_commands[0] + " " + target_node
    
    try:
    # Send the message to the client
        sock.sendto(node_status_message.encode(), node_address)

    finally:
        # Close the socket
        sock.close()

    # GO TO ALL THE CONFIG FILES AND REMOVE THE CONNECTION WITHIN THE FILE

else:
    print("INVALID COMMAND")

print("end reached")
