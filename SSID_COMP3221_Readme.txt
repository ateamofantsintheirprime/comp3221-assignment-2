# comp3221-assignment-2

# Readme (A detailed .txt file that outlines the coding environment, version of packages
used, instructions to run your program, and commands to reproduce the experimental
results.

--------------------------------------------------------
Coding Environment and Version of Packages
bottle==0.12.23
click==8.1.3
contourpy==1.2.1
cycler==0.12.1
filelock==3.13.4
Flask==2.2.2
fonttools==4.51.0
fsspec==2024.3.1
importlib-metadata==5.0.0
importlib_resources==6.4.0
itsdangerous==2.1.2
Jinja2==3.1.2
joblib==1.4.0
kiwisolver==1.4.5
MarkupSafe==2.1.1
matplotlib==3.8.4
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.4
packaging==24.0
pillow==10.3.0
pycryptodome==3.17
pyparsing==3.1.2
python-dateutil==2.9.0.post0
scikit-learn==1.4.2
scipy==1.13.0
six==1.16.0
sympy==1.12
threadpoolctl==3.4.0
torch==2.2.2
typing_extensions==4.11.0
Werkzeug==2.2.2
zipp==3.9.0
--------------------------------------------------------


--------------------------------------------------------
Instructions to Run Program
To run the program, the syntax detailed in the specification sheet can be used as in:
- python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method> for the Client
- python COMP3221_FLServer.py <Port-Server> <Sub-Client> for the Server

Alternatively, the server + 5 client instance of the program can be ran (no subsampling) by doing 
"bash run.sh"
--------------------------------------------------------


--------------------------------------------------------
Commands to Reproduce Experimental results

The data in the final table of the report was generated using the following commands:

    Gradient Descent Column, K=1,2,3,4,0:
    ```
    clear;python3 COMP3221_FLServer.py 6000 1 & python3 COMP3221_FLClient.py client1 6001 0 &  python3 COMP3221_FLClient.py client2 6002 0 & python3 COMP3221_FLClient.py client3 6003 0 &  python3 COMP3221_FLClient.py client4 6004 0 & python3 COMP3221_FLClient.py client5 6005 0;
    clear;python3 COMP3221_FLServer.py 6000 2 & python3 COMP3221_FLClient.py client1 6001 0 &  python3 COMP3221_FLClient.py client2 6002 0 & python3 COMP3221_FLClient.py client3 6003 0 &  python3 COMP3221_FLClient.py client4 6004 0 & python3 COMP3221_FLClient.py client5 6005 0;    
    clear;python3 COMP3221_FLServer.py 6000 3 & python3 COMP3221_FLClient.py client1 6001 0 &  python3 COMP3221_FLClient.py client2 6002 0 & python3 COMP3221_FLClient.py client3 6003 0 &  python3 COMP3221_FLClient.py client4 6004 0 & python3 COMP3221_FLClient.py client5 6005 0;
    clear;python3 COMP3221_FLServer.py 6000 4 & python3 COMP3221_FLClient.py client1 6001 0 &  python3 COMP3221_FLClient.py client2 6002 0 & python3 COMP3221_FLClient.py client3 6003 0 &  python3 COMP3221_FLClient.py client4 6004 0 & python3 COMP3221_FLClient.py client5 6005 0;
    clear;python3 COMP3221_FLServer.py 6000 0 & python3 COMP3221_FLClient.py client1 6001 0 &  python3 COMP3221_FLClient.py client2 6002 0 & python3 COMP3221_FLClient.py client3 6003 0 &  python3 COMP3221_FLClient.py client4 6004 0 & python3 COMP3221_FLClient.py client5 6005 0;
    ```

    Mini-batch Gradient Descent (64) Column, K=1,2,3,4,0:
    [Change batch_size on line 25 of COMP3221_FLClient.py to be 64]
    ```
    clear;python3 COMP3221_FLServer.py 6000 1 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 2 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 3 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 4 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 0 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    ```

    Mini-batch Gradient Descent (64) Column, K=1,2,3,4,0:
    [Change batch_size on line 25 of COMP3221_FLClient.py to be 32, otherwise the comand is the same]
    ```
    clear;python3 COMP3221_FLServer.py 6000 1 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 2 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 3 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 4 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    clear;python3 COMP3221_FLServer.py 6000 0 & python3 COMP3221_FLClient.py client1 6001 1 &  python3 COMP3221_FLClient.py client2 6002 1 & python3 COMP3221_FLClient.py client3 6003 1 &  python3 COMP3221_FLClient.py client4 6004 1 & python3 COMP3221_FLClient.py client5 6005 1;
    ```

--------------------------------------------------------