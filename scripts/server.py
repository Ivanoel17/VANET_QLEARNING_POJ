import socket
import json
import numpy as np
import os

# Constants
HOST = '127.0.0.1'
PORT = 5000

class QLearningServer:
    def __init__(self):
        # Check if the q_table.npy file exists when starting the server
        self.q_table_path = 'q_table.npy'
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        else:
            raise FileNotFoundError(f"Model file {self.q_table_path} not found. Please train the model first.")
        
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

    def handle_client(self, conn):
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            try:
                state = json.loads(data.decode())
                print(f"Received: {state}")
                
                # Current parameters
                current_power = state['power']
                current_beacon = state['beacon']
                current_cbr = state['cbr']
                
                # Select action using the trained model
                action = self.select_action((current_power, current_beacon, current_cbr))
                
                # Determine new values
                new_power = max(5, min(30, current_power + (-1 if action == 0 else 1)))
                new_beacon = max(1, min(20, current_beacon + (-1 if action == 0 else 1)))
                
                # Send response
                response = {
                    'power': new_power,
                    'beacon': new_beacon,
                }
                conn.send(json.dumps(response).encode())
                print(f"Sent: {response}")
                
            except Exception as e:
                print(f"Error: {e}")
                break

    def select_action(self, state):
        # Simplified state discretization
        POWER_BINS = [5, 15, 25, 30]
        BEACON_BINS = [1, 5, 10, 20]
        CBR_BINS = [0.0, 0.3, 0.6, 1.0]
        EPSILON = 0.1
        
        def discretize(value, bins):
            return np.digitize(value, bins) - 1
        
        power_idx = discretize(state[0], POWER_BINS)
        beacon_idx = discretize(state[1], BEACON_BINS)
        cbr_idx = discretize(state[2], CBR_BINS)
        
        if np.random.random() < EPSILON:
            return np.random.choice([0, 1])  # 0: decrease, 1: increase
        return np.argmax(self.q_table[power_idx, beacon_idx, cbr_idx])

    def start(self):
        while True:
            conn, addr = self.server.accept()
            print(f"Connected: {addr}")
            self.handle_client(conn)
            conn.close()
