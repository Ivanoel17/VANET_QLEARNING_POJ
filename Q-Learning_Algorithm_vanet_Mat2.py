import socket
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import csv

# Q-learning Parameters
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 1000  # Total number of episodes

# Placeholder for Q-values
Q = {}

# Define the goal state
goal_state = (4, 4)  # Target position in the grid (4, 4)

# Transmission parameters
transmission_power = 1.0
beacon_rate = 0.5
cbr_target = 0.65

# Initialize Q-table for all states and actions
try:
    # Try to load Q-table from file
    Q = np.load('q_table.npy')  # Load Q-table from .npy file
    print("Q-table loaded from q_table.npy")
except FileNotFoundError:
    # If file doesn't exist, initialize a new Q-table
    Q = np.random.rand(5, 5, 4)  # Initialize a new Q-table (5x5 grid with 4 actions)
    print("Q-table not found, initializing new Q-table.")

# Function to get possible actions (4 possible movements)
def get_possible_actions(state):
    return [0, 1, 2, 3]  # Move Up, Down, Left, Right

# Function to calculate reward based on state and action
def get_reward(state, action, transmission_power, beacon_rate, cbr):
    reward = -1  # Default penalty for each step
    if state == goal_state:
        return 100  # Reward for reaching the goal
    
    if transmission_power > 0.5: reward += 2
    if beacon_rate > 0.4: reward += 3
    
    # CBR-based reward
    cbr_diff = abs(cbr - cbr_target)
    if cbr_diff < 0.1: reward += 5
    elif cbr_diff < 0.2: reward -= 2
    else: reward -= 5

    return reward

# Function to get next state based on action
def get_next_state(state, action):
    x, y = state[:2]  # Only take the first two elements (x, y)

    if action == 0:  # Move Up
        x = max(0, x - 1)
    elif action == 1:  # Move Down
        x = min(4, x + 1)
    elif action == 2:  # Move Left
        y = max(0, y - 1)
    elif action == 3:  # Move Right
        y = min(4, y + 1)

    return (x, y)

# Function to select action using epsilon-greedy strategy
def select_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_possible_actions(state))  # Random action
    else:
        # Select the best action based on current Q-table values
        return max(get_possible_actions(state), key=lambda a: Q[state[0], state[1], a])

# Create a socket server to receive data from MATLAB
def start_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5000))  # Listen on localhost:5000
    server_socket.listen(1)
    print("Server listening on port 5000...")

    rewards_per_timestep = []  # Track reward per timestep
    rewards_per_episode = []  # Track reward per episode

    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")

        # Timeout for socket (10 seconds)
        try:
            server_socket.settimeout(10)  # Timeout after 10 seconds if no data is received
            client_socket, client_address = server_socket.accept()

            print(f"Connection established with {client_address}")

            # Receive data from MATLAB
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                print(f"Received data: {data}")
                batch_data = json.loads(data)

                # Process data (same logic as in Flask)
                responses = {}

                for veh_id, vehicle_data in batch_data.items():
                    # Extract vehicle parameters
                    current_power = vehicle_data.get("transmissionPower", 0.5)
                    current_beacon = vehicle_data.get("beaconRate", 0.5)
                    cbr = vehicle_data.get("CBR", 0.65)
                    neighbors = vehicle_data.get("neighbors", 0)
                    current_snr = vehicle_data.get("SNR", 10)

                    # Compose the state for Q-learning (only position x and y are used for Q-table)
                    state = [neighbors, current_snr]  # This can be adjusted depending on how you want to handle state
                    
                    # Ensure that state[0] and state[1] are within the valid range of (0, 4)
                    state[0] = min(max(state[0], 0), 4)
                    state[1] = min(max(state[1], 0), 4)

                    # Select action using epsilon-greedy
                    action_idx = select_action(state, epsilon=epsilon_start)

                    # Simulate optimization: adjust transmission power and beacon rate
                    new_power = current_power + random.uniform(-0.3, 0.3)  # Increased variation
                    new_beacon = current_beacon + random.uniform(-0.3, 0.3)  # Increased variation
                    adjusted_mcs = 15  # MCS adjustment based on SNR (simplified)

                    # Save the results in the responses dictionary
                    responses[veh_id] = {
                        "transmissionPower": float(new_power),
                        "beaconRate": float(new_beacon),
                        "MCS": adjusted_mcs
                    }

                    # Calculate reward and update Q-values
                    reward = get_reward(state, action_idx, new_power, new_beacon, cbr)
                    rewards_per_timestep.append(reward)
                    total_reward = sum(rewards_per_timestep)

                    # Update Q-value
                    old_q_value = Q[state[0], state[1], action_idx]  # Direct access for numpy array
                    future_q_value = max([Q[get_next_state(state[:2], action_idx)[0], get_next_state(state[:2], action_idx)[1], a] for a in get_possible_actions(state)])

                # Print the response in the terminal (for debugging)
                print("[DEBUG] Sending RL response to MATLAB:")
                print(responses)

                # Send the response back to MATLAB
                client_socket.sendall(json.dumps(responses).encode('utf-8'))

                # Close the connection
                client_socket.close()

        except socket.timeout:
            print("No data received for 10 seconds, continuing...")

        # Only plot after completing all episodes
        if episode + 1 == n_episodes:
            # Plotting reward per timestep
            plt.plot(rewards_per_timestep)
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.title('Reward per Timestep')
            plt.show()

            # Plotting total reward per episode
            plt.plot(rewards_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward per Episode')
            plt.show()

if __name__ == '__main__':
    start_socket_server()
