import numpy as np
import random
import matplotlib.pyplot as plt
import csv

# Parameter Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Starting value for epsilon
epsilon_min = 0.01  # Minimum value for epsilon
epsilon_decay = 0.995  # Decay rate for epsilon
n_episodes = 100  # Total number of episodes

# Placeholder for Q-values
Q = {}

# Define the goal state (misalnya di grid 5x5, goal_state adalah titik (4, 4))
goal_state = (4, 4)  # Menentukan goal_state sebagai posisi (4, 4) dalam grid

# Transmission parameters
transmission_power = 1.0  # Initial transmission power (can be adjusted)
beacon_rate = 0.5  # Beacon rate, from 0 to 1
cbr_target = 0.65  # Target CBR (Channel Busy Ratio)

def get_possible_actions(state):
    return [0, 1, 2, 3]  # Example: four possible actions (e.g., move up, down, left, right)

def get_reward(state, action, transmission_power, beacon_rate, cbr):
    # Reward structure based on VANET or specific goal condition
    # Example: reward depends on network quality, distance, etc.
    
    # Misalnya, memberikan reward lebih besar jika mencapai goal_state
    if state == goal_state:
        return 100  # Positive reward for reaching the goal
    
    # Custom reward logic based on parameters
    reward = -1  # Default reward for each step

    # Adjust reward based on transmission power, beacon rate, and CBR
    if transmission_power > 0.5:  # High transmission power can help
        reward += 2
    
    if beacon_rate > 0.4:  # High beacon rate improves network information flow
        reward += 3

    # CBR-based reward
    cbr_diff = abs(cbr - cbr_target)
    if cbr_diff < 0.1:  # Reward if CBR is close to target (within ±0.1)
        reward += 5
    elif cbr_diff < 0.2:  # Slight penalty if CBR is still relatively close
        reward -= 2
    else:  # Larger penalty if CBR is far from the target
        reward -= 5

    return reward

def get_next_state(state, action):
    x, y = state
    if action == 0:  # Move Up
        x = max(0, x - 1)
    elif action == 1:  # Move Down
        x = min(4, x + 1)
    elif action == 2:  # Move Left
        y = max(0, y - 1)
    elif action == 3:  # Move Right
        y = min(4, y + 1)
    
    return (x, y)

# Initialize Q-table for all states and actions
for i in range(5):
    for j in range(5):
        for a in range(4):  # assuming 4 possible actions
            Q[(i, j, a)] = 0  # Initialize Q-values to 0

# Function to select action (epsilon-greedy)
def select_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_possible_actions(state))  # Random action
    else:
        return max(get_possible_actions(state), key=lambda a: Q.get((state[0], state[1], a), 0))  # Best action based on Q-value

# List to track rewards per timestep and per episode
rewards_per_timestep = []
rewards_per_episode = []

# CSV log file to store actions and CBR
csv_file = "training_log.csv"
csv_headers = ['Episode', 'Timestep', 'Action', 'State', 'Transmission Power', 'Beacon Rate', 'CBR', 'Reward']

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

    # Q-learning Algorithm
    for episode in range(n_episodes):
        state = (0, 0)  # Reset the environment to the initial state
        total_reward = 0
        done = False
        timestep_rewards = []  # Track rewards at each timestep

        while not done:
            action = select_action(state, epsilon_start)  # Select an action using epsilon-greedy
            next_state = get_next_state(state, action)
            
            # Adjust transmission power, beacon rate, and CBR at each timestep (could be dynamic)
            # Example: these could be based on environmental factors or agent performance
            transmission_power = random.uniform(0.5, 1.0)  # Simulated dynamic change
            beacon_rate = random.uniform(0.3, 0.7)  # Simulated dynamic change
            cbr = random.uniform(0.4, 1.0)  # Simulated dynamic change, target 0.65

            reward = get_reward(next_state, action, transmission_power, beacon_rate, cbr)  # Get reward for the next state

            # Update Q-value
            old_q_value = Q.get((state[0], state[1], action), 0)
            future_q_value = max([Q.get((next_state[0], next_state[1], a), 0) for a in get_possible_actions(next_state)])
            Q[(state[0], state[1], action)] = old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)

            # Log to CSV
            writer.writerow([episode, len(timestep_rewards), action, state, transmission_power, beacon_rate, cbr, reward])

            # Update total reward and state
            total_reward += reward
            state = next_state
            timestep_rewards.append(reward)

            if state == goal_state:  # Goal reached, terminate the episode
                done = True
        
        rewards_per_episode.append(total_reward)  # Track total reward for the episode
        rewards_per_timestep.extend(timestep_rewards)  # Track reward per timestep
        epsilon_start = max(epsilon_min, epsilon_start * epsilon_decay)  # Decay epsilon

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
