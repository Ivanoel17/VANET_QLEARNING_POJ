import numpy as np
import random
import csv

# Constants
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]
CBR_TARGET = 0.65
LEARNING_RATE = 0.01  # Reduced learning rate for smoother updates
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0  # Start with full exploration
EPSILON_END = 0.05  # Reduced minimum exploration
EPSILON_DECAY = 0.995  # Smooth decay factor for exploration
EPSILON = EPSILON_START  # Set initial epsilon to start high
MAX_STEPS = 3000  # Total steps (no episodes now)

# Initialize Q-table
q_table = np.zeros((len(POWER_BINS), len(BEACON_BINS), len(CBR_BINS), 2))

# Discretize the state values
def discretize(value, bins):
    return np.digitize(value, bins) - 1

def calculate_reward(cbr):
    # Calculate the absolute difference between CBR and target CBR
    diff = abs(cbr - CBR_TARGET)
    
    # Positive reward for small deviation (less than 0.2)
    if diff <= 0.5:
        print("CBR",cbr)
        reward = (1 - diff)  
    else:
        # Negative penalty for large deviation (greater than 0.2)
        reward = - (diff - 0.2)  

    return reward


# Select action based on epsilon-greedy policy
def select_action(state):
    power_idx = discretize(state[0], POWER_BINS)
    beacon_idx = discretize(state[1], BEACON_BINS)
    cbr_idx = discretize(state[2], CBR_BINS)
    
    if random.random() < EPSILON:
        return random.choice([0, 1])  # 0: decrease, 1: increase
    return np.argmax(q_table[power_idx, beacon_idx, cbr_idx])

# Update the Q-table
def update_q_table(state, action, reward, new_state):
    old_idx = discretize(state[0], POWER_BINS), discretize(state[1], BEACON_BINS), discretize(state[2], CBR_BINS)
    new_idx = discretize(new_state[0], POWER_BINS), discretize(new_state[1], BEACON_BINS), discretize(new_state[2], CBR_BINS)
    
    old_q = q_table[old_idx + (action,)]
    max_new_q = np.max(q_table[new_idx])
    q_table[old_idx + (action,)] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_new_q - old_q)

# Moving average function for smoothing rewards
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Training function (step-based)
def train():
    timesteps = []
    cumulative_rewards = []
    cumulative_reward = 0  # Start with 0 cumulative reward
    timestep_counter = 0  # Initialize timestep counter
    
    state = (random.choice(POWER_BINS), random.choice(BEACON_BINS), random.choice(CBR_BINS))  # Initial state
    
    # Open the CSV file for logging timestep and cumulative reward
    with open('training_rewards.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestep', 'Cumulative Reward'])  # Write header
        
        # Loop for total steps
        while timestep_counter < MAX_STEPS:
            action = select_action(state)
            
            # Simulate new state based on action
            new_power = max(5, min(30, state[0] + (-1 if action == 0 else 1)))
            new_beacon = max(1, min(20, state[1] + (-1 if action == 0 else 1)))
            new_cbr = state[2]  # CBR doesn't change in this simplified model
            
            reward = calculate_reward(state[2])
            cumulative_reward += reward  # Add the reward to cumulative reward
            
            # Update Q-table
            update_q_table(state, action, reward, (new_power, new_beacon, new_cbr))
            
            state = (new_power, new_beacon, new_cbr)  # Update state
            
            # Log timestep and cumulative reward immediately
            writer.writerow([timestep_counter, cumulative_reward])
            
            timestep_counter += 1  # Increment timestep
            
            # Decay epsilon to reduce exploration over time
            global EPSILON
            EPSILON = max(EPSILON_END, EPSILON * EPSILON_DECAY)
    
    # Save Q-table to file
    np.save('q_table.npy', q_table)

# Start training
train()
