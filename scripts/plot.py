import csv
import matplotlib.pyplot as plt

def plot_rewards():
    timesteps = []
    cumulative_rewards = []

    # Read the training rewards CSV
    with open('training_rewards.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            timesteps.append(int(row[0]))  # Timestep
            cumulative_rewards.append(float(row[1]))  # Cumulative reward
    
    # Plot Timestep vs Cumulative Reward
    plt.plot(timesteps, cumulative_rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress (Timestep vs Cumulative Reward)')
    plt.show()
