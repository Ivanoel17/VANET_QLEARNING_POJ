
# VANET Q-Learning Optimization

This project contains scripts for training a Q-learning agent to optimize beacon rate and power transmission in a VANET (Vehicular Ad-hoc Network) environment. The setup includes:

1. **train.py** - The agent training script.
2. **server.py** - A server that uses the trained model for inference.
3. **plot.py** - Visualizes the training rewards over episodes.
4. **main.py** - A command-line interface to run training, plotting, or starting the server.

## Requirements

- Python 3.6 or higher
- Required libraries: `numpy`, `matplotlib`, `csv`

## Usage

1. **Train the model:**
   ```bash
   python main.py --train
   ```

2. **Plot training rewards:**
   ```bash
   python main.py --plot
   ```

3. **Start the server:**
   ```bash
   python main.py --start-server
   ```

## Files

- **q_table.npy** - The trained Q-table model.
- **training_rewards.csv** - Log file containing reward data for each training episode.
