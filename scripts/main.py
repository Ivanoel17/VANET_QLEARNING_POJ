import argparse
import train
import plot

def main():
    parser = argparse.ArgumentParser(description="Run Q-learning for VANET optimization.")
    parser.add_argument('--train', action='store_true', help="Train the Q-learning model.")
    parser.add_argument('--plot', action='store_true', help="Plot the training rewards.")
    parser.add_argument('--start-server', action='store_true', help="Start the Q-learning server.")

    args = parser.parse_args()

    if args.train:
        train.train()
    elif args.plot:
        plot.plot_rewards()
    elif args.start_server:
        import server
        server.QLearningServer().start()

if __name__ == "__main__":
    main()
