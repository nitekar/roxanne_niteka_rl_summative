import argparse
import os
import sys

def parse_args():
 parser = argparse.ArgumentParser(description="Run RL algorithms for AgroTrackEnv")

parser.add_argument(
    "--algorithm",
    type=str,
    required=True,
    choices=["dqn", "ppo", "a2c", "reinforce"],
    help="Which RL algorithm to run"
)

parser.add_argument("--episodes", type=int, default=50000)
parser.add_argument("--eval_episodes", type=int, default=5)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--output", type=str, default="models")

if __name__ == "__main__":
 args = parse_args()

print(f"Running {args.algorithm} for {args.episodes} timesteps...")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if args.algorithm == "dqn":
    from training.dqn_training import main as dqn_main
    dqn_main()

elif args.algorithm == "ppo":
    from training.ppo_training import main as ppo_main
    ppo_main()

elif args.algorithm == "a2c":
    from training.a2c_training import main as a2c_main
    a2c_main()

elif args.algorithm == "reinforce":
    from training.reinforce_training import main as reinforce_main
    reinforce_main()

return parser.parse_args()