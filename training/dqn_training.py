import argparse
import csv
import os
from itertools import product
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

from environment.agrotrack_env import AgroTrackEnv
from training.eval_utils import evaluate_policy

def make_dqn(env_fn, params):
    return DQN(
        policy=MlpPolicy,
        env=env_fn(),
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        batch_size=params['batch_size'],
        buffer_size=params['buffer_size'],
        train_freq=params['train_freq'],
        target_update_interval=params['target_update_interval'],
        exploration_fraction=params['exploration_fraction'],
        exploration_initial_eps=params['exploration_initial_eps'],
        exploration_final_eps=params['exploration_final_eps'],
        verbose=0,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=str, default='models/dqn')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, 'dqn_results.csv')
    keys = ['learning_rate','gamma','batch_size','buffer_size','train_freq','target_update_interval',
            'exploration_fraction','exploration_initial_eps','exploration_final_eps']
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward','std_reward','avg_length','grazing_balance_mean'])

    # Hyperparameter grid
    grid = {
        'learning_rate': [1e-3,5e-4],
        'gamma':[0.95,0.99],
        'batch_size':[32,64],
        'buffer_size':[5000,20000],
        'train_freq':[4],
        'target_update_interval':[1000,2000],
        'exploration_fraction':[0.2,0.4],
        'exploration_initial_eps':[1.0],
        'exploration_final_eps':[0.05]
    }

    combos = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    if len(combos) < args.runs:
        while len(combos) < args.runs:
            mod = combos[-1].copy()
            mod['gamma'] = 0.9 if mod['gamma'] != 0.9 else 0.99
            combos.append(mod)

    best_reward = float('-inf')
    best_model_path = None
    run_id = 0

    for params in combos:
        run_id += 1
        print(f"[DQN] Run {run_id}/{len(combos)} params={params}")
        env_fn = lambda: AgroTrackEnv()
        model = make_dqn(env_fn, params)
        model.learn(total_timesteps=args.episodes)
        metrics = evaluate_policy(env_fn, model, episodes=args.eval_episodes)
        print(f"Eval metrics: {metrics}")

        with open(results_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id] + [params[k] for k in keys] +
                            [metrics['avg_reward'], metrics['std_reward'], metrics['avg_length'], metrics['grazing_balance_mean']])
        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output,'best_dqn.zip')
            model.save(best_model_path)
        model.env.close()

    print(f"Best DQN model: {best_model_path} avg_reward={best_reward:.2f}")

if __name__ == "__main__":
    main()
