import argparse
import csv
import os
from itertools import product
from stable_baselines3 import A2C
from environment.agrotrack_env import AgroTrackEnv
from training.eval_utils import evaluate_policy

def make_a2c(env_fn, params):
    return A2C(
        "MlpPolicy",
        env=env_fn(),
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        n_steps=params['n_steps'],
        ent_coef=params['ent_coef'],
        verbose=0
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=str, default='models/a2c')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output,'a2c_results.csv')
    keys = ['learning_rate','gamma','n_steps','ent_coef']
    with open(results_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward','std_reward','avg_length','grazing_balance_mean'])

    grid = {
        'learning_rate':[1e-3,5e-4],
        'gamma':[0.95,0.99],
        'n_steps':[5,20,50],
        'ent_coef':[0.0,0.01,0.05]
    }

    combos = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    if len(combos) < args.runs:
        while len(combos) < args.runs:
            combos.append(combos[-1].copy())

    best_reward = float('-inf')
    best_model_path = None
    run_id = 0

    for params in combos:
        run_id += 1
        print(f"[A2C] Run {run_id}/{len(combos)} params={params}")
        env_fn = lambda: AgroTrackEnv()
        model = make_a2c(env_fn, params)
        model.learn(total_timesteps=args.episodes)
        metrics = evaluate_policy(env_fn, model, episodes=args.eval_episodes)
        print(f"Eval metrics: {metrics}")

        with open(results_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id]+[params[k] for k in keys]+
                            [metrics['avg_reward'],metrics['std_reward'],metrics['avg_length'],metrics['grazing_balance_mean']])
        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output,'best_a2c.zip')
            model.save(best_model_path)
        model.env.close()

    print(f"Best A2C model: {best_model_path} avg_reward={best_reward:.2f}")

if __name__ == "__main__":
    main()
