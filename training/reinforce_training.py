import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from environment.agrotrack_env import AgroTrackEnv
from training.eval_utils import evaluate_policy

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

def reinforce_train(env_fn, params, total_episodes):
    env = env_fn()
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PolicyNetwork(obs_size, action_size, params['hidden_size'])
    optimizer = optim.Adam(policy.parameters(), lr=params['learning_rate'])
    gamma = params['gamma']

    for ep in range(total_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        truncated = False

        while not done and not truncated:
            state = torch.FloatTensor(obs)
            probs = policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            obs, reward, done, truncated, info = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=str, default='models/reinforce')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output,'reinforce_results.csv')
    keys = ['learning_rate','gamma','hidden_size']
    with open(results_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward','std_reward','avg_length','grazing_balance_mean'])

    grid = {
        'learning_rate':[1e-3,5e-4],
        'gamma':[0.95,0.99],
        'hidden_size':[64,128]
    }
    from itertools import product
    combos = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
    if len(combos) < args.runs:
        while len(combos) < args.runs:
            combos.append(combos[-1].copy())

    best_reward = float('-inf')
    best_model_path = None
    run_id = 0

    for params in combos:
        run_id += 1
        print(f"[REINFORCE] Run {run_id}/{len(combos)} params={params}")
        env_fn = lambda: AgroTrackEnv()
        policy = reinforce_train(env_fn, params, total_episodes=args.episodes)

        # Evaluation
        class PolicyWrapper:
            def predict(self, obs, deterministic=True):
                obs_t = torch.FloatTensor(obs)
                probs = policy(obs_t)
                action = torch.argmax(probs).item() if deterministic else Categorical(probs).sample().item()
                return action, None

        metrics = evaluate_policy(env_fn, PolicyWrapper(), episodes=args.eval_episodes)
        print(f"Eval metrics: {metrics}")

        with open(results_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id]+[params[k] for k in keys]+
                            [metrics['avg_reward'],metrics['std_reward'],metrics['avg_length'],metrics['grazing_balance_mean']])

        # Save model
        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output,'best_reinforce.pth')
            torch.save(policy.state_dict(), best_model_path)

    print(f"Best REINFORCE model: {best_model_path} avg_reward={best_reward:.2f}")

if __name__ == "__main__":
    main()
