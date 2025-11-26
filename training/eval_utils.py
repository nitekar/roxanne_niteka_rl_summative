import numpy as np

def evaluate_policy(env_fn, model, episodes: int = 5):
    """
    Evaluate a RL model over a number of episodes.
    Returns dictionary with avg_reward, std_reward, avg_length, grazing_balance_mean.
    """
    rewards = []
    lengths = []
    grazing_balance = []

    for ep in range(episodes):
        obs, _ = env_fn().reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_fn().step(action)
            ep_reward += reward
            steps += 1
            if 'grazing_balance' in info:
                grazing_balance.append(info['grazing_balance'])

        rewards.append(ep_reward)
        lengths.append(steps)

    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_length': np.mean(lengths),
        'grazing_balance_mean': np.mean(grazing_balance) if grazing_balance else 0.0
    }
