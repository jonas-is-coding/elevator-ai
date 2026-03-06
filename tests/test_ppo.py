import sys
sys.path.insert(0, '..')

from stable_baselines3 import PPO
from environment.elevator_env import ElevatorEnv

print("Lade Modell")
model = PPO.load("../training/models/ppo_elevator.zip")
print("Modell geladen")
total_rewards = []
all_pickups = []
all_dropoffs = []
all_waits = []

for run in range(1000):
    env = ElevatorEnv()
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    total_pickups = 0
    total_dropoffs = 0
    total_wait = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        total_pickups += info["pickups"]
        total_dropoffs += info["dropoffs"]
        total_wait += info["avg_wait"]

        if truncated:
            break

    avg = episode_reward / steps
    total_rewards.append(avg)
    all_pickups.append(total_pickups)
    all_dropoffs.append(total_dropoffs)
    all_waits.append(total_wait / steps)

    print(f"Run {run + 1} | Reward: {episode_reward / steps:.2f}| Pickups: {total_pickups} | Dropoffs: {total_dropoffs} | Avg Wait: {total_wait / steps:.2f}")

print(f"\n--- Average over 1000 Runs ---")
print(f"Mean Avg Reward:  {sum(total_rewards) / len(total_rewards):.2f}")
print(f"Mean Pickups:     {sum(all_pickups) / len(all_pickups):.0f}")
print(f"Mean Dropoffs:    {sum(all_dropoffs) / len(all_dropoffs):.0f}")
print(f"Mean Avg Wait:    {sum(all_waits) / len(all_waits):.2f}")
print(f"Mean Pickups:     {sum(all_pickups) / len(all_pickups):.0f}")