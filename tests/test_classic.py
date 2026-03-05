import sys
sys.path.insert(0, '..')

from environment.elevator_env import ElevatorEnv
from agents.classic_agent import scan_action

env = ElevatorEnv()
obs, info = env.reset()

total_reward = 0
steps = 0

while True:
    action = scan_action(env.building)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if steps % 100 == 0:
        waiting = sum(1 for f in env.building.floors if f.waitingUp or f.waitingDown)
        positions = [e.currentFloor for e in env.building.elevators]
        print(f"Step {steps:4d} | Time: {env.time_of_day:02d}:00 | "
              f"Waiting Floors: {waiting:2d} | "
              f"Elevators: {positions} | "
              f"Avg Reward: {total_reward / steps:.2f}")

    if terminated or truncated:
        break

print(f"\n--- Result ---")
print(f"Steps: {steps}")
print(f"Average Reward: {total_reward / steps:.2f}")