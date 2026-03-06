from environment.elevator_env import ElevatorEnv
from agents.ppo_agent import create_model

env = ElevatorEnv()
model = create_model(env)
model.learn(total_timesteps=5_000_000)
model.save("models/ppo_elevator")