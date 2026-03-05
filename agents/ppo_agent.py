from stable_baselines3 import PPO

def create_model(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # Hyperparameter hier zentral
        n_steps=2048,
    )