from stable_baselines3 import PPO

def create_model(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.05,
        learning_rate=0.0001,
    )