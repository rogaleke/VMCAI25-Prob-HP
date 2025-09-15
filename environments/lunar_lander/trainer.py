import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import os

models_dir = "models/PPO_1"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make("LunarLander-v3")
env.reset()


model = PPO(
    "MlpPolicy",
    env,
    verbose=1, ent_coef=0.01, batch_size=64, gae_lambda=0.98, gamma=0.999, n_steps=1024, tensorboard_log=logdir)

TIMESTEPS = 2000000
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_1", log_interval=10)
    model.save(f"{models_dir}/{TIMESTEPS*i}_steps")


    mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20)

    print(f"Time Step {TIMESTEPS*i} -- mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    