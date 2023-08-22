import gymnasium as gym
from stable_baselines3 import PPO, DQN
import os

MODEL = "Pong-DQN"
MODELS_DIR = "models"
MODELS_PATH = os.path.join(MODELS_DIR, MODEL)
LOG_DIR = "logs"

if (not os.path.exists(MODELS_PATH)):
    os.makedirs(MODELS_PATH)

if (not os.path.exists(LOG_DIR)):
    os.makedirs(LOG_DIR)

TIMESTEPS = 10000

env = gym.make("ALE/Pong-v5", render_mode="human")

env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

# Save the model every TIMESTEPS steps
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{MODELS_PATH}/{TIMESTEPS * i}")
