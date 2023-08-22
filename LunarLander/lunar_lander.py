import gymnasium as gym
from stable_baselines3 import A2C, PPO
import os

MODEL = "PPO"
MODELS_DIR = "models"
MODELS_PATH = os.path.join(MODELS_DIR, MODEL)
LOG_DIR = "logs"

if (not os.path.exists(MODELS_PATH)):
    os.makedirs(MODELS_PATH)

if (not os.path.exists(LOG_DIR)):
    os.makedirs(LOG_DIR)

TIMESTEPS = 10000

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

# Mlp = Multi-layer perceptron
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

# Save the model every TIMESTEPS steps
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{MODELS_PATH}/{TIMESTEPS * i}")

# episodes = 10
# for ep in range(episodes):
#     obs, info = env.reset()
#     terminated = False
#     truncated = False

#     while (not terminated and not truncated):
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)

env.close()
