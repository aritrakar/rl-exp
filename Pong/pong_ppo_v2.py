from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# Environment setup
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# PPO model with a CNN policy
model = PPO("CnnPolicy", vec_env, verbose=1)

# Directories for saving models and logs
MODEL = "Pong-PPO"
MODELS_DIR = "models"
MODELS_PATH = os.path.join(MODELS_DIR, MODEL)
LOG_DIR = "logs"

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Training
TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{MODELS_PATH}/{TIMESTEPS * i}")

# Evaluation
for _ in range(50):  # Number of episodes
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        done = any(dones)
        vec_env.render("human")
