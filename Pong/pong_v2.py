from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C
import os

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0, )
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)

model = A2C("CnnPolicy", vec_env, verbose=1)

obs = vec_env.reset()


MODEL = "Pong-PPO"
MODELS_DIR = "models"
MODELS_PATH = os.path.join(MODELS_DIR, MODEL)
LOG_DIR = "logs"

if (not os.path.exists(MODELS_PATH)):
    os.makedirs(MODELS_PATH)

if (not os.path.exists(LOG_DIR)):
    os.makedirs(LOG_DIR)

TIMESTEPS = 10000

for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{MODELS_PATH}/{TIMESTEPS * i}")


for i in range(1, 50):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
