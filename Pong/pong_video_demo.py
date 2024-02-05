import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import cv2

# Function to load the environment
def load_env():
    vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env

# Function to load the model
def load_model(path, env):
    model = PPO.load(path, env)
    return model

# Directory where models are saved
MODEL_DIR = "models/Pong-PPO"

# Frames per model and video settings
NUM_FRAMES = 400
SKIP = 4
FPS = 40
FRAME_FILE = "./Pong/all_frames.npz"

def collect_frames():
    # Initialize a NumPy array to store all frames
    # Assuming you know the frame dimensions (height, width, channels)
    # You might need to adjust these based on your environment's specifics
    height, width, channels = (210, 160, 3)
    
    # 29//SKIP epochs + 2 for final epoch because double the no. of frames
    total_frames = ((30 - 1) // SKIP + 1 + 1) * NUM_FRAMES
    all_frames = np.empty((total_frames, height, width, channels), dtype=np.uint8)

    print("all_frames.shape: ", all_frames.shape)
    print("all_frames[0].shape: ", all_frames[0].shape)

    frame_idx = 0
    frames = NUM_FRAMES
    for epoch in range(1, 30, SKIP):  # Stop before the last epoch
        model_path = f"{MODEL_DIR}/{epoch * 10000}.zip"
        env = load_env()
        model = load_model(model_path, env)

        obs = env.reset()
        if (epoch == 29):
            frames = NUM_FRAMES * 2

        for _ in range(frames):
            action, _states = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)

            frame = env.render(mode='rgb_array')
            all_frames[frame_idx] = frame
            frame_idx += 1

            if dones.any():
                obs = env.reset()

        env.close()

    env.close()

    # Save the NumPy array of frames
    np.savez_compressed(FRAME_FILE, all_frames)

def load_frames():
    with np.load(FRAME_FILE) as data:
        return data['arr_0']

# Main
try:
    # Attempt to load frames if they already exist
    all_frames = load_frames()
    print("Loaded saved frames.")
except IOError:
    # Collect frames if not already saved
    print("Collecting frames...")
    collect_frames()
    all_frames = load_frames()

# Verify frame integrity
for i, frame in enumerate(all_frames):
    if frame is None:
        print(f"Frame at index {i} is None")
    elif frame.shape != all_frames[0].shape:
        print(f"Frame at index {i} has inconsistent dimensions")

print("Shape of the first frame:", all_frames[0].shape)

# Create a video from the frames
height, width, layers = all_frames[0].shape
size = (width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./etc/pong_progress.mp4', fourcc, FPS, size)

# Calculate number of frames per epoch assuming it's uniform
frames_per_epoch = NUM_FRAMES

# Initialize a variable for the last epoch for clarity
last_epoch = 29

# Add text to each frame and write to the video
for idx, frame in enumerate(all_frames):
    # Determine the epoch number based on the frame index
    if idx < frames_per_epoch * (last_epoch // SKIP):
        epoch_number = (idx // frames_per_epoch) * SKIP + 1
    else:
        # For the last epoch (29)
        epoch_number = last_epoch
    
    text = f'Epoch: {epoch_number}'
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Maintain the same scale as before
    font_color = (255, 255, 255)  # White color
    outline_color = (0, 0, 0)  # Black outline
    line_type = cv2.LINE_AA
    thickness = 2
    
    # Calculate the text position so the text is centered
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 10  # Adjust if necessary
    
    position = (text_x, text_y)

    # Add outline to text for better clarity
    cv2.putText(frame, text, position, font, font_scale, outline_color, thickness*3, line_type)  # Thicker outline
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)  # Regular text on top

    # Convert RGB to BGR for OpenCV and write the frame
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print("Video has been saved.")
