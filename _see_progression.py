"""
This script evaluates several trained DQN models on a 2D shooter environment.
It loads each model, runs it in the game environment, and visualizes its performance.
The last model is run twice — once on level 1 and again on level 2 — to assess generalization.
Finally, it displays a PNG image that shows the reward progression during training.
"""

import sys
import time
import torch
import pygame
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gymenv import ShooterEnv         # custom environment compatible with Gym interface
from dqn_model import DQN             # DQN model architecture
from settings import FPS              # frame rate for the game environment
from dqn_render import visualize_episode  # function to run an episode and render it

# paths to different checkpoints of the trained models
model_paths = [
    "trained_models/_lvl1_ep20.pt", 
    "trained_models/_lvl1_ep100.pt", 
    "trained_models/_lvl1_ep310.pt", 
    "trained_models/_lvl1_complete.pt"
]

# select device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loop through each model checkpoint
for i, path in enumerate(model_paths):
    runs = 3 if i == len(model_paths) - 1 else 1  # run the final model twice (on level 1 and level 2)

    for run_idx in range(runs):
        level = run_idx + 1
        print(f"Running model: {path} | Level: {level}")

        # create environment for the specified level
        env = ShooterEnv(render_mode="human", start_at=level)

        # initialize model with appropriate input/output dimensions
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        model = DQN(obs_size, n_actions)

        # load trained model weights
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        # run one episode and visualize it
        reward = visualize_episode(model, device, start_level=level)
        print(f"Total reward: {reward}\n")

        # clean up the environment
        env.close()

# load and display reward progression PNG image
print("Displaying reward progression...\n")
img = mpimg.imread("trained_models/_lvl1_rewardplot.png")
plt.figure(figsize=(10, 6))  
plt.imshow(img)              
plt.axis("off")             
plt.show()
