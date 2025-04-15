# must be first import otherwise gymnasium messes up PyGame A/V framework
from gymenv import ShooterEnv

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time
import numpy as np

# Q-learning hyperparameters
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_decay = 0.995  # decay rate for epsilon
min_epsilon = 0.05  # minimum exploration rate
max_episodes = 64_000  # number of episodes to train
target_update_freq = 10  # how often to update the target network
batch_size = 64  # size of the batch for training

class DQN(nn.Module):
    '''
    Deep Q-Network (DQN) neural network architecture.
    '''

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class ReplayBuffer:
    '''
    Maintains a queue of previous action/observation values
    '''

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train_episode(env, online_net, stable_net, replay, optimizer):
    # Initialize this episode with the observation (state) being the input
    # to the neural network (a tensor).
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0

    # Play this episode.
    while not done:

        # Choose the next action (get an action, don't track gradients).
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = online_net(state.unsqueeze(0))
                action = q_vals.argmax().item()

        # Take a step forward in the game environment.
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Add this step to our replay buffer and convert to a torch object.
        replay.push((state.numpy(), action, reward, next_state, done))
        state = torch.tensor(next_state, dtype=torch.float32)

        # Don't train the network until we have enough experiences.
        if len(replay) >= batch_size:

            # Take a 'replay buffer' worth of states and run through NN.
            states, actions, rewards, next_states, dones = (
                replay.sample(batch_size)
            )
            q_values = (
                online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            )

            # Compute the current Q-Values using stable network as
            # (but don't track gradients yet)
            with torch.no_grad():
                next_q_values = stable_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            # Train online NN (q_net) using gradient descent
            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Total reward helps monitor progress
    return total_reward


def train_dqn():
    '''
    Train a Deep Q-Learning neural network to play our shooter game. The
    '''
    global epsilon
    env = ShooterEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"Creating online and stable neural networks "
          f"({n_actions}) actions / ({obs_size}) observation vars)")

    # Online (q_net) and target (target_net) networks with target being a copy
    # of q_net that is refreshed slowly to maintain learning stability.
    online_net = DQN(obs_size, n_actions)
    stable_net = DQN(obs_size, n_actions)
    stable_net.load_state_dict(online_net.state_dict())

    # Use ADAM gradient descent and a replay buffer to store state transitions.
    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    replay = ReplayBuffer()

    # Main learning loop, going through each episode one at a time.
    for episode in range(1, max_episodes):
        # Train the neural networks.
        online_net.train()
        stable_net.train()
        total_reward = train_episode(
            env, online_net, stable_net, replay, optimizer
        )

        # Update target network (the stable network) every X episodes.
        if episode % target_update_freq == 0:
            print(f"(updated stable network with latest from online network)")
            stable_net.load_state_dict(online_net.state_dict())

        # Report on the learning progress (and render agent every X episodes).
        if episode > 0 and episode % 1000 == 0:
            print(f"\n>>> Visualizing agent at episode {episode} <<\n")
            visualize_episode(online_net)
            torch.save(online_net.state_dict(), f"dqn_shooter_{episode}.pt")
        print(f"Episode {episode:>6} | "
              f"Reward: {total_reward:>10.2f} | "
              f"Epsilon: {epsilon:.3f}")

        # Epsilon greedy decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # All done, close out the environment
    env.close()
