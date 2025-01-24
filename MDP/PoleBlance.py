import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------
# 1. Hyperparameters and Configuration
# ----------------------------------------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99         # Discount factor
LR = 1e-3            # Learning rate
BATCH_SIZE = 64
MEMORY_SIZE = 10_000
EPS_START = 1.0      # Initial epsilon for ε-greedy
EPS_END = 0.01       # Final epsilon
EPS_DECAY = 0.995    # Multiply epsilon by this each episode
TARGET_UPDATE = 10   # How often to update target network
NUM_EPISODES = 300   # Number of training episodes

# Toggle to watch the agent after training (requires a display)
WATCH_AFTER_TRAIN = True
NUM_WATCH_EPISODES = 3
RENDER_DELAY = 0.02


# ----------------------------------------------------
# 2. Q-Network Definition
# ----------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# 3. Experience Replay Buffer
# ----------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------
# 4. Training Function (one mini-batch update)
# ----------------------------------------------------
def train_step(q_net, target_net, optimizer, replay_buffer):
    """Perform one batch update of the Q-network using data from replay buffer."""
    if len(replay_buffer) < BATCH_SIZE:
        return  # Not enough data yet

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Convert to PyTorch tensors
    states_tensor = torch.tensor(states)
    actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
    rewards_tensor = torch.tensor(rewards)
    next_states_tensor = torch.tensor(next_states)
    dones_tensor = torch.tensor(dones)

    # Current Q-values (for the actions taken)
    q_values = q_net(states_tensor).gather(1, actions_tensor)

    # Next Q-values (from target network)
    with torch.no_grad():
        # For each next_state, get max Q-value among actions
        max_next_q_values = target_net(next_states_tensor).max(dim=1)[0]
        # If done, there is no future reward
        target_q_values = rewards_tensor + GAMMA * max_next_q_values * (1 - dones_tensor)

    # Compute loss (MSE)
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ----------------------------------------------------
# 5. Utility: Watch Trained Agent
# ----------------------------------------------------
def watch_agent(q_net, env_name=ENV_NAME, num_episodes=3, render_delay=0.02):
    """Render and watch the trained agent for a few episodes."""
    env = gym.make(env_name, render_mode="human")  # Gym>=0.26
    for episode in range(num_episodes):
        state, _ = env.reset(seed=42+episode)  # or simply env.reset()
        done = False
        total_reward = 0

        while not done:
            time.sleep(render_delay)  # Slow down so we can see
            state_t = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                q_vals = q_net(state_t)
                action = q_vals.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        print(f"[Watch Mode] Episode {episode+1} finished with reward = {total_reward}")
    env.close()


# ----------------------------------------------------
# 6. Utility: Plotting Rewards
# ----------------------------------------------------
def plot_rewards(reward_history):
    plt.figure(figsize=(8,5))
    plt.plot(reward_history, label='Episode Reward')
    plt.title('Training Performance Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()


# ----------------------------------------------------
# 7. Main: Putting It All Together
# ----------------------------------------------------
def main():
    # Create the CartPole environment
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]  # Should be 4 for CartPole
    n_actions = env.action_space.n           # Should be 2 (left/right)

    # Initialize networks
    q_net = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())  # Start with same weights

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    reward_history = []

    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=episode)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor([state], dtype=torch.float32)
                    q_vals = q_net(state_t)
                    action = q_vals.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Train
            train_step(q_net, target_net, optimizer, replay_buffer)

            state = next_state
            total_reward += reward

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Periodically update the target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        reward_history.append(total_reward)
        print(f"Episode {episode+1}/{NUM_EPISODES} | Total Reward = {total_reward} | Epsilon = {epsilon:.3f}")

    env.close()

    # Plot the training reward curve
    plot_rewards(reward_history)

    # Optionally watch the agent
    if WATCH_AFTER_TRAIN:
        watch_agent(q_net, ENV_NAME, num_episodes=NUM_WATCH_EPISODES, render_delay=RENDER_DELAY)


# ----------------------------------------------------
# 8. Run from the command line or IDE
# ----------------------------------------------------
if __name__ == "__main__":
    main()
