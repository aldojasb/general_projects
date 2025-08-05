"""
Training script for PPO/DDPG agents.

This module provides:
- Main training loop
- Agent training orchestration
- Experiment management
"""

# TODO: Implement training script 

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- Step 1: Define Environment & Actor-Critic Networks --- #

# Assume you have a Gym-like environment ready for CSTR
env = gym.make("CSTR-v0")

state_dim = env.observation_space.shape[0]  # [Ca, Cb, T]
action_dim = env.action_space.shape[0]      # cooling jacket temp adjustment

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # log_std bounded internally
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        actor_features = self.actor(state)
        action_mean = self.mean_head(actor_features)
        action_std = self.log_std.clamp(-20, 2).exp()  # avoiding too large variances

        state_value = self.critic(state)
        return action_mean, action_std, state_value

# Initialize actor-critic model and optimizers
model = ActorCriticNet(state_dim, action_dim)
actor_optimizer = optim.Adam([{'params': model.actor.parameters()}, {'params': [model.mean_head.weight, model.mean_head.bias, model.log_std]}], lr=3e-4)
critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)


# --- Step 2: Experience Collection (Rollouts) --- #

def collect_trajectories(model, env, steps=2048):
    state = env.reset()
    states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    for _ in range(steps):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std, value = model(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        next_state, reward, done, _ = env.step(action.numpy())

        # Collect experience
        states.append(state)
        actions.append(action.numpy())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())
        log_probs.append(log_prob.item())

        state = next_state
        if done:
            state = env.reset()

    # Convert to tensors
    return (states, actions, rewards, dones, values, log_probs)


# --- Step 3: Compute Advantages (GAE) --- #

def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    values = np.array(values + [0])  # bootstrap next value

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values[:-1]
    # Normalization (critical!)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# --- Step 4 & 5: PPO Update (Clipped Objective & Value Loss) --- #

def ppo_update(model, states, actions, log_probs_old, returns, advantages, clip=0.2, epochs=10):
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    log_probs_old = torch.FloatTensor(log_probs_old)

    for _ in range(epochs):
        mean, std, values = model(states)
        dist = torch.distributions.Normal(mean, std)

        log_probs_new = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        # Clipped surrogate objective
        ratios = torch.exp(log_probs_new - log_probs_old)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Critic loss (MSE)
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss with entropy bonus
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Avoid exploding gradients
        actor_optimizer.step()

        # Update critic separately for stability
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


# --- Main training loop --- #
num_updates = 1000
for update in range(num_updates):
    # Collect rollouts
    states, actions, rewards, dones, values, log_probs_old = collect_trajectories(model, env)

    # Compute GAE advantages
    advantages, returns = compute_gae(rewards, dones, values)

    # Update PPO policy and value network
    ppo_update(model, states, actions, log_probs_old, returns, advantages)

    if (update + 1) % 100 == 0:
        print(f"Update {update + 1}/{num_updates} completed.")

# Save trained Actor-Critic model after training
torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth")  # clearly storing both actor and critic parameters
