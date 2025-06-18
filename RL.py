import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import os
import argparse
import time

class PhysicsEnvWrapper:
    def __init__(self, frame_size=(64, 64)):
        self.frame_size = frame_size
        self.reset()
        
    def reset(self):
        self.rigid_bodies = []
        self.fluid_particles = []
        self.frames = []
        return self._get_obs()
    
    def _get_obs(self):
        # Return flattened observation
        return torch.rand(3 * 64 * 64) * 2 - 1  # Range [-1, 1]
    
    def step(self, action):
        # Reshape action to frame dimensions
        action_frame = action.view(3, 64, 64)
        clipped_action = torch.clamp(action_frame, -1, 1)
        new_frame = torch.tanh(clipped_action)
        
        # Compute simple reward
        reward = torch.rand(1).item() * 0.2 - 0.1  # Small random reward [-0.1, 0.1]
        self.frames.append(new_frame)
        done = len(self.frames) >= 10
        return self._get_obs(), reward, done, {}

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=3 * 64 * 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 3 * 64 * 64),  # Output same size as input
            nn.Tanh()
        )
        
        # Initialize weights carefully
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.1)
                
    def forward(self, x):
        return self.net(x)

class PPO:
    def __init__(self, policy, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=5000)
        
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
        
    def update(self, batch_size=256, mini_epochs=6):
        if len(self.memory) < batch_size:
            return
            
        states, actions, rewards, next_states, dones, old_log_probs = zip(*random.sample(self.memory, batch_size))
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
        old_log_probs = torch.stack(old_log_probs).view(-1, 1)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for _ in range(mini_epochs):
            # Forward pass
            action_means = self.policy(states)
            
            # Create distribution with stable stddev
            dist = torch.distributions.Normal(
                action_means, 
                torch.ones_like(action_means) * 0.5 + 1e-6
            )
            
            # Calculate log probs with stability
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            log_probs = torch.clamp(log_probs, -20, 20)
            old_log_probs = torch.clamp(old_log_probs, -20, 20)
            
            # Compute advantages
            values = rewards  # Simplified value function
            advantages = rewards - values.detach()
            
            # PPO clipped objective
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Skip update if NaN detected
            if torch.isnan(loss).any():
                continue
                
            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            time.sleep(1)
                
        self.memory.clear()

def train(epochs=10000, model_path="saved_policies"):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    env = PhysicsEnvWrapper()
    policy = PolicyNetwork()
    ppo = PPO(policy, lr=3e-4)
    
    max_episodes = epochs  # 使用命令行参数指定的epochs
    max_steps = 10
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                action_mean = policy(state.unsqueeze(0))
                dist = torch.distributions.Normal(
                    action_mean, 
                    torch.ones_like(action_mean) * 0.5 + 1e-6
                )
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
            
            next_state, reward, done, _ = env.step(action.squeeze(0))
            ppo.store_transition(state, action.squeeze(0), reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        ppo.update()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.4f}")
    
    # 使用命令行参数指定的保存路径 
    save_path = 'saved_policies/final_policy.pth'
    os.makedirs('saved_policies', exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"Policy network saved to {save_path}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='PPO Training for Physics Simulation')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='Number of training epochs (default: 50000)')
    parser.add_argument('--model_path', type=str, default="saved_policies",
                        help='Path to save trained model (default: saved_policies)')
    
    args = parser.parse_args()
    
    # 使用命令行参数调用训练函数
    train(epochs=args.epochs, model_path=args.model_path)