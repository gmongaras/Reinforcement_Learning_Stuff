import numpy as np
import gymnasium as gym
import copy
import torch
from torch import nn


name = "CartPole-v1"
seed = None



# Setup the environment
env = gym.make(name)
state, info = env.reset(seed=seed)




# Policy is stochastic
class Policy():
   def __init__(self, ):
      # Size of state space
      self.state_size = 4
      
      # Get the number of actions
      self.num_actions = env.action_space.n
      
      # Probability of nonoptimal action
      self.epsilon = 0.1
      
      # Learning rate
      self.learning_rate = 1.0
      
      # Q (state-action) values
      self.q_values = {}
      
      self.acc_steps = 4
      self.num_steps = 0
      
      
      # Model
      self.device = torch.device("cuda:0")
      self.model = nn.Sequential(
         nn.Linear(self.state_size, 16),
         nn.SiLU(),
         nn.Linear(16, 32),
         nn.SiLU(),
         nn.Linear(32, 32),
         nn.SiLU(),
         nn.Linear(32, 32),
         nn.SiLU(),
         nn.Linear(32, 16),
         nn.SiLU(),
         nn.Linear(16, self.num_actions),
      ).to(self.device)
      self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.01)
      
      
   def quantize(self, mi, ma, x, bins):
      # return (x / ((ma - mi) / bins)).round()
      r = ma - mi
      step = r / (bins)
      return ((x-mi)/step).round()
   
   def reformat_state(self, state):
      state = copy.deepcopy(state)
      
      # Cart position (-4.8 - 4.8)
      # state[0] = self.quantize(-4.8, 4.8, state[0].clip(-4.8, 4.8), 4)
      state[0] = state[0] / 9.6
      
      # Cart velocity
      # state[1] = self.quantize(-10, 10, state[1].clip(-10, 10), num_bins)
      state[1] = state[1].clip(-10, 10) / 20
      
      # Pole angle
      # state[2] = self.quantize(-0.418, 0.418, state[2].clip(-0.418, 0.418).round(3), num_bins)
      state[2] = state[2] / 0.836
      
      # Angular velocity
      # state[3] = self.quantize(-10, 10, state[3].clip(-10, 10), num_bins)
      state[3] = state[3].clip(-10, 10) / 20
      
      # Not going to use the position as it adds more states
      # state = [state[1], state[2], state[3]]
      
      # Make torch tensor
      state = torch.tensor(state).to(self.device)
      
      return state
      
      
   def __call__(self, state, stochastic=True, get_probs=False):
      state = self.reformat_state(state)
      
      # Get all values for every (state, value)
      # inputs = torch.stack([
      #    torch.cat((state, torch.nn.functional.one_hot(torch.tensor(a), self.num_actions).to(state.device))) 
      #    for a in range(0, self.num_actions)
      # ], dim=0)
      values = self.model(state)
      
      # Get the best action
      probs = values.squeeze(-1).softmax(-1)
      best_action = torch.argmax(probs, dim=-1).cpu().numpy().item()
         
      # Return the best action if not stochastic
      if not stochastic:
         if get_probs:
            return (best_action, probs, values)
         return best_action
      
      # Chance of 1-epsilon for best action
      if np.random.random(1)[0] > self.epsilon:
         if get_probs:
            return (best_action, probs, values)
         return best_action
      
      # Select a random action
      p = probs.detach().cpu().numpy()
      p[best_action] = 0.0
      action = np.random.choice(np.arange(0, self.num_actions), p=p/p.sum(), size=1)[0]
      
      if get_probs:
         return (action, probs, values)
      return action
      
      
   
   # Update the Q value, quantize the state
   def update_q_value2(self, state, action, cumulative_rewards, next_state, gamma, values, phi, terminal_state=False):
      # Quantize state
      state = self.reformat_state(state)
      next_state = self.reformat_state(next_state)
      
      
      
      # Get the next state value from the model
      if not terminal_state:
         next_action, _, next_values = self(state, get_probs=True, stochastic=False)
         next_value = next_values[next_action]
      
         # Add the predicted next state value to the values list
         cumulative_rewards.append(next_value.item())
      
      
      
      # Calculate the cumulative reward
      # cumulative_reward = sum([gamma**i * cumulative_reward[i] for i in range(len(cumulative_reward)-1, -1, -1)])
      cumulative_reward = sum([gamma**i * cumulative_rewards[i] for i in range(0, len(cumulative_rewards))])
      
      # New Q value for this state, action
      if len(values) < 2:
         # inputs = torch.stack([
         #    torch.cat((next_state, torch.nn.functional.one_hot(torch.tensor(a), self.num_actions).to(state.device))) 
         #    for a in range(0, self.num_actions)
         # ], dim=0)
         vals = self.model(next_state)
      else:
         vals = values[1]
      vals = values[0][action]
      new_q = values[0][action] + self.learning_rate*phi*(cumulative_reward - (vals.squeeze(-1).max(-1).values))
      
      # Loss is the difference between the current value and new Q value
      # loss = torch.nn.functional.mse_loss(values[0][action], new_q.detach())
      label = values[0].clone()
      label[action] = new_q
      loss = torch.nn.functional.mse_loss(values[0], label.detach())
      
      # Backprop
      loss.backward()
      self.num_steps += 1
      
      # Update model
      if self.num_steps >= self.acc_steps:
         self.optim.step()
         self.optim.zero_grad()
         self.num_steps = 0
      
      
      
   # Get q value for state action
   def get_q_value(self, state, action):
      state = self.reformat_state(state)
      
      if (state, action) not in self.q_values:
         self.q_values[(state, action)] = 0
      
      return self.q_values[(state, action)]






# Create policy
policy = Policy()





# Train model
next_state = state
total_rew = 0
terminated, truncated = False, False
horizon = 2
gamma = 0.95
for step in range(1, 1000001):
   # Iterate until the horizon has been reached or the end of the
   # episode has been reacehd
   h = 0
   G = []
   values = []
   cumulative_phi = 1
   while not terminated and not truncated and h < horizon:
      ### Collect data for this state
      state = next_state
      
      # Get action from policy
      action, probs, value = policy(state, get_probs=True)
      
      # Get the phi conversion
      phi = torch.nn.functional.one_hot(torch.tensor(action).to(probs.device), policy.num_actions)[action] / probs[action]
      # cumulative_phi *= phi
      
      # Get next state and reward
      next_state, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
         reward = -10
      # else:
      #    reward = 0
      
      # Get the behavior to policy ratio
      # cumulative_phi *= policy_probs[action]/behavior_probs[action]
      
      
      
      ### Update G, the total reward for the horizon
      G.append(reward)
      
      ### Update the list of predicted values at each state
      values.append(value)
      
      h += 1
      
      
   # Update model
   if len(values) > 0:
      policy.update_q_value2(state, action, G, next_state, gamma, values, cumulative_phi, terminated or truncated)
   
   # Restart
   if terminated or truncated:
      state, info = env.reset(seed=seed)
      terminated = False
      truncated = False
      
      
   # Inference
   if step % 1000 == 0:
      total_rew = 0
      env = gym.make(name, render_mode="human")
      state, info = env.reset(seed=seed)
      terminated, truncated = False, False
      while not terminated and not truncated:
         state = next_state
         action = policy(state, stochastic=False)
         next_state, reward, terminated, truncated, info = env.step(action)
         total_rew += reward
      print("Reward: ", total_rew)
      
      env = gym.make(name)
      state, info = env.reset(seed=seed)

env.close()