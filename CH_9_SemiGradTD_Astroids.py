import numpy as np
import gymnasium as gym
import copy
import torch
from torch import nn


name = "ALE/Asteroids-v5"
seed = None



# Setup the environment
env = gym.make(name)
state, info = env.reset(seed=seed)




# Policy is stochastic
class Policy():
    def __init__(self, ):
        # Size of state space
        self.state_size = 3
        
        # Get the number of actions
        self.num_actions = env.action_space.n
        
        # Probability of nonoptimal action
        self.epsilons = np.linspace(0.1, 0.001, 10_000)
        self.epsilon = self.epsilons[0]
        
        # Learning rate
        self.learning_rate = 0.001
        
        # Q (state-action) values
        self.q_values = {}
        
        self.acc_steps = 8
        self.num_steps = 0
        
        
        # Model
        self.device = torch.device("cpu")
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8960, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        ).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        
    def quantize(self, mi, ma, x, bins):
        # return (x / ((ma - mi) / bins)).round()
        r = ma - mi
        step = r / (bins)
        return ((x-mi)/step).round()
    
    def reformat_state(self, state):
        state = copy.deepcopy(state)
        
        # Move between 0 and 1
        state = state / 255.0
        
        # Make torch tensor
        state = torch.tensor(state).to(self.device).to(torch.float32)
        
        # Move channels first and unsqueeze
        state = state.permute(2, 0, 1).unsqueeze(0)
        
        return state
        
        
    def __call__(self, state, stochastic=True, get_probs=False):
        state = self.reformat_state(state)
        
        # Get all values for every (state, value)
        # inputs = torch.stack([
        #    torch.cat((state, torch.nn.functional.one_hot(torch.tensor(a), self.num_actions).to(state.device))) 
        #    for a in range(0, self.num_actions)
        # ], dim=0)
        values = self.model(state)[0]
        
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
        
        # # Select a random action
        # p = probs.detach().cpu().numpy()
        # p[best_action] = 0.0
        # action = np.random.choice(np.arange(0, self.num_actions), p=p/p.sum(), size=1)[0]
        
        p = np.ones(probs.shape[0])
        p[best_action] = 0.0
        p = p / p.sum()
        action = np.random.choice(np.arange(0, self.num_actions), p=p, size=1)[0]
        
        if get_probs:
            return (action, probs, values)
        return action
      
   
    # Update the Q value, quantize the state
    def update_q_value(self, cumulative_reward, state, values, phi):
        state = self.reformat_state(state)
        
        # Loss is the difference between the cumulative reward and the current value
        loss = (cumulative_reward.detach() - values)**2
        
        # Reweight loss according to the importance sampling ratio
        loss = loss * phi
        
        # Backprop
        loss.backward()
        self.num_steps += 1
        
        # Update model
        if self.num_steps >= self.acc_steps:
            self.optim.step()
            self.optim.zero_grad()
            # for param in self.model.parameters():
            #     param.data = param.data - 1/2 *  self.learning_rate * phi * param.grad.data
            #     param.grad.data.zero_()
            self.num_steps = 0
            
            
    def get_q_value(self, state, action):
        state = self.reformat_state(state)
        return self.model(state)[0][action]
    
    def get_q_values(self, state):
        state = self.reformat_state(state)
        return self.model(state)[0]






# Create policy
policy = Policy()





# Train model
next_state = state
total_rew = 0
terminated, truncated = False, False
horizon = 3
gamma = 0.99
for step in range(1, 1000001):
    # Update model epsilon
    if step < len(policy.epsilons):
        policy.epsilon = policy.epsilons[step]
    
    # Reset environment
    init, info = env.reset(seed=seed)
    
    T = np.inf
    
    # Iterate until the horizon has been reached or the end of the
    # episode has been reacehd
    t = 0
    tau = -1
    rewards = {}
    states = {}
    actions = {}
    prev_lives = info["lives"]
    while tau != T-1:
        # If the end of the episode has not been reached, then
        # take an action
        if t < T:
            # Take action according to policy
            action = policy(state)
            actions[t] = action
            states[t] = state
            
            # Get next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            rewards[t+1] = reward
            states[t+1] = state
            
            # If the next state is terminal, then set T to the current time
            # to do one last update
            if terminated or truncated:
                T = t + 1
                
                if terminated:
                    rewards[t+1] = 0
                
                # Large negative termination reward if the pole reached the edge
                # if state[0] < -4.5 or state[0] > 4.5:
                #     rewards[t+1] = -100
                # else:
                #     rewards[t+1] = -1
                
            # Otherwise, get the next action from the policy
            else:
                actions[t+1] = policy(next_state)
                
                # Reward is -1 when an astroid is not hit
                if rewards[t+1] == 0:
                    rewards[t+1] = 0
            
            
        # Tau is the time whose estimate of the value of the state is being updated
        tau = t - horizon + 1
        
        # If tau is greater than or equal to 0, then update the value of the state
        if tau >= 0:
            # Cumulative reweighing - importance sampling ratio
            phi = 1
            for i in range(tau+1, min(tau+horizon, T-1)+1):
                beh = policy.get_q_values(states[i]).softmax(-1)[actions[i]].item() + 1e-4
                pol = 1.0 if actions[i] == policy(states[i], stochastic=False) else 0.0
                phi *= pol / beh
            
            # Get the cumulative discounted reward
            G = sum([gamma**(i-tau-1) * rewards[i] for i in range(tau+1, min(tau+horizon, T)+1)])
            
            # If this is not the last state, then add the future predicted value of the next state
            if tau + horizon < T:
                G += gamma**horizon * policy.get_q_value(states[tau+horizon], actions[tau+horizon])
            else:
                G = torch.tensor(G).to(policy.device)
                
            # Update the value of the state
            policy.update_q_value(G, states[tau], policy.get_q_value(states[tau], actions[tau]), phi)
            
            
            
        
        t += 1
        state = next_state
        
    # Restart environment
    state, info = env.reset(seed=seed)
    terminated = False
    truncated = False
      
      
    # Inference
    if step % 10 == 0:
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