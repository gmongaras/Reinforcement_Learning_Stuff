import numpy as np
import gymnasium as gym
import copy


name = "CartPole-v1"
seed = None



# Setup the environment
env = gym.make(name)
state, info = env.reset(seed=seed)




# Policy is stochastic
class Policy():
   def __init__(self, ):
      # Get the number of actions
      self.num_actions = env.action_space.n
      
      # Probability of nonoptimal action
      self.epsilon = 0.1
      
      # Learning rate
      self.learning_rate = 0.1
      
      # Q (state-action) values
      self.q_values = {}
      
      
   def quantize(self, mi, ma, x, bins):
      # return (x / ((ma - mi) / bins)).round()
      r = ma - mi
      step = r / (bins)
      return ((x-mi)/step).round()
   
   def reformat_state(self, state):
      state = copy.deepcopy(state)
      num_bins = 20
      
      # Cart position (-4.8 - 4.8)
      state[0] = self.quantize(-4.8, 4.8, state[0].clip(-4.8, 4.8), 4)
      
      # Cart velocity
      state[1] = self.quantize(-10, 10, state[1].clip(-10, 10), num_bins)
      
      # Pole angle
      state[2] = self.quantize(-0.418, 0.418, state[2].clip(-0.418, 0.418).round(3), num_bins)
      
      # Angular velocity
      state[3] = self.quantize(-10, 10, state[3].clip(-10, 10), num_bins)
      
      # Not going to use the position as it adds more states
      # state = [state[1], state[2], state[3]]
      
      # Make tuple
      state = tuple(state)
      
      return state
      
      
   def __call__(self, state, stochastic=True, get_probs=False):
      state = self.reformat_state(state)
      
      # Get the best action
      vals = [
         self.q_values[(state, action)]
         if (state, action) in self.q_values else -np.inf
         for action in range(0, self.num_actions)
      ]
      best_action = np.argmax(vals)
      
      # If all are -infinity, choose a random action
      if np.all(np.isinf(vals)):
         best_action = np.random.choice(np.arange(0, self.num_actions), size=1)[0]
         
      # Probabilities of each action (stochastic)
      probs = np.ones(self.num_actions)*(self.epsilon/(self.num_actions-1))
      probs[best_action] = 1.0-self.epsilon
         
      # Return the best action if not stochastic
      if not stochastic:
         if get_probs:
            pol_probs = np.zeros(self.num_actions)
            pol_probs[best_action] = 1.0
            return (best_action, probs, pol_probs)
         return best_action
      
      # Select a random action
      action = np.random.choice(np.arange(0, self.num_actions), p=probs, size=1)[0]
      
      if get_probs:
         pol_probs = np.zeros(self.num_actions)
         pol_probs[best_action] = 1.0
         return (action, probs, pol_probs)
      return action
   
   
   # Update the Q value, quantize the state
   def update_q_value(self, state, action, reward, next_state):
      # Quantize state
      state = self.reformat_state(state)
      next_state = self.reformat_state(next_state)
      
      SA = (tuple(state), action)
      
      # Update q value
      if SA not in self.q_values:
         self.q_values[SA] = 0
      self.q_values[SA] = (
         # Current value
         self.q_values[SA]
         
         # Error for update
         + self.learning_rate * (
            reward + max([
               self.q_values[(tuple(next_state), a)]
               if (tuple(next_state), a) in self.q_values else 0
               for a in range(0, self.num_actions)
            ])
            # New value estimate
            
            - self.q_values[SA]
            # Current value
         )
      )
      
   # Update the Q value, quantize the state
   def update_q_value2(self, state, action, cumulative_reward, next_state, gamma, phi):
      # Quantize state
      state = self.reformat_state(state)
      next_state = self.reformat_state(next_state)
      
      SA = (tuple(state), action)
      
      # Calculate the cumulative reward
      cumulative_reward = sum([gamma**i * cumulative_reward[i] for i in range(len(cumulative_reward)-1, -1, -1)])
      
      # Update q value
      if SA not in self.q_values:
         self.q_values[SA] = 0
      self.q_values[SA] = (
         # Current value
         self.q_values[SA]
         
         # Error for update
         + self.learning_rate * phi * (
            cumulative_reward + max([
               self.q_values[(tuple(next_state), a)]
               if (tuple(next_state), a) in self.q_values else 0
               for a in range(0, self.num_actions)
            ])
            # cumulative_reward + self.q_values[(tuple(state), action)]
            # New value estimate
            
            - self.q_values[SA]
            # Current value
         )
      )
      
      
      
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
gamma = 0.9
for step in range(1, 1000001):
   # Iterate until the horizon has been reached or the end of the
   # episode has been reacehd
   h = 0
   G = []
   cumulative_phi = 1
   while not terminated and not truncated and h < horizon:
      ### Collect data for this state
      state = next_state
      
      # Get action from policy
      action, behavior_probs, policy_probs = policy(state, get_probs=True)
      
      # Get next state and reward
      next_state, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
         reward = -100
      # else:
      #    reward = 0
      
      # Get the behavior to policy ratio
      # cumulative_phi *= policy_probs[action]/behavior_probs[action]
      
      
      
      ### Update G, the total reward for the horizon
      G.append(reward)
      
      h += 1
      
      
   # Update model
   policy.update_q_value2(state, action, G, next_state, gamma, cumulative_phi)
   
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