import numpy as np
import time



# Known actions
actions = {
    "up": np.array([-1, 0]),
    "down": np.array([1, 0]),
    "left": np.array([0, -1]),
    "right": np.array([0, 1])
}



class Environment():
    def __init__(self, ):
        # Environment is a 12x4 gridworld where the reward is -1 everywhere
        self.env = -np.ones((4, 12))
        # Reward is -100 where there is a cliff
        self.env[-1, 1:-1] = -100
        self.env[-2, 3:-3] = -100
        # Reward is 0 in the goal state
        self.env[-1, -1] = 0
        
        # Reset the environment
        self.reset_env()
        
    def reset_env(self, ):
        # Reset to the starting state
        self.state = np.array([self.env.shape[0]-1, 0])
        
        # Reset the end state
        self.end_state = np.array([self.env.shape[0]-1, self.env.shape[1]-1])
        
    def take_action(self, action):
        # State transition
        self.state = (self.state + actions[action])
        self.state[0] = self.state[0].clip(0, self.env.shape[0]-1)
        self.state[1] = self.state[1].clip(0, self.env.shape[1]-1)
        
        # Reward for action
        reward = self.env[self.state[0], self.state[1]]
        
        # If the reward is -100 (cliff), go back to the start
        if reward == -100:
            self.state = np.array([self.env.shape[0]-1, 0])
            
        return reward
    
    def simulate_action(self, action):
        # State transition
        state = (self.state + actions[action])
        state[0] = state[0].clip(0, self.env.shape[0])
        state[1] = state[1].clip(0, self.env.shape[1])
        
        # Reward for action
        return self.env[state]
    
    # Has the episode ended?
    def episode_ended(self, ):
        return np.all(self.state == self.end_state)
    
    
    
    
    
    
    
class Policy():
    def __init__(self, ):
        self.epsilon = 0.1
        
        self.learning_rate = 0.1
        
        # Discount
        self.gamma = 1.0
        
        self.q_values = {}
        
        self.actions = list(actions.keys())
        
    def __call__(self, state):
        # Get all values for each action
        values = [self.q_values[(state, action)] if (state, action) in self.q_values else 0 for action in self.actions]
        
        # Get "best" action
        best_idx = np.argmax(values)
        
        # Get probs for other actions
        probs = [self.epsilon/(len(self.actions)-1) for i in range(0, len(self.actions))]
        
        # Change prob for best action
        probs[best_idx] = 1 - (sum(probs) - probs[0])
        
        # Get action
        return np.random.choice(self.actions, size=1, p=probs)[0]
    
    def best_action(self, state):
        return self.actions[np.argmax([self.q_values[(state, action)] if (state, action) in self.q_values else 0 for action in self.actions])]
    
    def update(self, state, next_state, action, reward):
        # Q value in current state
        cur_state_value = self.q_values[(state, action)] if (state, action) in self.q_values else 0
        
        # Get the value for each action in the next state
        next_state_values = [self.q_values[(next_state, action)] if (next_state, action) in self.q_values else 0 for action in self.actions]
        
        # Get max Q value for update
        max_q_value = max(next_state_values)
        
        # Update q value
        self.q_values[(state, action)] = cur_state_value + self.learning_rate * (reward + self.gamma*max_q_value - cur_state_value)
        
        
        
        
        

# Create environment and policy
env = Environment()
policy = Policy()




# Optimize the policy
for ep in range(0, 500):
    # Reset the Environment
    env.reset_env()
    
    # Sum or episode rewards
    sum_rew = 0
    
    # Iterate until the episode ends
    while not env.episode_ended():
        # Current state
        state = env.state
        
        # Get action based on policy
        action = policy(tuple(state))
        
        # Take action, get reward
        reward = env.take_action(action)

        # Update Q value
        policy.update(tuple(state), tuple(env.state), action, reward)
        
        sum_rew += reward
        
    print(sum_rew)
    
    
# Show policy and how it moves on a grid, no randomness
env.reset_env()
while not env.episode_ended():
    state = [list(i) for i in (env.env == -100).astype(int)]
    # Put X in current states
    state[env.state[0]][env.state[1]] = "X"
    print(np.array(state))
    print(policy.best_action(tuple(env.state)))
    print()
    env.take_action(policy.best_action(tuple(env.state)))
    time.sleep(0.5)