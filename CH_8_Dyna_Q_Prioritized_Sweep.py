import numpy as np
import time
from queue import PriorityQueue



# Known actions
known_actions = {
    "up": np.array([-1, 0]),
    "down": np.array([1, 0]),
    "left": np.array([0, -1]),
    "right": np.array([0, 1])
}



class Environment():
    def __init__(self, ):
        # Environment is a 12x4 gridworld where the reward is -1 everywhere
        self.env = -np.ones((6, 9))
        # Reward is 0 in the goal state
        self.env[0, -1] = 0
        
        # Walls in environment
        self.walls = np.zeros_like(self.env)
        self.walls[3, :-1] = 1
        
        # Reset the environment
        self.reset_env()
        
        # end state
        self.end_state = np.array([0, self.env.shape[1]-1])
        
        # Indices for all parts of the environment that's not a wall or an end state
        self.starting_states = [
            np.array([i, j])
            for i in range(0, self.env.shape[0]) 
            for j in range(0, self.env.shape[1])
            if self.walls[i, j] != 1 and np.all([i, j] != self.end_state)
        ]
        self.starting_states_idx = [i for i in range(0, len(self.starting_states))]
        
    def reset_env(self, ):
        # Reset to the starting state
        self.state = np.array([self.env.shape[0]-1, 3])
        
    def random_reset_env(self, ):
        # Reset to the starting state to a random state
        self.state = self.starting_states[np.random.choice(self.starting_states_idx, size=1)[0]]
        
    def take_action(self, action):
        # State transition
        state_ = (self.state + known_actions[action])
        state_[0] = state_[0].clip(0, self.env.shape[0]-1)
        state_[1] = state_[1].clip(0, self.env.shape[1]-1)
        
        # Update if the state is not on a wall
        if self.walls[state_[0], state_[1]] == 0:
            self.state = state_
        
        # Reward for action
        reward = self.env[self.state[0], self.state[1]]
            
        return reward
    
    def simulate_action(self, action):
        # State transition
        state_ = (self.state + known_actions[action])
        state_[0] = state_[0].clip(0, self.env.shape[0]-1)
        state_[1] = state_[1].clip(0, self.env.shape[1]-1)
        
        # Move back if the state is on a wall
        if self.walls[state_[0], state_[1]] == 1:
            state_ = self.state
        
        # Reward for action
        return self.env[state_, state_]
    
    # Has the episode ended?
    def episode_ended(self, ):
        return np.all(self.state == self.end_state)
    
    
    
    
    
    
    
class Model():
    def __init__(self, ):
        # Stochastic probability
        self.epsilon = 0.1
        
        # Q value tabel (state, action) -> value
        self.q_values = {}
        
        # Environment model table (state, action) -> (reward, next state)
        self.env_model = {}
        
        # List of known actions
        self.actions = list(known_actions.keys())
        self.actions_idx = [i for i in range(0, len(self.actions))]
        
    def __call__(self, state, deterministic=True):
        # Get all values for each action
        values = [self.q_values[(state, action)] if (state, action) in self.q_values else 0 for action in self.actions]
        
        # Get "best" action
        best_idx = np.argmax(values)
        
        # If deterministic, return that action
        if deterministic:
            return self.actions[best_idx]
        
        # Get probs for other actions
        probs = [self.epsilon/(len(self.actions)-1) for i in range(0, len(self.actions))]
        
        # Change prob for best action
        probs[best_idx] = 1 - (sum(probs) - probs[0])
        
        # Get non deterministic action
        return np.random.choice(self.actions, size=1, p=probs)[0]
    
    def get_q_value(self, state, action):
        if (state, action) in self.q_values.keys():
            return self.q_values[(state, action)]
        self.q_values[(state, action)] = 0
        return 0
    
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
        
    # Used to get the probability of a state-action pair
    def get_probs(self, state, action, deterministic=True):
        if deterministic:
            return self.best_action(state) == action
        raise NotImplementedError # I'm lazy
        
        
        
        
        

# Create environment and policy
env = Environment()
model = Model()






# Number of steps for the n-step algorithm (how many
# steps/observations does the algorithm update with
# simulated data from the world model)
n = 10



discount_rate = 0.95
theta = 0.1 # Threshold for the priority
learning_rate = 0.1



# Priority queue
PQueue = PriorityQueue()



# Optimize the policy
for ep in range(0, 1000):
    # # If the environment episode has ended, reset it
    # if env.episode_ended():
    #     env.reset_env()
    
    # Reset the Environment to a random state
    env.random_reset_env()
    
    # Initialize state
    state = tuple(env.state)
    
    # Take action
    action = model(state, deterministic=False)
    
    # Get reward and next state from the action in the environment
    reward = env.take_action(action)
    next_state = tuple(env.state)
    
    
    
    
    
    # Update the world model
    model.env_model[(state, action)] = (reward, next_state)
    
    # Get the priority of the reward - higher priority means the value
    # of the state-action changed more
    priority = abs(
        reward
        + discount_rate*max([
            model.get_q_value(next_state, a)
            for a in model.actions
        ])
        # Updated value
        
        - model.get_q_value(state, action)
        # Old value
    )
    
    # Add state, action to priority queue if the priority is above the threshold
    if priority > theta:
        # Negative priority as lower is chosen first in the queue
        PQueue.put((-priority, (state, action)))
        
    
    
    
    
    
    
    # Loop n times or until the queue is not empty
    # to update based on simulated data
    it = 0
    while it < n and not PQueue.empty():
        # Get the state, action with the most priority
        p, SA = PQueue.get()
        sim_state, sim_action = SA 
        
        # Get simulated reward from env model
        reward, next_stae = model.env_model[(sim_state, sim_action)]
        
        # Update q value of this state, action on simulated experience
        model.q_values[(sim_state, sim_action)] = (
            model.get_q_value(sim_state, sim_action) # Old value
            + learning_rate * (
                reward
                + discount_rate*max([
                    model.get_q_value(next_state, a)
                    for a in model.actions
                ])
                # New value of state-action
                
                - model.get_q_value(sim_state, sim_action)
                # Old value of state-action
            )
        )
        
        
        
        
        
        # Iterate over all state,action pairs that lead
        # to the state that was just updated
        for SA in model.env_model.keys():
            if model.env_model[SA][1] != sim_state:
                continue
            
            state_, action_ = SA
            
            # Get predicted reward for this state and action
            reward_ = model.env_model[SA][0]
            
            # Get priority of this state,action
            priority = abs(
                reward_
                + discount_rate*max([
                    model.get_q_value(sim_state, a)
                    for a in model.actions
                ])
                # Updated value
                
                - model.get_q_value(state_, action_)
                # Old value
            )
            
            # Add state, action to priority queue if the priority is above the threshold
            if priority > theta:
                # Negative priority as lower is chosen first in the queue
                PQueue.put((-priority, (state_, action_)))
                
        
        
        
        
        
        it += 1

    
    
# Show policy and how it moves on a grid, no randomness
env.reset_env()
while not env.episode_ended():
    state = [list(i) for i in (env.walls == 1).astype(int)]
    # Put X in current states
    state[env.state[0]][env.state[1]] = "X"
    print(np.array(state))
    print(model.best_action(tuple(env.state)))
    print()
    env.take_action(model.best_action(tuple(env.state)))
    time.sleep(0.5)