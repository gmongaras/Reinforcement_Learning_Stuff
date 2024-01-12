import numpy as np
import time



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
        self.env = -np.ones((10, 12))
        # Reward is -100 where there is a cliff
        self.env[-1, 1:-1] = -100
        self.env[-2, 3:-3] = -100
        self.env[1:, 7] = -100
        self.env[:-3, 5] = -100
        self.env[1:, 3] = -100
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
        self.state = (self.state + known_actions[action])
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
        state = (self.state + known_actions[action])
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
        
        self.actions = list(known_actions.keys())
        self.actions_idx = [i for i in range(0, len(self.actions))]
        
    def __call__(self, state, deterministic=True, return_probs=False):
        # Get all values for each action
        values = [self.q_values[(state, action)] if (state, action) in self.q_values else 0 for action in self.actions]
        
        # Get "best" action
        best_idx = np.argmax(values)
        
        # If deterministic, return that action
        if deterministic:
            return best_idx
        
        # Get probs for other actions
        probs = [self.epsilon/(len(self.actions)-1) for i in range(0, len(self.actions))]
        
        # Change prob for best action
        probs[best_idx] = 1 - (sum(probs) - probs[0])
        
        # Get non deterministic action
        if return_probs:
            # Idx of the action chosen
            idx = np.random.choice(self.actions_idx, size=1, p=probs)[0]
            
            # action taken, nondeterministic prob, deterministic probs
            return self.actions[idx], probs[idx], idx == best_idx
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
policy = Policy()






# Number of steps for the n-step algorithm (how many
# steps/observations does the algorithm update with
# a single reward)
n = 1


# Sigmas for each step in the n-step chain
sigmas = np.linspace(1.0, 0.0, num=n)



# Optimize the policy
for ep in range(0, 1000):
    # Reset the Environment
    env.reset_env()
    
    # Initialize the first state and action
    state = env.state
    action = policy(tuple(state), deterministic=False)
    
    # Rewards, actions, and states for each step in the episode
    rewards = {}
    actions = {0:action}
    states  = {0:tuple(state)}
    
    # Phi is the importance sampling reweighing factor
    phis = {}
    
    # Sigmas are the interpolation factor between SARSA and
    # tree backup. 1 is SARSA, 0 is tree update
    sigmas = {}
    
    # Sum or episode rewards
    sum_rew = 0
    
    # Iterate until the end of the episode is reached
    t = 0
    T = np.inf
    while not env.episode_ended():
        # If we have not reached the end of the episode, take the
        # action predicted from the behavior policy
        if t < T:
            # Take action, get reward and next state
            reward = env.take_action(action)
            state = env.state
            
            # Store in memory
            rewards[t+1] = reward
            states[t+1] = tuple(state)
            
        # If the state is terminal, set T to be the next step so that the
        # episode ends and the model is updated one last time
        if env.episode_ended():
            T = t + 1
        
        # If the state is not terminal, then we can get a new action from the behavior policy
        # which will be used to transition to the next state.
        else:
            # Get a new action from the behavior policy
            action, behavior_Probs, policy_Probs = policy(tuple(state), deterministic=False, return_probs=True)
            
            # Store the action
            actions[t+1] = action
            
            # Store sigma value for this step in the n-step chain
            sigmas[t+1] = 0.6
            
            # Store the importance sampling reweighing factor
            # NOTE: This is the importance sampling for the SARSA method
            phis[t+1] = policy_Probs/behavior_Probs
            
            
            
            
        # Update tau, this is used to go though the n steps before updating
        tau = t - n + 1
        
        
        
        
        
        # If tau > 0, then we know there's been n steps and we can update
        if tau >= 0:
            # Used to skip the last step in the episode
            if t + 1 < T:
                G = policy.get_q_value(states[t+1], actions[t+1])
                
            # Iterate from the end of the n-steps to the
            # beginning of the n-steps
            for k in range( min(t+1, T), tau, -1 ):
                # Initialize G to the reward at time T if the end of the episode has been reached
                if k == T:
                    G = rewards[T] # reward at time T
                
                # If the end of the episode has not been reached,
                # update G with the current value for this step in
                # the n-step chain
                else:
                    # Get the value of each state-action weighed by the probabilities.
                    # NOTE: This is the tree-backup method
                    V_bar = sum([
                        policy.get_probs(states[k], a) * policy.get_q_value(states[k], a) for a in policy.actions
                    ])
                    
                    # Update the cumulative value
                    G = (
                        rewards[k] # Reward for the current action in this state
                        + policy.gamma * ( # 
                            sigmas[k] * phis[k] # This is the importance sampling reweigh factor for the SARSA method
                            + (1-sigmas[k]) * policy.get_probs(states[k], actions[k]) # This is the reweigh factor for tree backup
                        )
                        * (G - policy.get_q_value(states[k], actions[k])) # Error in the current q value
                        + policy.gamma * V_bar # weighted expected value for this state (based on probs of policy)
                    )
                        
                        
            # Update the value for the state-action at the beginning of the n-step
            # chain. This is essentially updating the current state-value beased on
            # n steps into the future as G is from n steps into the future.
            policy.q_values[(states[tau], actions[tau])] = policy.get_q_value(states[tau], actions[tau]) \
                + policy.learning_rate * (G - policy.get_q_value(states[tau], actions[tau]))
              
              
        # Go to next step  
        t += 1
        
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