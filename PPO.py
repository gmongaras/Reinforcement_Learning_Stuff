import gymnasium
import torch
from torch import nn
from copy import deepcopy


# Continuous Policy network
class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Policy, self).__init__()
        
        # Feed forward with discrete actions
        self.ff = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
            nn.LogSoftmax(dim=-1)
        )
        
        
    def forward(self, s_t):
        if not isinstance(s_t, torch.Tensor):
            s_t = torch.tensor(s_t, dtype=torch.float32)
        return self.ff(s_t)
    
    def sample_action(self, s_t, deterministic=False):
        # Get action probabilities
        log_probs = self.forward(s_t)
        
        # Get action
        if deterministic:
            action = torch.argmax(log_probs)
        else:
            action = torch.distributions.Categorical(logits=log_probs).sample()
        
        return action.item(), log_probs[:, action.item()]
    
    # Get probs of a specific state, action
    def probs_of_action(self, s_t, a_t):
        log_probs = self.forward(s_t)
        return log_probs.gather(1, a_t.unsqueeze(1))
    
    
# Value network
class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        
        # Feed forward - state to value of the state
        self.ff = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
    def forward(self, s_t):
        if not isinstance(s_t, torch.Tensor):
            s_t = torch.tensor(s_t, dtype=torch.float32)
        return self.ff(s_t)
        
        
        
# PPO agent
class PPOAgent():
    def __init__(self, num_inputs, num_actions, gamma=0.99, K_epochs=4, eps_clip=0.2, learning_rate=0.0003):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # Policy and value networks
        self.policy = Policy(num_inputs, num_actions)
        self.value = Value(num_inputs)
        
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)
        self.optimizer_value = torch.optim.AdamW(self.value.parameters(), lr=learning_rate)
        
        
        
        
# Run the agent in the environment
def collect_data(env, agent, horizon):
    """
    Collects data from the environment by running the given policy.
    
    Parameters:
        env: The environment instance.
        policy: The policy model to use for action selection.
        horizon: The number of steps to collect data for.
    
    Returns:
        A tuple containing lists of states, actions, rewards, log probabilities, state values, and done flags.
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    
    s_t, _ = env.reset()
    done = False

    for _ in range(horizon):
        # Convert state to tensor for policy
        state_tensor = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # Get value and action for this state
        # No need to compute gradients for data collection
        with torch.no_grad():
            action, log_prob = agent.policy.sample_action(state_tensor)
            value = agent.value(state_tensor)
        
        # Take action in environment according to the policy
        s_t1, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Reward is 0 if the episode is terminated - died
        if terminated:
            reward = 0
        # Reward is 2 if the episode is truncated - went to the end
        if truncated:
            reward = 2
        
        # Store data
        states.append(s_t)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)
        
        # Update state
        s_t = s_t1
        
        # Reset environment if done
        if done:
            s_t, _ = env.reset()
    
    return states, actions, rewards, log_probs, values, dones





# Process rewards and values
def process_rewards(rewards, values, gamma, lambda_, dones):
    """
    Compute the returns and the Generalized Advantage Estimation (GAE)
    for each timestep in a trajectory.

    Parameters:
        rewards: List of rewards for each timestep in the trajectory.
        values: List of value estimates V(s) for each timestep.
        gamma: Discount factor.
        lambda_: GAE parameter for balancing bias-variance.
        dones: List indicating whether a timestep is the last in an episode.

    Returns:
        returns: The computed returns for each timestep.
        advantages: The computed GAE advantages for each timestep.
    """
    
    # Convert lists to tensors
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32).squeeze()  # Ensure values is a 1D tensor
    dones = torch.tensor(dones, dtype=torch.float32)

    # Initialize tensors for returns and advantages
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    # Temporary variables for the reversed computation
    gae = 0
    next_value = 0

    # Compute returns and GAE advantages in reverse order
    for t in reversed(range(len(rewards))):
        # If the current timestep is the last in an episode, the next non-terminal state value is 0
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t]
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        ## Compute the TD error and GAE
        
        # delta = r + gamma * V(s') * (1 - done) - V(s)
        # TD error is the difference between the value of the current state and the value of the next state
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE is a weighted combination of the TD error and the previous GAE
        gae = delta + gamma * lambda_ * next_non_terminal * gae
        
        # Compute the return: GAE + value estimate
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return returns, advantages
    
    
    
    
    
num_iters = 1000
num_epochs = 3
num_actors = 1
horizon_T = 512
epsilon_clip = 0.2
minibatch_size = 64
gamma = 0.99
GAE_param = 0.95
    
    


# Gym Pole environment
env = gymnasium.make('CartPole-v1')
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
s_t, info = env.reset()
terminated = truncated = False

# PPO agent
agent = PPOAgent(num_inputs, num_actions)

# Initialize the old policy with the current one
policy_old = deepcopy(agent.policy)















# PPO agent update function
def update_agent(agent, states, actions, log_probs_old, returns, advantages, eps_clip, K_epochs, minibatch_size):
    actions = torch.tensor(actions).detach()
    states = torch.tensor(states).detach()
    log_probs_old = torch.stack(log_probs_old).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

    for _ in range(K_epochs):
        # Create random indices for minibatches
        indices = torch.randperm(len(states))

        for i in range(0, len(states), minibatch_size):
            idx = indices[i:i+minibatch_size]
            sampled_states = states[idx]
            sampled_actions = actions[idx]
            sampled_log_probs_old = log_probs_old[idx]
            sampled_returns = returns[idx]
            sampled_advantages = advantages[idx]

            # Compute new log probabilities and state values
            log_probs = agent.policy.probs_of_action(sampled_states, sampled_actions)
            state_values = agent.value(sampled_states)
            state_values = state_values.squeeze()

            # Ratio for clipping
            ratios = torch.exp(log_probs - sampled_log_probs_old)

            # Objective function components
            surr1 = ratios * sampled_advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * sampled_advantages

            # Loss function
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (sampled_returns - state_values).pow(2).mean()

            # Total loss
            loss = policy_loss + value_loss

            # Take optimization step
            agent.optimizer_policy.zero_grad()
            agent.optimizer_value.zero_grad()
            loss.backward()
            agent.optimizer_policy.step()
            agent.optimizer_value.step()

# Your main training loop
for iteration in range(num_iters):
    # Reset the environment and collect data
    states, actions, rewards, log_probs, values, dones = collect_data(env, agent, horizon_T)
    
    # Process collected data (compute returns, advantages)
    returns, advantages = process_rewards(rewards, values, gamma, GAE_param, dones)
    
    # Update the policy and value network
    update_agent(agent, states, actions, log_probs, returns, advantages, epsilon_clip, num_epochs, minibatch_size)
    
    # Visual Simulation of policy in environment
    env_ = gymnasium.make('CartPole-v1', render_mode="human")
    terminated = truncated = False
    s_t, info = env_.reset()
    # Run policy until termination
    while not terminated and not truncated:
        a_t, _ = agent.policy.sample_action(torch.tensor([s_t]), deterministic=True)
        s_t, r_t, terminated, truncated, info = env_.step(a_t)