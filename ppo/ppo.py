import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = [] #log probs
        self.vals = [] #state values
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) # np.arrange creates an array of evenly spaced values, thus we are creating an array of batch start indices
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices) # shuffle the indices for stochastic gradient ascent
        batches = [indices[i:i+self.batch_size] for i in batch_start] # for each batch start index, we create a batch of indices. Looks like [[0, 1, 2, ..., 63], [64, 65, 66, ..., 127], ...]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        # clear the memory after each trajectory
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module): # The nn.Module class is the base class for all neural network modules in PyTorch
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'): # checkpoint directory is where we will save the model
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims), # TODO Why do we unpack input_dims? How many dimensions does input_dims have?
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1) #dim=-1 means we are applying softmax across the action dimension and not the batch dimension
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)
        return distribution
    
    def save_checkpoint(self):
        # save the model parameters
        T.save(self.state_dict(), self.checkpoint_file) # state_dict() returns a dictionary containing a whole state of the module

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, inputs_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*inputs_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1) # output is 1 because Critic predicts the state value
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10): # hyperparameters from paper. N is the number of timesteps before we update the network
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epoch = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('Saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('Loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device) #  We want a 2D tensor with shape (batch_size, num_features). We are adding a batch dimension to the observation by wrapping it in a list
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample() # sample an action from the distribution

        probs = T.squeeze(dist.log_prob(action)).item() # squeeze removes redundant dimensions from the tensor. item() returns the value of this tensor as a standard Python number
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epoch): # we repeat this process n_epoch times
            
            # first we generate the batches that we will use to update the network
            states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
            
            # Calculate advantages
            advantage = np.zeros(len(rewards), dtype=np.float32) # initialize advantage array with zeros for each timestep
            # For every timestep that we have in memory, we calculate the discounted sum of future rewards and subtract the state value to get the advantage
            # We iterate over rewards because its standard practice to have a reward for each timestep, even if the reward is 0.
            # The advantage is the discounted sum of future rewards minus the estimated state value (from the critic)
            for t in range(len(rewards)-1): # we don't calculate the advantage for the last timestep because there is no next state, so we leave it as 0
                discount = 1 # we initialize the discount factor to 1 and decrease it for each timestep by multiplying it by gamma*lambda
                a_t = 0
                for k in range(t, len(rewards)-1): # adv = dt + gl*dt+1 + (gl)^2*dt+2 + ... + (gl)^(T-t-1)*dt+T-1; so here we are looping over the timesteps from t to T-1
                    value_next_state = self.gamma*values[k+1] * (1-int(dones[k])) # if dones[k] is 1, then we are at the end of the episode and there is no next state and so we multiply by 0. dones is a boolean array
                    td_error = rewards[k] + value_next_state - values[k]
                    a_t += discount*td_error
                    discount *= self.gamma*self.gae_lambda # rewards further in the future are discounted more
                advantage[t] = a_t

            # We end up with an array of advantages for each timestep, which represents how good Q(s,a) is compared to V(s)
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device) # values contains the estimated state values from the critic for all the states that the agent has visited in memory. We need it for the critic loss

            for batch in batches:

                # here a batch is a subset of the experience that we have in memory. each experience is a tuple of (state, action, old_prob, value, reward, done)
                # create tensors of relevant data for the current batch
                states_batch = T.tensor(states[batch], dtype=T.float).to(self.actor.device) # the state during each experience in the batch, TODO ?2D tensor?
                old_probs_batch = T.tensor(old_probs[batch]).to(self.actor.device) # 1D tensor of log probabilities for each action taken in each experience
                actions_batch = T.tensor(actions[batch]).to(self.actor.device) # tensor of actions taken in each experience. If discrete action space, then this is a 1D tensor. If continuous action space, then this is a 2D tensor TODO check

                # earlier in the training loop the agent interacts with the environment and stores old action probabilities in the memory
                # now we want to calculate the new action probabilities using the current policy
                dist = self.actor(states_batch) # forward method is automatically called because we inherited from nn.Module
                critic_value = self.critic(states_batch)
                critic_value = T.squeeze(critic_value)

                # Calculate loss for actor
                new_probs = dist.log_prob(actions_batch) # dist.log_prob gives the logprob of the given actions, so we get the logprob of each action in actions_batch
                prob_ratio = new_probs.exp() / old_probs_batch.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch] # clamp takes three arguments: the tensor, the min value, and the max value. It returns a tensor with the same shape as the input tensor but with all the values clamped between min and max
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() # we negate the loss because we want to maximize instead of minimize

                # Calculate loss for critic using TD learning TODO check if it's TD learning
                returns = advantage[batch] + values[batch]  # current estimate of the return by the critic
                critic_loss = (returns-critic_value)**2 # Loss is the MSE between the returns and the critic value
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss # TODO why do we combine the actor and critic loss like this?

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
        self.memory.clear_memory()

            
