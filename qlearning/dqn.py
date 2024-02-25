import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DQN, self).__init__()

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        # print(self.input_dims) # TODO cleanup
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.network = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims), # TODO understand the shape of input_dims
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions)
        )

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # TODO understand the shape of input_dims
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # actions = self.fc3(x)

        actions = self.network(state)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)] # e.g. [0, 1, 2, 3] if there are 4 actions
        self.mem_size = max_mem_size
        self.eps_end= eps_end
        self.eps_dec = eps_dec

        self.mem_counter = 0

        self.DQN = DQN(self.lr, n_actions, input_dims=input_dims,
                            fc1_dims=256, fc2_dims=256)
        
        # Initialising memories
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # state where the action was taken
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # state the agent ended up in
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, action, state, reward, state_, done):
        """Store a new experience in memory"""
        index = self.mem_counter % self.mem_size # find the index of the memory to be replaced
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # take best known action
            state = T.tensor([observation]).to(self.DQN.device) # we put observation in a list because we want a batch of size 1
            actions = self.DQN.forward(state)
            action = T.argmax(actions).item()
        else:
            # take random action epsilon % of the time
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_counter < self.batch_size:
            return # don't learn until we have enough memories for a batch
        
        self.DQN.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size) # position of the last memory

        batch = np.random.choice(max_mem, self.batch_size, replace=False) # choose a random batch of memories, don't select the same memory twice. np.random.choice(5, 3) = [1, 3, 4]
        batch_index = np.arange(self.batch_size, dtype=np.int32) # np.arange(5) = [0, 1, 2, 3, 4]

        state_batch = T.tensor(self.state_memory[batch]).to(self.DQN.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.DQN.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.DQN.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.DQN.device)

        action_batch = self.action_memory[batch] # we're only using this for indexing so doesn't need to be a tensor

        q_current = self.DQN.forward(state_batch)[batch_index, action_batch] # we index into the actions we took in the batch
        q_next = self.DQN.forward(new_state_batch) # 2D tensor of the q values for each action in the batch, shape (state in the batch, n_actions)

        # terminal_batch is a boolean tensor of shape (batch_size, ) where True means the state is terminal.
        # When we use it to index into q_next we get a 1D tensor of shape (n_actions, ) only for the cases where the state is terminal.
        # We set these to 0 because there is no future reward for a terminal state.
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] # dim=1 means we take the max over the actions, not the states; max returns a tuple (values, indices) so we take the first element

        loss = self.DQN.loss(q_target, q_current).to(self.DQN.device)
        loss.backward()
        self.DQN.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end # decay epsilon







        