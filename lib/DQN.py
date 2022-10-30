import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lib.replay_memory import Transition, ReplayMemory


class NeuralNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on otuput of Convolution 2D layers and therefore the input image size, so compute it.

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size-1)-1)//stride+1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = torch.sigmoid(x)
        return self.head(x.view(x.size(0), -1))


class DCQL:

    def __init__(self, screen_height, screen_width, n_actions):
        self.n_actions = n_actions
        self.batch_size = 2
        self.gamma = 0.9
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 200

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = NeuralNetwork(
            screen_height, screen_width, n_actions).to(self.device)
        self.target_net = NeuralNetwork(
            screen_height, screen_width, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = np.random.random()
        epsilon_threshold = self.epsilon_end + \
            (self.epsilon_start-self.epsilon_end) * \
            np.exp(-1.*self.steps_done/self.epsilon_decay)
        self.steps_done += 1
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(1, self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transtions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transtions))
        # Compute a mask of non-final states and concatenate the batch elements. a final state would have been the one after which simulation ended
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would have been taken for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)
        """
        Compute V(s_{t+1}) for all next states.
        Exprected values of actions for non-final_next_states are computed based on "older" target_net selecting their best reward with max(1)[0].
        This is merged based on the mask, such that we will have either the expected state value or 0 in case the state was final.
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values*self.gamma)+reward_batch

        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
