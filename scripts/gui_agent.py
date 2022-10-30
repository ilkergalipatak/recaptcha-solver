import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import matplotlib.pyplot as plt
import matplotlib
from envs.recaptcha import Recaptcha
from lib.DQN import DCQL
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from itertools import count



class GuiAgent:
    def __init__(self):
        self.env = Recaptcha()
        screen_height, screen_width = self.env.resolution
        n_actions = self.env.action_space.n
        self.agent = DCQL(screen_height, screen_width, n_actions)
        self.DCQN_params = {}
        self.n_episodes = 50
        self.target_update = 10
        self.episode_durations = []
        self.bacground = np.random.randint(10, 20, size=self.env.resolution)

    def plot_durations(self):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()

        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        return

    def get_screen(self, position):
        I = np.copy(self.bacground)
        I[position[0]][position[1]] = 255

        # Convert to float, rescale, convert to torch tensor (CHW) this does not require a copy
        screen_height, screen_width = self.env.resolution
        screen = I.reshape(1, screen_height, screen_width)
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)

        # Resize and ad a batch dimension (BCHW)
        resize = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()])

        return resize(screen).unsqueeze(0).to(self.agent.device)

    def render(self):
        for episode in range(self.n_episodes):
            # Initialize the environment and state
            self.env.reset()
            state = self.get_screen(self.env.state)
            for t in count():
                # Select and perform an action
                action = self.agent.select_action(state)
                _, done, reward = self.env.step(action.item())
                tmp = reward
                reward = torch.tensor([reward], device=self.agent.device)

                # Observe next state
                if not done:
                    next_state = self.get_screen(self.env.state)
                else:
                    next_state = None

                # Store the transition in memory
                self.agent.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization on the target network
                loss = self.agent.optimize_model()
                print(f'step {t}, get reward {tmp} with loss {loss}')
                if done:
                    self.episode_durations.append(t+1)
                    self.plot_durations()
                    break

            # Update the target network, copying all weights and biases in DCQN
            if episode % self.target_update == 0:
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict())

        print('Completed')
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    agent = GuiAgent()
    agent.render()
