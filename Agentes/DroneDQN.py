import datetime
import time

from Agentes.Drone import Drone, LinearEpsilonAnnealingExplorer, DQN
import math
import random
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

steps_done = 0
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DroneDQN(Drone):
    def __init__(self, config,
                 gamma=0.99,
                 eps_start=.9,
                 eps_end=0.05,
                 eps_decay=200,
                 learning_rate=0.00025,
                 momentum=0.95,
                 batch_size=16,
                 memory_size=500000,
                 train_after=10000,
                 train_interval=4,
                 target_update_interval=10000,
                 num_actions=4):
        self.config = config
        # if gpu is to be used
        self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iniciar_drone(config=self.config)
        print(self.nombre, "Ha iniciado")

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update_interval
        #self.input_shape = input_shape
        self.n_actions = num_actions
        self.vervose = self.config.vervose

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        #self._explorer = explorer

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        print('sample',sample,'eps_threshold',eps_threshold,'steps_done',steps_done,'sample > eps_threshold',sample > eps_threshold)
        if steps_done > self.BATCH_SIZE and sample > eps_threshold:
            print("Accion Entrenamiento")
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print('state.size',state.size)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            print("Accion Aleatoria")
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
        plt.savefig("./Plots/fig"+name)

    def optimize_model(self):
        print('Optimize_model')
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        print('non_final_mask',non_final_mask.size(),'non_final_next_states',non_final_next_states.size(),'state_batch',state_batch.size(),'action_batch',action_batch.size(),'reward_batch',reward_batch.size())

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        print('state_action_values',state_action_values)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def interpret_action(self, action):
        if action == 0:
            self.moveDerecha(self.config.mov, self.config.vel)
        elif action == 1:
            self.moveIzquierda(self.config.mov, self.config.vel)
        elif action == 2:
            self.moveDelante(self.config.mov, self.config.vel)
        elif action == 3:
            self.moveAtras(self.config.mov, self.config.vel)

    def compute_reward(self, action):

        self.interpret_action(action)

        self.collision_info = self.getCollision()
        self.quad_state = self.getQuadState()
        self.quad_vel = self.getQuadVel()
        # thresh_dist = 7
        # beta = 1
        #
        # z = -10
        # pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]),
        #        np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]),
        #        np.array([541.3474, 143.6714, -32.07256])]
        #
        # quad_pt = np.array(list((self.quad_state.x_val, self.quad_state.y_val, self.quad_state.z_val)))

        if self.collision_info.has_collided:
            reward = -100
        else:
            reward = np.int16(np.negative(self.quad_state.y_val) + self.quad_state.x_val).item()
        return reward
        #     dist = 10000000
        #     # for i in range(0, len(pts) - 1):
        #     #    dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(
        #     #        pts[i] - pts[i + 1]))
        #     dist = min()
        #     # print(dist)
        #     if dist > thresh_dist:
        #         reward = -10
        #     else:
        #         reward_dist = (math.exp(-beta * dist) - 0.5)
        #         reward_speed = (np.linalg.norm([self.quad_vel.x_val, self.quad_vel.y_val, self.quad_vel.z_val]) - 0.5)
        #         reward = reward_dist + reward_speed
        #
        # return reward

    def isDone(self,reward):
        done = False
        if reward <= -10:
            done = True
        return done

    def start(self):
        self.init_screen = self.getLidar()
        print('init_screen', self.init_screen.size(), self.init_screen.type())
        _, _, screen_height, screen_width = self.init_screen.shape
        self.policy_net = DQN(self.n_actions,self.BATCH_SIZE).to(self.device)
        self.target_net = DQN(self.n_actions,self.BATCH_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

    # def run(self):
    #     print("RUN DroneDQN reset")
    #     self.reset_env()
    #     print("RUN DroneDQN reset out")
    #     time.sleep(1)
    #     self.takeoffPositionStart(self.config.vel)
    #
    #     self.last_screen = self.getLidar()
    #     self.current_screen = self.getLidar()
    #     state = torch.tensor(np.subtract(self.current_screen, self.last_screen), device=self.device, dtype=torch.float)
    #     for t in count():
    #         # Select and perform an action
    #         action = self.select_action(state)
    #         reward = self.compute_reward(action)
    #         done = self.isDone(reward)
    #         # _, reward, done, _ = env.step(action.item())
    #         reward = torch.tensor([reward], device=self.device)
    #
    #         # Observe new state
    #         self.last_screen = self.current_screen
    #         self.current_screen = self.getLidar()
    #         if not done:
    #             next_state = self.current_screen - self.last_screen
    #         else:
    #             next_state = None
    #
    #         # Store the transition in memory
    #         self.memory.push(state, action, next_state, reward)
    #
    #         # Move to the next state
    #         state = next_state
    #
    #         # Perform one step of the optimization (on the target network)
    #         self.optimize_model()
    #         if done:
    #             self.episode_durations.append(t + 1)
    #             self.plot_durations()
    #             break