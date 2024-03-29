import datetime
import time
import os
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
from datetime import datetime

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
                 gamma=0.999,
                 eps_start=.9,
                 eps_end=0.05,
                 eps_decay=50,
                 learning_rate=0.00025,
                 momentum=0.95,
                 batch_size=4,
                 memory_size=500000,
                 train_after=10000,
                 train_interval=10,
                 target_update_interval=10,
                 num_actions=2):

        self.config = config
        # if gpu is to be used
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iniciar_drone(config=self.config)
        print(self.nombre, "Ha iniciado")

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update_interval
        # self.input_shape = input_shape
        self.n_actions = num_actions
        self.vervose = self.config.vervose

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        # self._explorer = explorer

        # Jarain78
        init_xpos = self.getPosition().x_val
        init_ypos = self.getPosition().y_val
        init_zpos = self.getPosition().z_val

        # print(init_xpos, init_ypos, init_zpos)
        # self.drone.moveTo(init_xpos + 20, init_ypos, -2, 1.5)

        self.quad_state = self.getQuadState()
        self.goal_x = init_xpos + 20
        self.goal_y = init_ypos
        self.init_distance = self.calculateDistance(self.goal_y,
                                                    self.goal_x, init_ypos, init_xpos)

        self.min_set_reward = -1000
        self.max_set_reward = 1000
        self.old_distance = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        # print('sample', sample, 'eps_threshold', eps_threshold, 'steps_done', steps_done, 'sample > eps_threshold',
        #      sample > eps_threshold)

        if steps_done > self.BATCH_SIZE and sample > eps_threshold:
            print("Accion Entrenamiento")
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("Policy Net: ", self.policy_net(state))
                # print("Select Action: ", self.policy_net(state).max(1)[1])
                # print("Select Action: ", self.policy_net(state).max(1)[1].view(1, 1))

                state1 = state.clone().detach().requires_grad_(True)
                return self.policy_net(state1).max(1)[1].view(1, 1)


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
        plt.grid()
        plt.pause(0.001)  # pause a bit so that plots are updated
        # name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
        plt.show()
        plt.savefig(
            "E:\\Dropbox\\Cognitive_Service_Project\\TFMDroneDQN-master\\Logs\Plots\\" + str(datetime.now()).replace(
                ':', '_') + ".png", dpi=200)

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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device="cpu", dtype=torch.uint8)

        aa = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(aa)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #print("state_batch: ", state_batch)
        print("action_batch: ", action_batch)
        print("reward_batch: ", reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        next_state_values = torch.zeros(self.BATCH_SIZE, device="cpu")
        # print(next_state_values)
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
        print("*" * 20, action)
        if action == 0:
            print("Adelante")
            self.moveDelante(self.config.mov, self.config.vel)
        elif action == 1:
            print("Izquierda")
            self.moveIzquierda(self.config.mov, self.config.vel)
        elif action == 2:
            print("Derecha")
            self.moveDerecha(self.config.mov, self.config.vel)
        elif action == 3:
            print("Atras")
            self.moveAtras(self.config.mov, self.config.vel)
        elif action == 4:
            print("Arriba")
            self.moveArriba(self.config.mov, self.config.vel)
        elif action == 5:
            print("Abajo")
            # self.moveAbajo(self.config.mov, self.config.vel)
            pass

    # jarain78
    def interpret_action_1(self, action):
        scaling_factor = 1.5

        if action == 0:
            print("RIGNT")
            quad_offset = (0, scaling_factor, 0)
        elif action == 1:
            print("LEFT")
            quad_offset = (0, -scaling_factor, 0)
        elif action == 2:
            print("FORWARE")
            quad_offset = (scaling_factor, 0, 0)
        elif action == 3:
            print("BACKWARE")
            quad_offset = (-scaling_factor, 0, 0)
        elif action == 4:
            print("UP")
            quad_offset = (0, 0, -scaling_factor)
        elif action == 5:
            print("DOWN")
            quad_offset = (0, 0, scaling_factor)

        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(quad_vel.x_val + quad_offset[0], quad_vel.y_val + quad_offset[1],
                                        quad_vel.z_val + quad_offset[2], 5).join()

        # self.client.moveToPositionAsync(quad_offset[0], quad_offset[1], quad_offset[2], 0.8,
        #                                vehicle_name=self.nombre).join()

    def calculateDistance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

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
            reward = -1000
        else:
            # reward = np.int16(np.negative(self.quad_state.y_val) + self.quad_state.x_val).item()
            x_objetivo = self.goal_x
            y_objetivo = self.goal_y
            maxreword = 1000
            distance = self.calculateDistance(self.quad_state.x_val, self.quad_state.y_val, x_objetivo, y_objetivo)

            if distance > maxreword:
                reward = 0
            else:
                reward = maxreword - distance
        print("=" * 20, reward)

        return int(reward)

    def isDone(self, reward):
        done = False
        if reward <= -1000:
            done = True
        return done

    def start(self):
        self.init_screen = self.getLidar()
        '''print('init_screen', self.init_screen.size(), self.init_screen.type())'''
        _, _, screen_height, screen_width = self.init_screen.shape

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)

        # self.policy_net = My_DQN(500, 500, self.n_actions).to(self.device)
        # self.target_net = My_DQN(500, 500, self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.01, momentum=0.2)

        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

    def save(self, id):
        filename = 'tmp/models/model_{}.pth.tar'.format(id)
        dirpath = './modelos'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        checkpoint = {
            'net': self.agent.net.state_dict(),
            'target': self.agent.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_step': self.total_step
        }
        torch.save(checkpoint, filename)

    def load(self, filename, device='cpu'):
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        ## Deal with the missing of bn.num_batches_tracked
        net_new = DQN()
        tar_new = DQN()

        for k, v in ckpt['net'].items():
            for _k, _v in self.agent.net.state_dict().items():
                if k == _k:
                    net_new[k] = v

        for k, v in ckpt['target'].items():
            for _k, _v in self.agent.target.state_dict().items():
                if k == _k:
                    tar_new[k] = v

        self.agent.net.load_state_dict(net_new)
        self.agent.target.load_state_dict(tar_new)
        ## -----------------------------------------------

        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_step = ckpt['total_step']