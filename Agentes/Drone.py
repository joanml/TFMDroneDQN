import os
import pprint
from datetime import datetime

import airsim
import numpy as np
from airsim import ImageRequest
from random import randrange
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class State:
    Stop = 0
    Armed = 1
    TakeOff = 2
    Fly = 3
    Landing = 4
    Disarmed = 5


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)
class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)
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

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Drone():

    # Funcionan
    def iniciar_drone(self, n, L1, L2, GPS, ):
        airsim.YawMode.is_rate = False
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.setInfo(n=n, L1=L1, L2=L2, GPS=GPS)
        print(self.nombre, " Conectado")
    def reset_env(self):
        self.client.reset()

    def setInfo(self, n, L1, L2, GPS):
        self.nombre = n
        self.nombreLidar1 = L1
        self.nombreLidar2 = L2
        self.nombreGPS = GPS
        self.client.enableApiControl(True, vehicle_name=self.nombre)
        self.client.armDisarm(True, vehicle_name=self.nombre)

    def getPosition(self):
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.nombre).position

    def takeoff(self):
        landed = self.client.getMultirotorState(vehicle_name=self.nombre).landed_state
        if landed == airsim.LandedState.Landed:
            print(self.nombre, " taking off...")
            self.client.takeoffAsync(vehicle_name=self.nombre)
        else:
            print(self.nombre, " already flying...")
            self.client.hoverAsync(vehicle_name=self.nombre).join()

    def landing(self):
        landed = self.client.getMultirotorState(vehicle_name=self.nombre).landed_state
        if landed == airsim.LandedState.Flying:
            print(self.nombre, " taking landing...")
            self.client.landAsync(vehicle_name=self.nombre).join()
        else:
            print(self.nombre, " already landing...")

    def moveTo(self, horizontal, lateral, altura, v):
        return self.client.moveToPositionAsync(horizontal, lateral, altura, v, vehicle_name=self.nombre).join()

    def moveToAcelerador(self, pitch, roll, throttle, yaw_rate, duration):
        return self.client.moveByAngleThrottleAsync(pitch=pitch, roll=roll, throttle=throttle, yaw_rate=yaw_rate,
                                                    duration=duration, vehicle_name=self.nombre).join()

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        return points

    @property
    def getLidar1(self):
        info = self.client.getLidarData(lidar_name=self.nombreLidar1, vehicle_name=self.nombre)
        points = self.parse_lidarData(info)
        return points

    # No usadas
    def getGps(self):
        gps_data = self.client.getGpsData(gps_name=self.nombreGPS,
                                          vehicle_name=self.nombre)  # self, gps_name=self.nombreGPS, vehicle_name=self.nombre)
        s = pprint.pformat(gps_data)
        return s

    def getOrientacion(self):
        return self.client.getOrientation(vehicle_name=self.nombre)

    def getAllInfo(self):
        return self.client.getMultirotorState(vehicle_name=self.nombre)

    def moveZ(self, Z, velocidad):
        return self.client.moveToZAsync(z=-Z, velocity=velocidad, vehicle_name=self.nombre).join()

    def moveVel(self, vx, vy, vz, t):
        return self.client.moveByVelocityAsync(vx=-vx, vy=-vy, vz=-vz, duration=t, vehicle_name=self.nombre).join()

    def moveByAngleZ(self, pitch, roll, z, yaw, duration):
        return self.client.moveByAngleZAsync(pitch=pitch, roll=roll, z=z, yaw=yaw, duration=duration,
                                             vehicle_name=self.nombre).join()

    def saveImage(self):
        responses = self.client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)],
                                             vehicle_name=self.nombre)
        response = responses[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        print(img1d)
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        print(img_rgb)
        # img = Image.fromarray(img_rgb, 'RGB')
        # print(img)
        # original image is fliped vertically
        img_rgb = np.flipud(img_rgb)

        # write to png
        dt = datetime.now()
        airsim.write_png(os.path.normpath('./Logs/' +
                                          str(self.nombre) +
                                          str(dt.date()) +
                                          str(dt.hour) +
                                          str(dt.minute) +
                                          str(dt.second) +
                                          str(dt.microsecond) + '.png'), img_rgb)

    def getImage(self):
        responses =  self.client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        print(len(img1d))
        return 0
        png_image = self.client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=self.nombre)

        print(png_image)







    def getLidar2(self):
        lindar = self.client.getLidarData(lidar_name=self.nombreLidar2, vehicle_name=self.nombre)

        dt = datetime.now()
        file = open('./Logs/' +
                    str(self.nombre) +
                    str(self.nombreLidar2) +
                    str(dt.date()) +
                    str(dt.hour) +
                    str(dt.minute) +
                    str(dt.second) +
                    str(dt.microsecond) + '.txt', 'w')
        file.write(str(lindar))
        file.close()
        return lindar

    def getCollision(self):
        return self.client.getCollisionInfo(vehicle_name=self.nombre)

    # pruebas
    def moveAngulo(self, angulo, z=10):
        self.client.moveOnPathAsync(
            [airsim.Vector3r(0, -253, z),
             airsim.Vector3r(125, -253, z),
             airsim.Vector3r(125, 0, z),
             airsim.Vector3r(0, 0, z),
             airsim.Vector3r(0, 0, -20)],
            12,
            120,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0),
            20,
            1).join()
        self.client.moveToPositionAsync(0, 0, z, 1).join()
        # return self.client.moveByAngleThrottleAsync(vehicle_name=self.nombre, duration=(angulo * 3)/360, yaw_rate=angulo,throttle=0,roll=0,pitch=0).join()
        # yaw_rate=angulo, duration=(angulo * 3) / 360, vehicle_name = self.nombre).join()

    def followMe(self, x, y, z, v):
        myPosition = self.getPosition()
        # distancia = np.linalg.norm(a-b)

        self.moveTo(x, y, z, v)

    def moveDerecha(self, y, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val), int(posicion.y_val) + y, int(posicion.z_val), vel)
    def moveIzquierda(self, y, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val), int(posicion.y_val) - y, int(posicion.z_val), vel)
        #print("IZQUIERDA", posicion)
    def moveDelante(self,x,vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val) + x, int(posicion.y_val), int(posicion.z_val), vel)
    def moveAtras(self,x,vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val) - x, int(posicion.y_val), int(posicion.z_val), vel)
    def moveArriba(self,z, vel):
        posicion = self.getPosition()
        if posicion.z_val - z > -50:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), int(posicion.z_val) - z, vel)
        else:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), -50, vel)
    def moveAbajo(self,z,vel):
        posicion = self.getPosition()
        if posicion.z_val + z < 0:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), int(posicion.z_val) + z, vel)
        else:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), -1, vel)


class Master(Drone):
    def __init__(self, jidSlave, n, L1, L2, GPS):
        self.iniciar_drone(n=n, L1=L1, L2=L2, GPS=GPS)
        print(self.nombre, " Ha iniciado como Master")


class Slave(Drone):
    def __init__(self, n, L1, L2, GPS):
        self.iniciar_drone(n=n, L1=L1, L2=L2, GPS=GPS)
        print(self.nombre, " Ha iniciado como Slave")


class DroneDQN(Drone):
    def __init__(self, n, L1,L2,GPS,input_shape, nb_actions,
                 gamma=0.99, eps_start=1,eps_end = 0.1,eps_decay=1000000, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4, target_update_interval=10000,
                 monitor=True, num_actions = 4):

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iniciar_drone(n=n,L1=L1,L2=L2,GPS=GPS)
        print(self.nombre, "Ha iniciado")

        self.BATCH_SIZE = minibatch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update_interval

        lidar = self.getLidar1()
        x, y = self.lidar2XY(lidar)

        self.init_screen  = self.xy2Image(x, y)
        _, _, screen_height, screen_width = self.init_screen.shape

        self.n_actions = num_actions
        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.episode_durations = []


    def select_action(self,state):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
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

    def optimize_model(self):
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
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

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

    def run(self):
        num_episodes = 50
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.client.reset()
            last_screen = self.getLidar1()
            current_screen = self.getLidar1()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen = self.getLidar1()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')