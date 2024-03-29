import airsim
import random
import math
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import time
from ImageProcessing.ImageProcessing import ImageProcessing
from ConfigFiles import ConfigAirSimCamera
from datetime import datetime

resize = T.Compose([T.ToPILImage(),
                    T.Resize(64, interpolation=Image.CUBIC),
                    T.ToTensor()])


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


class Replay_Memory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample mini-batches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """

    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #mini-batch() if you want to retrieve samples directly.

        Attributes:
            size (int): The mini-batch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=7, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

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


# -----------------------------------------------------------------------------------------------------------------------


class Drone():

    # Funcionan
    def iniciar_drone(self, config):
        airsim.YawMode.is_rate = False
        self.config = config
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.setInfo(name=self.config.name, L1=self.config.lidar1, L2=self.config.lidar2, GPS=self.config.gps)
        self.vervose = self.config.vervose
        self.image_processing = ImageProcessing()
        self.conta_img = 0
        self.conta_img_1 = 0
        self.position_list = []
        print(self.nombre, " Conectado")

        dateTimeObj = datetime.now()
        self.file1 = open("Logs/Txt/" + str(dateTimeObj).replace(
            ':', '_') + ".txt", "w")

    def reset_env(self):
        self.client.reset()
        time.sleep(1)

    def takeoff(self):
        landed = self.client.getMultirotorState(vehicle_name=self.nombre).landed_state
        print("LandedState: ", airsim.LandedState.Landed)
        if landed == airsim.LandedState.Landed:
            print(self.nombre, " taking off...")
            self.client.takeoffAsync(vehicle_name=self.nombre).join()
            print(self.nombre, " en el aire...")
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

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        print('reshape', int(points.shape[0] / 3))
        if int(points.shape[0] / 3) == 0:
            print(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        return points

    def create_blank(self, width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    def getLidar(self):
        info = self.client.getLidarData(lidar_name=self.nombreLidar1, vehicle_name=self.nombre)
        # print(len(info.point_cloud),info.point_cloud)

        while len(info.point_cloud) < 4:
            time.sleep(0.5)
            info = self.client.getLidarData(lidar_name=self.nombreLidar1, vehicle_name=self.nombre)
            # print(len(info.point_cloud))

        # print('Lidar, info OK')
        points = self.parse_lidarData(info)
        # print('Lidar, point OK')
        x = list()
        y = list()
        for m in points:
            # print(m[1],m[0])
            x.append(m[1])
            y.append(m[0])
        x = np.asarray(x)
        y = np.asarray(y)

        # print('Lidar, 2XY OK')

        fig = Figure(figsize=(5, 5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        ax.plot(x, y)  # , '-o')
        ax.axis('off')
        canvas.draw()

        s, (width, height) = canvas.print_to_buffer()
        # print(width, height)
        from PIL import Image
        im = Image.frombytes("RGBA", (width, height), s)

        # jarain78
        open_cv_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        image = self.create_blank(width, height)
        image = cv2.add(image, open_cv_image)
        imag90, image180, image270 = self.image_processing.rotateImage(image)
        dateTimeObj = datetime.now()
        print(dateTimeObj)
        cv2.imwrite('Logs/LidarImages/lidar_imag_' + str(dateTimeObj).replace(
            ':', '_') + '.jpg', imag90)

        if self.conta_img_1 == 100:
            self.conta_img_1 = 0

        self.conta_img_1 = self.conta_img_1 + 1

        im = imag90

        screen = np.ascontiguousarray(im, dtype=np.float32) / 255
        print('screen', screen.size)
        screen = torch.from_numpy(screen)

        screen = np.swapaxes(screen, 0, 2)
        print('screen', screen.size(), screen.type())

        # return resize(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to('cpu')

    # Jarain78

    def storedg_drone_vectors(self):
        currente_possition = str(self.getPosition())
        currente_vel = str(self.getQuadVel())
        dateTimeObj = datetime.now()

        self.file1.writelines(str(dateTimeObj).replace(
            ':', '_') + " " + currente_possition + " " + currente_vel)

    def close_file(self):
        self.file1.close()

    '''def getPositionImage(self):
        position = self.getPosition()
        self.position_list.append([int(position.x_val), int(position.y_val) + y, int(position.z_val)])

        if len(self.position_list)==10:
            fig = Figure(figsize=(5, 5), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            for i in self.position_list:
                ax.scatter(i)
            ax.axis('off')
            canvas.draw()
        self.position_list = 0'''

    def getCamera(self, save_img=False):
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        rawImage = self.client.simGetImage("2", ConfigAirSimCamera.cameraTypeMap[
            ConfigAirSimCamera.cameraType])

        air_sim_str2arr = airsim.string_to_uint8_array(rawImage)

        image = self.image_processing.get_preview_image(rawImage, air_sim_str2arr)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        canny_image = self.image_processing.canny(image)
        img_1 = self.image_processing.region_of_interest(image)

        # if save_img or self.conta_img > 10:
        dateTimeObj = datetime.now()
        cv2.imwrite("Logs/Images/" + str(dateTimeObj).replace(
            ':', '_') + "_camera.png", image)

        if self.conta_img == 10:
            self.conta_img = 0

        self.conta_img = self.conta_img + 1

        screen = np.ascontiguousarray(image, dtype=np.float32) / 255
        print('screen', screen.size)
        screen = torch.from_numpy(screen)

        screen = np.swapaxes(screen, 0, 2)
        print('screen', screen.size(), screen.type())

        # return resize(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to('cpu')

    def getCollision(self):
        return self.client.simGetCollisionInfo(vehicle_name=self.nombre)

    def getQuadState(self):
        return self.client.getMultirotorState(vehicle_name=self.nombre).kinematics_estimated.position

    def getQuadVel(self):
        return self.client.getMultirotorState(vehicle_name=self.nombre).kinematics_estimated.linear_velocity

    def moveDerecha(self, y, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val), int(posicion.y_val) + y, int(posicion.z_val), vel)

    def moveIzquierda(self, y, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val), int(posicion.y_val) - y, int(posicion.z_val), vel)
        # print("IZQUIERDA", posicion)

    def moveDelante(self, x, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val) + x, int(posicion.y_val), int(posicion.z_val), vel)

    def moveAtras(self, x, vel):
        posicion = self.getPosition()
        self.moveTo(int(posicion.x_val) - x, int(posicion.y_val), int(posicion.z_val), vel)

    def setInfo(self, name, L1, L2, GPS):
        self.nombre = name
        self.nombreLidar1 = L1
        self.nombreLidar2 = L2
        self.nombreGPS = GPS
        self.client.enableApiControl(True, vehicle_name=self.nombre)
        self.client.armDisarm(True, vehicle_name=self.nombre)

    def moveArriba(self, z, vel):
        posicion = self.getPosition()
        if posicion.z_val - z > -50:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), int(posicion.z_val) - z, vel)
        else:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), -50, vel)

    def moveAbajo(self, z, vel):
        posicion = self.getPosition()
        if posicion.z_val + z < 0:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), int(posicion.z_val) + z, vel)
        else:
            self.moveTo(int(posicion.x_val), int(posicion.y_val), -1, vel)

    def getPosition(self):
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.nombre).position

    def moveTo(self, horizontal, lateral, altura, v):
        return self.client.moveToPositionAsync(horizontal, lateral, altura, v, vehicle_name=self.nombre).join()

    # Jarain78

    def show_path(self):
        pass
        #self.simShowPawnPath(True, 3, 8, "Drone_1")



    def moveToPositionNoSync(self, x, y, z, velocity):
        '''result = self.client.moveOnPathAsync(
            [airsim.Vector3r(0, -10, z), airsim.Vector3r(0, -10, z), airsim.Vector3r(0, 0, z),
             airsim.Vector3r(0, 0, z), airsim.Vector3r(0, 0, -20)],
            12, 120,
            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()
        print(result)'''
        self.client.moveToPositionAsync(x, y, z, 1).join()

    def moveToXYZ(self, goal_pos_x=10, goal_pos_y=10, goal_pos_z=10):
        global x
        global y
        global theta

        posicion = self.getPosition()

        x = posicion.x_val
        y = posicion.y_val
        z = posicion.z_val

        inc_x = goal_pos_x - x
        inc_y = goal_pos_y - y

        angle_to_goal = math.atan2(goal_pos_y - inc_y, goal_pos_x - inc_x)

        if abs(angle_to_goal) > 0.1:
            linear_speed_x = 0.0
            linear_speed_y = 0.3
            linear_speed_z = 0.0
        else:
            linear_speed_x = 0.5
            linear_speed_y = 0.0
            linear_speed_z = 0.0

        self.moveTo(inc_x, inc_y, int(posicion.z_val), 0.5)

        return linear_speed_x, linear_speed_y, linear_speed_z, inc_x, inc_y

    def takeoffPositionStart(self, vel):
        self.takeoff()
        self.moveArriba(2, vel)
        print("DQNAgent takeoff")

    def moveDroneOnPath(self):
        # AirSim uses NED coordinates so negative axis is up.
        # z of -7 is 7 meters above the original launch point.
        z = -7

        # see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

        # this method is async and we are not waiting for the result since we are passing timeout_sec=0.
        result = self.client.moveOnPathAsync(
            [airsim.Vector3r(0, -253, z), airsim.Vector3r(125, -253, z), airsim.Vector3r(125, 0, z),
             airsim.Vector3r(0, 0, z), airsim.Vector3r(0, 0, -20)],
            12, 120,
            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()
        self.client.moveToPositionAsync(0, 0, z, 1).join()

    # No usadas
    '''
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
    def moveToAcelerador(self, pitch, roll, throttle, yaw_rate, duration):
        return self.client.moveByAngleThrottleAsync(pitch=pitch, roll=roll, throttle=throttle, yaw_rate=yaw_rate,
                                                    duration=duration, vehicle_name=self.nombre).join()
    def getLidar1(self):
        info = self.client.getLidarData(lidar_name=self.nombreLidar1, vehicle_name=self.nombre)
        print('Lidar, info OK')
        points = self.parse_lidarData(info)
        print('Lidar, point OK')
        #x,y = self.lidar2XY(points)
        return points
    def lidar2XY(lidar):
        x = list()
        y = list()
        for m in lidar:
            # print(m[1],m[0])
            x.append(m[1])
            y.append(m[0])

        x = np.asarray(x)
        y = np.asarray(y)
        print('Lidar, 2XY OK')

        return (x, y)
    def xy2Image(x, y):
        fig = Figure()  # figsize=(5, 5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        ax.plot(x, y, '-o')
        ax.axis('off')
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        # print(width, height)
        from PIL import Image
        im = Image.frombytes("RGBA", (width, height), s)
        # im.show()
        print('Lidar, 2Img OK')
        return im
    def getLidar1Img(self):
        lidar = self.getLidar1()
        print('lidar, getLidar1Img OK')

        (x, y) = self.lidar2XY(lidar)
        print('x, y, getLidar1Img OK')
        im = self.xy2Image(x, y)
        print('im, getLidar1Img OK')
        return im
    '''
