import gym
import airsim
import time
from gym import error, spaces, utils
from gym.utils import seeding

'''
     Actions:
         Type: Discrete(2)
         Num    Action
         0      aterrizaje
         1      takeoff
         2      move left
         3      move right
         


'''

class AirSimEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }

  def __init__(self):
    # connect to the AirSim simulator
    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    print("Taking off")
    self.client.moveByVelocityZAsync(0, 0, -20, 8).join()
    time.sleep(3)

  def step(self, action):
    i=10
    self.client.moveByVelocityZAsync(-1 * i, -1 * i, -20 - i, 15)


  def reset(self):
    pass
  def render(self, mode='human'):
    pass
  def close(self):
    pass