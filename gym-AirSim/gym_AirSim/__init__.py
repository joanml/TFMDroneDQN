import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='AirSim-v0',
    entry_point='gym_AirSim.envs:AirSim_env',
)
