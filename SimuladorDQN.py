import time
import Agentes
from Agentes.Drone import Drone
from Agentes.AgenteSpadeDrone import Config, DQNAgent


if __name__ == "__main__":

    jidDrone = "senderDroneAgent1@gtirouter.dsic.upv.es"
    DroneConfig = Config(jidDrone, "123", "Drone_1", "MyLidar1_1", "MyLidar1_2", "GPS_1")

    dqndrone = DQNAgent(DroneConfig)

    dqndrone.start()