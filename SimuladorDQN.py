import time
import Agentes
from Agentes.Drone import Drone
from Agentes.AgenteSpadeDrone import Config, DQNAgent


if __name__ == "__main__":

    jidDrone = "senderDroneAgent1@gtirouter.dsic.upv.es"
    DroneConfig = Config(jidDrone, "123", "Drone_1", "MyLidar1_1", 2, 2,
                         vervose=False, num_episodes=50)

    dqndrone = DQNAgent(DroneConfig)

    dqndrone.start()

    while dqndrone.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            dqndrone.stop()
            break
