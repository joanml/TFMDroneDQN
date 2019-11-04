from Agentes.Drone import Drone


class Slave(Drone):
    def __init__(self,config):
        self.iniciar_drone(config)
        print(self.nombre, " Ha iniciado como Slave")