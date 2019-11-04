from Agentes.Drone import Drone

class Master(Drone):
    def __init__(self, config):
        self.iniciar_drone(config=config)
        print(self.nombre, " Ha iniciado como Master")