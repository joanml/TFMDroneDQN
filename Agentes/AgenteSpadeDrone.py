import time
import datetime

import airsim
from lxml.html.diff import token
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template
from Agentes.Drone import Master, Slave, DroneDQN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import sleep




class Config():
    def __init__(self, jid, password, name, lidar1, lidar2, gps, jidSlave="", vel=2, mov=2):
        self.jid = jid
        self.password = password
        self.jidSlave = jidSlave
        self.name = name
        self.lidar1 = lidar1
        self.lidar2 = lidar2
        self.gps = gps
        self.vel = vel
        self.mov = mov


class PeriodicSenderAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, jid=config.jid, password=config.password, verify_security=False)
        self.config = config

    async def setup(self):
        print(f"PeriodicSenderAgent started at {datetime.datetime.now().time()}")
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        b = self.InformBehav(period=2, start_at=start_at, config=self.config)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b)

    class InformBehav(PeriodicBehaviour):
        def __init__(self, period, start_at, config):
            PeriodicBehaviour.__init__(self, period=period, start_at=start_at)
            self.config = config

        async def on_start(self):
            self.drone = Master(jidSlave=self.config.jidSlave,
                                n=self.config.name,
                                L1=self.config.lidar1,
                                L2=self.config.lidar2,
                                GPS=self.config.gps)
            self.drone.takeoff()
            initial_possition = self.drone.getPosition()
            print("Initial Possition: " + str(initial_possition) + " GPS: ")  # + str(self.drone.getGps()))
            self.counter = 0

        async def run(self):
            print(f"PeriodicSenderBehaviour running at {datetime.datetime.now().time()}: {self.counter}")
            msg = Message(to=self.config.jid)  # Instantiate the message

            drone_pos = self.drone.getPosition()
            msg.body = str(int(drone_pos.x_val)) + ';' + str(int(drone_pos.y_val)) + ';' + str(int(drone_pos.z_val))

            await self.send(msg)
            print("Message sent!")
            # self.drone.moveToAcelerador(pitch=0,roll=0,throttle=100,yaw_rate=90,duration=1)
            # self.drone.moveAngulo(90)

            if self.counter == 0:
                #self.drone.moveTo(0, -20, -20, self.config.vel)
                self.drone.moveArriba(20, self.config.vel)
            elif self.counter == 1:
                #self.drone.moveTo(70, -20, -20, self.config.vel)
                self.drone.moveIzquierda(20, self.config.vel)
            elif self.counter == 2:
                #self.drone.moveTo(70, 30, -20, self.config.vel)
                self.drone.moveDelante(20, self.config.vel)
            elif self.counter == 3:
                #self.drone.moveTo(0, 30, -20, self.config.vel)
                self.drone.moveDerecha(20, self.config.vel)
            elif self.counter == 4:
                #self.drone.moveTo(0, 0, -20, self.config.vel)
                self.drone.moveAtras(20, self.config.vel)
            elif self.counter == 5:
                self.drone.moveIzquierda(20,self.config.vel)
            elif self.counter == 6:
                self.drone.moveAbajo(20,self.config.vel)
                #self.counter = 1

            print(self.config.name,self.drone.getPosition())

            if self.counter == 7:
                self.kill()
            self.counter += 1

        async def on_end(self):
            # stop agent from behaviour
            await self.agent.stop()


class ReceiverAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, jid=config.jid, password=config.password, verify_security=False)
        self.config = config

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav(config=self.config)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b)

    class RecvBehav(CyclicBehaviour):
        def __init__(self, config):
            CyclicBehaviour.__init__(self)
            self.config = config

        async def on_start(self):
            self.drone = Slave(n=self.config.name,
                               L1=self.config.lidar1,
                               L2=self.config.lidar2,
                               GPS=self.config.gps)
            self.drone.takeoff()
            initial_possition = self.drone.getPosition()
            print("Initial Possition: " + str(initial_possition) + " GPS: ")  # + str(self.drone.getGps()))

        async def run(self):
            print("RecvBehav running")
            msg = await self.receive(timeout=30)  # wait for a message for 10 seconds
            if msg:
                print("Message received with content: {}".format(msg.body))
                coordenadas = [float(i) for i in str(msg.body).split(';')]
                print(coordenadas)
                self.drone.moveTo(coordenadas[0], coordenadas[1], coordenadas[2], self.config.vel)
            else:
                print("Did not received any message after 10 seconds")
                self.kill()

        async def on_end(self):
            await self.agent.stop()


class DQNAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, jid=config.jid, password=config.password, verify_security=False)
        self.config = config

    async def setup(self):
        print("DQNAgent started")
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        b = self.DQNBehav(period=2, start_at=start_at, config=self.config)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b)

    class DQNBehav(PeriodicBehaviour):
        def __init__(self, period, start_at, config):
            PeriodicBehaviour.__init__(self, period=period, start_at=start_at)
            self.config = config

        async def on_start(self):
            print("DQNAgent on_start",self.config.name)

            self.drone = DroneDQN(n=self.config.name,
                               L1=self.config.lidar1,
                               L2=self.config.lidar2,
                               GPS=self.config.gps)
            self.drone.takeoff()
            self.drone.moveArriba(5,self.config.vel)
            print("DQNAgent takeoff")

        async def run(self):
            print("DQNAgent run")
            ##response = self.drone.getLidar1()

            lidar = self.drone.getLidar1
            #camera = self.drone.getImage()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for m in lidar:
                print(m[0],m[1],m[2])
                ax.scatter(m[0],m[1],m[2], marker='o')
            plt.show()

            #plt.imshow(lidar)
            #plt.imshow(camera)
            #plt.gray()
            #plt.show()

            #sleep(0.5)

            self.drone.moveDelante(self.config.mov, self.config.vel)
            #plt.close()



        async def on_end(self):
            print("DQNAgent stop")
            self.drone.landing()
            await self.agent.stop()