
import time
import datetime
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template
from Agentes.Drone import Master, Slave

class Config():
    def __init__(self, jid, password, name, lidar1, lidar2, gps,jidSlave=""):
        self.jid = jid
        self.password = password
        self.jidSlave = jidSlave
        self.name = name
        self.lidar1 = lidar1
        self.lidar2 = lidar2
        self.gps = gps

class PeriodicSenderAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self,jid=config.jid,password=config.password,verify_security=False)
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
            msg.body = str(drone_pos)

            await self.send(msg)
            print("Message sent!")

            if self.counter == 10:
                self.kill()
            self.counter += 1

        async def on_end(self):
            # stop agent from behaviour
            await self.agent.stop()

class ReceiverAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self,jid=config.jid,password=config.password,verify_security=False)
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
            else:
                print("Did not received any message after 10 seconds")
                self.kill()

        async def on_end(self):
            await self.agent.stop()

