import datetime
from itertools import count
import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template
from Agentes.Master import Master
from Agentes.Slave import Slave
from Agentes.DroneDQN import DroneDQN
import torch


class Config():
    def __init__(self,jid,password,name,lidar1,lidar2='',gps='',jidSlave="",vel=1,mov=1,vervose=False,num_episodes=0):
        self.jid = jid
        self.password = password
        self.jidSlave = jidSlave
        self.name = name
        self.lidar1 = lidar1
        self.lidar2 = lidar2
        self.gps = gps
        self.vel = vel
        self.mov = mov
        self.vervose = vervose
        self.num_episodes = num_episodes


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
            self.drone = Master(self.config)
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
                # self.drone.moveTo(0, -20, -20, self.config.vel)
                self.drone.moveArriba(20, self.config.vel)
            elif self.counter == 1:
                # self.drone.moveTo(70, -20, -20, self.config.vel)
                self.drone.moveIzquierda(20, self.config.vel)
            elif self.counter == 2:
                # self.drone.moveTo(70, 30, -20, self.config.vel)
                self.drone.moveDelante(20, self.config.vel)
            elif self.counter == 3:
                # self.drone.moveTo(0, 30, -20, self.config.vel)
                self.drone.moveDerecha(20, self.config.vel)
            elif self.counter == 4:
                # self.drone.moveTo(0, 0, -20, self.config.vel)
                self.drone.moveAtras(20, self.config.vel)
            elif self.counter == 5:
                self.drone.moveIzquierda(20, self.config.vel)
            elif self.counter == 6:
                self.drone.moveAbajo(20, self.config.vel)
                # self.counter = 1

            print(self.config.name, self.drone.getPosition())

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
            self.drone = Slave(config=self.config)
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
            print("DQNAgent on_start", self.config.name)
            self.drone = DroneDQN(config=self.config)

            # self.drone.reset_env()

            self.drone.start()

            self.num_episodes = self.config.num_episodes
            print('Empiezan los episodios')

        async def run(self):
            print("DQNAgent run")
            # env.reset()
            # self.drone.run()

            self.drone.reset_env()
            self.drone.iniciar_drone(self.config)
            self.drone.takeoff()
            self.drone.last_screen = self.drone.getLidar()
            self.drone.current_screen = self.drone.getLidar()
            self.drone.state = torch.tensor(np.subtract(self.drone.current_screen, self.drone.last_screen),
                                            device=self.drone.device, dtype=torch.float)
            # xx = self.drone.current_screen - self.drone.last_screen
            # self.drone.state = torch.tensor(self.drone.state, device="cpu", dtype=torch.float)
            # self.drone.state = np.subtract(self.drone.current_screen, self.drone.last_screen).clone().detach()

            old_next_state = self.drone.state

            for t in count():
                print("-" * 20 + str((self.drone.state.shape)))
                # Select and perform an action
                print("Count:", t)
                self.drone.action = self.drone.select_action(self.drone.state)
                print("Accion:", self.drone.action)
                reward = self.drone.compute_reward(self.drone.action)
                print("Reward:", reward)
                done = self.drone.isDone(reward)
                print("Done:", done)
                # _, reward, done, _ = env.step(action.item())
                self.drone.reward = torch.tensor([[reward]], device=self.drone.device)
                print("Reward tensor:", self.drone.reward)
                # Observe new state
                self.drone.last_screen = self.drone.current_screen
                self.drone.current_screen = self.drone.getLidar()

                if not done:
                    next_state = self.drone.current_screen - self.drone.last_screen  # np.subtract(self.drone.current_screen, self.drone.last_screen)
                else:
                    next_state = old_next_state

                # Store the transition in memory
                '''print("Memory push:", self.drone.state.size(), self.drone.action.size(), next_state.size(),
                      self.drone.reward.size())'''

                self.drone.memory.push(self.drone.state, self.drone.action, next_state, self.drone.reward)
                # print("Memory size:", self.drone.memory.__len__())
                # Move to the next state
                self.drone.state = next_state

                # Perform one step of the optimization (on the target network)
                self.drone.optimize_model()

                print("#" * 20, self.drone.getCollision().has_collided)
                if done or self.drone.getCollision().has_collided:
                    self.drone.episode_durations.append(t + 1)
                    self.drone.plot_durations()
                    break
                old_next_state = next_state

            self.num_episodes -= 1

            if self.num_episodes <= 0:
                print('Complete')
                self.kill(exit_code=10)

        async def on_end(self):
            print("DQNAgent stop")
            self.drone.landing()
            await self.agent.stop()
