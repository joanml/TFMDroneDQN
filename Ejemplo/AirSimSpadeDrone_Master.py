import time
import datetime
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message

from Agentes.Drone import Master


class PeriodicSenderAgent(Agent):
    class InformBehav(PeriodicBehaviour):
        async def run(self):
            print(f"PeriodicSenderBehaviour running at {datetime.datetime.now().time()}: {self.counter}")
            msg = Message(to="Drone_2@gtirouter.dsic.upv.es")  # Instantiate the message

            drone_pos = self.master_drone.getPosition()
            msg.body = str(drone_pos)

            await self.send(msg)
            print("Message sent!")

            if self.counter == 5:
                self.kill()
            self.counter += 1

        async def on_end(self):
            # stop agent from behaviour
            await self.agent.stop()

        async def on_start(self):
            self.master_drone = Master(jidSlave=senderagent, n="Drone1", L1="MyLidar1_1", L2="MyLidar1_2", GPS='GPS_1')
            self.master_drone.takeoff()
            initial_possition = self.master_drone.getPosition()
            print("Initial Possition: " + str(initial_possition) + " GPS: ")  # + str(self.master_drone.getGps()))
            self.counter = 0

    async def setup(self):
        print(f"PeriodicSenderAgent started at {datetime.datetime.now().time()}")
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        b = self.InformBehav(period=2, start_at=start_at)
        self.add_behaviour(b)


if __name__ == "__main__":

    senderagent = PeriodicSenderAgent("Drone_1@gtirouter.dsic.upv.es", "test_1")
    senderagent.start()

    while senderagent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            senderagent.stop()

            break
    print("Agents finished")
