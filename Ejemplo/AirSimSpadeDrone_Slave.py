import time
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour


class ReceiverAgent(Agent):
    class RecvBehav(CyclicBehaviour):
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

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav()
        self.add_behaviour(b)


if __name__ == "__main__":

    receiveragent = ReceiverAgent("Drone_2@gtirouter.dsic.upv.es", "test_2")
    future = receiveragent.start()
    future.result()  # wait for receiver agent to be prepared.


    while receiveragent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:

            receiveragent.stop()
            break
    print("Agents finished")
