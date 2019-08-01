import time
import asyncio
import airsim
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template


class SenderAgentOne(Agent):
    class InformBehav(OneShotBehaviour):
        async def on_start(self):
            print("InformBehav running")
            msg = Message(to="drone@gtirouter.dsic.upv.es")  # Instantiate the message
            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg.body = "Takeoff"  # Set the message content

            await self.send(msg)
            print("Message sent!")

        async def run(self):
            print("Sender sleep 10 seg")
            time.sleep(10)

            msg1 = Message(to="drone@gtirouter.dsic.upv.es")  # Instantiate the message
            msg1.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg1.body = "Move1"  # Set the message content
            await self.send(msg1)
            print("Message sent!")

            # stop agent from behaviour
            await self.agent.stop()

    async def setup(self):
        print("SenderAgent started")
        b = self.InformBehav()
        self.add_behaviour(b)


class ReceiverAgentCyclic(Agent):
    class RecvBehav(CyclicBehaviour):

        async def run(self):
            print("RecvBehav running")

            msg = await self.receive(timeout=30)  # wait for a message for 10 seconds
            if msg:
                print("Message received with content: {}".format(msg.body))
                if msg.body is "Takeoff":
                    self.client.armDisarm(True)
                    landed = self.client.getMultirotorState().landed_state
                    if landed == airsim.LandedState.Landed:
                        print("taking off...")
                        self.client.takeoffAsync().join()
                    else:
                        print("already flying...")
                        self.client.hoverAsync().join()
                elif msg.body is "Move1":
                    landed = self.client.getMultirotorState().landed_state
                    if landed == airsim.LandedState.Flying:
                        self.client.armDisarm(True)
                        self.client.moveToZ(500, 100)
                    else:
                        print("already flying...")
                        self.client.hoverAsync().join()
            else:
                print("Did not received any message after 10 seconds")

            # stop agent from behaviour
            if self.client.getMultirotorState().landed_state is airsim.LandedState.Landed:
                await self.agent.stop()

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav()
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b, template)

    # class MyBehav(CyclicBehaviour):
    #     async def on_start(self):
    #         print("Agente ",' START')
    #         self.counter = 0
    #         self.send(msg='Primer msg')
    #
    #     async def run(self):
    #         print("Agente ",' RUN')
    #         self.counter += 1
    #         if self.counter >= 3:
    #             self.kill(exit_code=10)
    #             return
    #         await asyncio.sleep(1)
    #
    #     async def on_end(self):
    #         print("Agente ",' STOP')


class SenderAgentCyclic(Agent):
    class InformBehav(CyclicBehaviour):
        async def on_start(self):
            print("InformBehav start")
            ###

        async def run(self):
            print("InformBehav running")
            ###

        async def on_end(self):
            print("InformBehav stop")
            ###

    async def setup(self):
        print("SenderAgent started")
        b = self.InformBehav()
        self.add_behaviour(b)


class SenderAgent(Agent):
    class InformBehav(OneShotBehaviour):
        def my_on_subscribe_callback(self,peer_jid):
            if True:
                self.approve(peer_jid)

        async def run(self):
            print("InformBehav running")
            msg = Message(to="receiver1234@gtirouter.dsic.upv.es")     # Instantiate the message
            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg.body = "Hello World"                    # Set the message content

            await self.send(msg)
            print("Message sent!")

            # stop agent from behaviour
            await self.agent.stop()

    async def setup(self):
        print("SenderAgent started")
        b = self.InformBehav()
        self.add_behaviour(b)

class ReceiverAgent(Agent):
    class RecvBehav(OneShotBehaviour):
        async def on_start(self):
            self.presence.subscribe("sender@gtirouter.dsic.upv.es")

        async def run(self):
            print("RecvBehav running")

            msg = await self.receive(timeout=30) # wait for a message for 10 seconds
            if msg:
                print("Message received with content: {}".format(msg.body))
            else:
                print("Did not received any message after 10 seconds")

            # stop agent from behaviour
            await self.agent.stop()

    async def setup(self):
        print("ReceiverAgent started")
        b = self.RecvBehav()
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b, template)

if __name__ == "__main__":
    receiveragent = ReceiverAgent("receiver1234@gtirouter.dsic.upv.es", "123")
    future = receiveragent.start()
    future.result() # wait for receiver agent to be prepared.
    senderagent = SenderAgent("sender@gtirouter.dsic.upv.es", "123")
    senderagent.start()

    while receiveragent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            senderagent.stop()
            receiveragent.stop()
            break
    print("Agents finished")