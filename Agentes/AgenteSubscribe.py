import time
import datetime
from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour, CyclicBehaviour
from spade.message import Message
from Agentes.Drone import Drone, Master, Slave


class UnoQuePasa(Agent,Master):
    def imprimir(self):

        print(self.client, self.jidSlave)

class SenderMasterAgent(Agent):

    def __init__(self, jid, password, jidSlave,n,L1,L2, GPS):
        self.jid = jid
        self.jidSlave = jidSlave
        self.password = password
        self.n=n
        self.L1 = L1
        self.L2 = L2
        self.GPS = GPS
        Agent.__init__(self,jid=jid, password=password)


    async def setup(self):
        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        b = self.Behav1(period=2, start_at=start_at, jidSlave=self.jidSlave,n=self.n,L1=self.L1,L2=self.L2,GPS=self.GPS)
        self.add_behaviour(b)

    class Behav1(PeriodicBehaviour, Master):
        def __init__(self, period, start_at, jidSlave, n,L1,L2,GPS):
            self.jidSlave = jidSlave
            Master.__init__(self, jidSlave=jidSlave,n=n,L1=L1,L2=L2,GPS=GPS)
            PeriodicBehaviour.__init__(self,period=period,start_at=start_at)

        async def on_start(self):
            self.counter = 0
            print("Master inicia el contador a ", self.counter)

        async def run(self):
            print("Master inicia RUN contador:", self.counter, " la posicion de master es ", self.getPosition())
            msg = Message(to=self.jidSlave, body= str(self.counter))  # Instantiate the message

            await self.send(msg)
            print("Position send!")

            if self.counter == 30:
                self.kill()
            self.counter += 1

        async def on_end(self):
            await self.agent.stop()

class ReciberSlaveAgent(Agent):
    def __init__(self, jid, password, n, L1, L2, GPS):
        self.jid = jid
        self.password = password
        self.name = n
        self.L1 = L1
        self.L2 = L2
        self.GPS = GPS


    async def setup(self):
        b = self.Behav2(n=self.name,L1=self.L1,L2=self.L2,GPS=self.GPS)
        self.add_behaviour(b)

    class Behav2(CyclicBehaviour,Slave):
        def __init__(self,n,L1,L2,GPS):
            Slave.__init__(self, n=n, L1=L1, L2=L2, GPS=GPS)

        async def run(self):
            msg = await self.receive(timeout=30)
            print("Position recived: ",msg)

        async def on_end(self):
            await self.agent.stop()

if __name__ == "__main__":
    senderMasterAgent = SenderMasterAgent("SenderMasterAgent11@gtirouter.dsic.upv.es", "receiver_password")
    future = senderMasterAgent.start()
    future.result() # wait for receiver agent to be prepared.
    reciberSlaveAgent = ReciberSlaveAgent("ReciberSlaveAgent11@gtirouter.dsic.upv.es", "sender_password")
    reciberSlaveAgent.start()

    while reciberSlaveAgent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            senderMasterAgent.stop()
            reciberSlaveAgent.stop()
            break
    print("Agents finished")