import time
import getpass

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade import quit_spade
from spade.message import Message


class Agent1(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(self.Behav1())

    class Behav1(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)

        async def run(self):
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available

            self.presence.set_available()
            self.presence.subscribe(self.agent.jid2)
            msg = Message(body='My GPS')  # Instantiate the message
            contacts = self.agent.presence.get_contacts()
            await self.send(msg)





class Agent2(Agent):
    async def setup(self):
        print("Agent {} running".format(self.name))
        self.add_behaviour(self.Behav2())

    class Behav2(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)
            self.presence.subscribe(jid)

        async def run(self):
            self.presence.set_available()
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available
            msg = self.presence.re
            print(msg)



if __name__ == "__main__":

    # jid1 = input("Agent1 JID> ")
    jid1 = "Agente91Cyc@gtirouter.dsic.upv.es"
    # passwd1 = getpass.getpass()
    passwd1 = '123'

    # jid2 = input("Agent2 JID> ")
    jid2 = "Agente92Cyc@gtirouter.dsic.upv.es"
    # passwd2 = getpass.getpass()
    passwd2 = '321'

    agent2 = Agent2(jid2, passwd2)
    agent1 = Agent1(jid1, passwd1)
    agent1.jid2 = jid2
    agent2.jid1 = jid1
    agent2.start()
    agent1.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    agent1.stop()
    agent2.stop()
    quit_spade()