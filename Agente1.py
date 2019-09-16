#!/usr/bin/gym_AirSim python3

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

class TestAgent(Agent):
	def setup(self):
		print("test agent setup")
		print("Behaviour añadido")

def main():
	print("Creating agent...")
	serverAgent = TestAgent("testagent1231236u1523@gtirouter.dsic.upv.es", "pass1234")
	print("Agent created, starting...")
	serverAgent.start()
	print("Agent started, stopping...")
	serverAgent.setup()



	
	serverAgent.stop()
	print("Agent stopped, closing!")

if __name__ == "__main__":
	main()