gym-AirSim
The AirSim environment is a multiagent domain featuring continuous state and action spaces. Currently, several tasks are supported:

AirSim
The AirSim task initializes a single offensive agent on the field and rewards +1 for scoring a goal and 0 otherwise. In order to score a goal, the agent will need to know how to approach the ball and kick towards the goal. The sparse nature of the goal reward makes this task very difficult to accomplish.

EmptyGoal
The AirSimEmptyGoal task features a more informative reward signal than the AirSim task. As before, the objective is to score a goal. However, AirSimEmtpyGoal rewards the agent for approaching the ball and moving the ball towards the goal. These frequent rewards make the task much more accessible.

AirSimAgainstKeeper
The objective of the AirSimAgainstKeeper task is to score against a goal keeper. The agent is rewarded for moving the ball towards the goal and for scoring a goal. The goal keeper uses a hand-coded policy developed by the Helios RoboCup team. The difficulty in this task is learning how to shoot around the goal keeper.

Installation
cd gym-AirSim
pip install -e .