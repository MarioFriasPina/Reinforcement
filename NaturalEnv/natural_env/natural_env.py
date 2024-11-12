"""
Natural Environment

This environment is based on the simple_world_comm_v3 environment from the MPE library.

### Description

This environment tries to simulate a natural environment, which contains multiple agents trying to survive in it, each with its own goals.
The environment contains 2 types of agents: predators and prey.
There are 4 types of objects: obstacles, food, water and forests.
The obstacles prevent agents from moving into them.
The agents need to have a certain amount of food and water to survive.
The prey agents can move into forests to hide from predators.

The prey agents must eat food in a greater quantity than predators to survive.

### Rewards

Prey agents:

- -10 for dying, either by predators or by running out of food or water
- [0 : -1] for suffering from hunger or thirst
- +1 for finding food or water

Predators agents:

- -10 for dying from running out of food or water
- [0 : -1] for suffering from hunger or thirst
- +2 for eating prey

### Action Space

[no_action, move_left, move_right, move_down, move_up]

### Observation Space



### Arguments
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=2,
        num_adversaries=4,
        num_obstacles=1,
        num_food=2,
        max_cycles=25,
        num_forests=2,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_food=num_food,
            max_cycles=max_cycles,
            num_forests=num_forests,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(
            num_good, num_adversaries, num_obstacles, num_food, num_forests
        )
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "natural_env_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, num_good, num_adversaries, num_obstacles, num_food, num_forests):