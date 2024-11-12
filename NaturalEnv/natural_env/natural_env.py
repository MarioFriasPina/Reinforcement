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
        world = World()

        # Crear presas (agentes buenos)
        world.good_agents = [Agent() for _ in range(num_good)]
        for i, agent in enumerate(world.good_agents):
            agent.name = f"prey_{i}"
            agent.color = np.array([0.0, 0.8, 0.0])  # Verde
            agent.hunger = 100
            agent.thirst = 100
            agent.is_prey = True
            agent.found_food = False
            agent.found_water = False

        # Crear depredadores (adversarios)
        world.adversaries = [Agent() for _ in range(num_adversaries)]
        for i, adversary in enumerate(world.adversaries):
            adversary.name = f"predator_{i}"
            adversary.color = np.array([0.8, 0.0, 0.0])  # Rojo
            adversary.hunger = 100
            adversary.thirst = 100
            adversary.is_prey = False
            adversary.caught_prey = False

        # Crear obstáculos
        world.obstacles = [Landmark() for _ in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle_{i}"
            obstacle.color = np.array([0.5, 0.5, 0.5])  # Gris

        # Crear fuentes de comida
        world.food_sources = [Landmark() for _ in range(num_food)]
        for i, food in enumerate(world.food_sources):
            food.name = f"food_{i}"
            food.color = np.array([0.9, 0.9, 0.0])  # Amarillo
            food.is_food = True

        # Crear fuentes de agua
        world.water_sources = [Landmark() for _ in range(num_food)]
        for i, water in enumerate(world.water_sources):
            water.name = f"water_{i}"
            water.color = np.array([0.0, 0.0, 1.0])  # Azul
            water.is_water = True

        # Crear bosques
        world.forests = [Landmark() for _ in range(num_forests)]
        for i, forest in enumerate(world.forests):
            forest.name = f"forest_{i}"
            forest.color = np.array([0.2, 0.8, 0.2])  # Verde oscuro
            forest.is_forest = True

        # Añadir todos los objetos al mundo
        world.landmarks = world.obstacles + world.food_sources + world.water_sources + world.forests
        world.agents = world.good_agents + world.adversaries

        return world

    def reward(self, agent):
        # Recompensas para presas
        if agent.is_prey:
            if agent.hunger <= 0 or agent.thirst <= 0:
                return -10  # Muere por hambre o sed (barra de vida)
            elif agent.found_food or agent.found_water:
                return +1  # Encuentra comida o agua
            elif agent.hunger < 20 or agent.thirst < 20:
                return -1  # Penalización por hambre o sed
        else:
            # Recompensas para depredadores
            if agent.hunger <= 0 or agent.thirst <= 0:
                return -10  # Muere por hambre o sed
            elif agent.caught_prey:
                return +2  # Atrapa a una presa
            elif agent.hunger < 20 or agent.thirst < 20:
                return -1  # Penalización por hambre o sed
        return 0

    def observation(self, agent, world):
        # Observaciones simples: posición del agente y distancias a los objetos más cercanos
        return np.array([agent.hunger, agent.thirst])