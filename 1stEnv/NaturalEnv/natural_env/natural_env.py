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

[[vel.x, vel.y], [pos.x, pos.y], [hunger, thirst], [in_forest, consuming], {obstacles}, {food}, {water}, {forests}, {prey_vel}, {prey_pos}, {predator_vel}, {predator_pos}]

### Arguments

``` python
natural_env.env(num_prey=6, num_predators=2, num_obstacles=1,
                num_food=2, num_water=2, num_forests=2, max_cycles=25, continuous_actions=False)
```

`num_prey`:  number of prey agents

`num_predators`:  number of predators

`num_obstacles`:  number of obstacles

`num_food`:  number of food locations that prey agents are rewarded at

`num_water`:  number of water locations that prey and predator agents are rewarded at

`num_forests`:  number of forests that prey agents can hide in to avoid predators

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`:  whether to use continuous actions

`render_mode`:  whether to render the environment

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
        num_prey=6,
        num_predators=2,
        num_obstacles=2,
        num_food=2,
        num_water=2,
        num_forests=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_prey=num_prey,
            num_predators=num_predators,
            num_obstacles=num_obstacles,
            num_food=num_food,
            num_water=num_water,
            num_forests=num_forests,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        scenario = Scenario()
        world = scenario.make_world(
            num_prey, num_predators, num_obstacles, num_food, num_water, num_forests
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
    def make_world(self, num_prey, num_predators, num_obstacles, num_food, num_water, num_forests):
        """
        Creates and populates the world.

        Args:
            num_prey: number of prey agents
            num_predators: number of predator agents
            num_obstacles: number of obstacles
            num_food: number of food locations
            num_water: number of water locations
            num_forests: number of forests
        Returns:
            world: an instance of the World class
        """
        world = World()

        # Create prey
        world.prey = [Agent() for _ in range(num_prey)]
        for i, agent in enumerate(world.prey):
            agent.name = f"prey_{i}"
            agent.color = np.array([0.0, 0.8, 0.0])  # Green

            agent.silent = True # Makes it so they don't communicate

            agent.size = 0.02
            agent.accel = 3.0
            agent.max_speed = 1.0

            agent.hunger = 50 # Maximum amount of hunger
            agent.hunger_rate = 2 # How much hunger is lost per step
            agent.thirst = 50 # Maximum amount of thirst
            agent.thirst_rate = 2 # How much thirst is lost per step

            agent.is_prey = True

        # Create predators
        world.predators = [Agent() for _ in range(num_predators)]
        for i, adversary in enumerate(world.predators):
            adversary.name = f"predator_{i}"
            adversary.color = np.array([0.8, 0.0, 0.0])  # Red

            adversary.silent = True # Makes it so they don't communicate

            adversary.size = 0.04
            adversary.accel = 3.0
            adversary.max_speed = 1.0

            adversary.hunger = 50
            adversary.hunger_rate = 1 # How much hunger is lost per step
            adversary.thirst = 50
            adversary.thirst_rate = 5 # How much thirst is lost per step
            
            adversary.is_prey = False

        # Create obstacles
        world.obstacles = [Landmark() for _ in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle_{i}"
            obstacle.color = np.array([0.5, 0.5, 0.5])  # Gray
            obstacle.size = 0.1

            obstacle.collide = True
            obstacle.boundary = False
            obstacle.movable = False

        # Create food sources for prey
        world.food_sources = [Landmark() for _ in range(num_food)]
        for i, food in enumerate(world.food_sources):
            food.name = f"food_{i}"
            food.color = np.array([0.9, 0.9, 0.0])  # Yellow

            food.collide = False
            food.movable = False
            food.boundary = False
            food.size = 0.05

        # Create water sources
        world.water_sources = [Landmark() for _ in range(num_water)]
        for i, water in enumerate(world.water_sources):
            water.name = f"water_{i}"
            water.color = np.array([0.0, 0.0, 1.0])  # Blue

            water.collide = False
            water.movable = False
            water.boundary = False
            water.size = 0.05

        # Create forests for prey to hide in
        world.forests = [Landmark() for _ in range(num_forests)]
        for i, forest in enumerate(world.forests):
            forest.name = f"forest_{i}"
            forest.color = np.array([0.2, 0.8, 0.2])  # Dark Green

            forest.collide = False
            forest.movable = False
            forest.boundary = False
            forest.size = 0.1

        # Add all entities to the world
        world.landmarks = world.obstacles + world.food_sources + world.water_sources + world.forests
        world.agents = world.prey + world.predators

        # Add boundaries to the world
        #world.landmarks += self.set_boundaries(world)

        return world
    
    def set_boundaries(self, world):
        """
        Creates the boundaries of the world.

        Args:
            world: an instance of the World class

        Returns:
            boundary_list: a list of landmarks that represent the boundaries of the world
        """
        boundary_list = []
        landmark_size = 0.1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)

        # Create boundaries around the world
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                landmark = Landmark()
                landmark.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(landmark)
        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                landmark = Landmark()
                landmark.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(landmark)

        # Set properties of boundaries
        for i, l in enumerate(boundary_list):
            l.name = "boundary %d" % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list
    
    def reset_world(self, world, np_random):
        # Start properties for agents
        for i, agent in enumerate(world.agents):
            agent.hunger = 50
            agent.thirst = 50

        # set random initial states for agents and landmarks
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.last_pos = agent.state.p_pos
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.hunger = 50
            agent.thirst = 50

        # set random initial states for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    
    def benchmark_data(self, agent, world):
        if not agent.is_prey:
            collisions = 0
            for a in self.preys(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0
        
    def is_collision(self, agent1, agent2):
        """
        Check if two agents collide
        """
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def preys(self, world):
        """
        Returns all preys in the world
        """
        return [agent for agent in world.agents if agent.is_prey]

    def predators(self, world):
        """
        Returns all predators in the world
        """
        return [agent for agent in world.agents if not agent.is_prey]

    # def reward(self, agent, world):
    #     """
    #     Returns the main reward
    #     """
    #     main_reward = (
    #         self.predator_reward(agent, world)
    #         if not agent.is_prey
    #         else self.prey_reward(agent, world)
    #     )
    #     return main_reward

    def reward(self, agent, world):
        """Devuelve una recompensa escalar derivada del vector multiobjetivo."""
        reward_vector = self.pareto_reward(agent, world)  # Calcula el vector multiobjetivo
        # Convierte el vector a escalar usando una suma ponderada
        scalar_reward = 0.5 * reward_vector[0] + 0.5 * reward_vector[1]  # Ponderaciones iguales
        agent.reward_vector = reward_vector  # Guarda el vector en el agente para análisis posterior
        return scalar_reward



    def outside_boundary(self, agent):
        """
        Returns True if the agent is outside the boundary
        """
        if (
            agent.state.p_pos[0] > 1
            or agent.state.p_pos[0] < -1
            or agent.state.p_pos[1] > 1
            or agent.state.p_pos[1] < -1
        ):
            return True
        else:
            return False

    def bound(self, x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)
    
    def transform(self, old_value, old_min, old_max, new_min, new_max):
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    def prey_reward(self, agent, world):
        rew = 0

        # 1. Incentivar la supervivencia
        # Pequeña recompensa constante por mantenerse vivo
        rew += 0.5

        # 2. Penalizar levemente por hambre y sed
        # Menor penalización para evitar que se acumulen grandes valores negativos
        hunger_penalty = 0.001 * (100 - agent.hunger)
        thirst_penalty = 0.001 * (100 - agent.thirst)
        rew -= hunger_penalty
        rew -= thirst_penalty

        # 3. Recompensa por encontrar comida y agua
        for food in world.food_sources:
            if self.is_collision(agent, food):
                rew += 10  # Aumentamos la recompensa para que sea un incentivo fuerte
                agent.hunger = min(100, agent.hunger + 30)

        for water in world.water_sources:
            if self.is_collision(agent, water):
                rew += 10  # Aumentamos la recompensa para que sea un incentivo fuerte
                agent.thirst = min(100, agent.thirst + 30)

        # 4. Penalizar por estar cerca de los límites del entorno
        boundary_penalty = 0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            if x > 0.9:  # Solo penalizamos si está muy cerca del límite
                boundary_penalty += 0.5 * (x - 0.9)

        rew -= boundary_penalty

        # 5. Limitar la recompensa negativa acumulada
        # Esto ayuda a estabilizar el entrenamiento
        rew = max(rew, -50)

        return rew

    
    # def predator_reward(self, agent, world):
    #     rew = 0
    #     shape = True
    #     preys = self.preys(world)

    #     # Increase hunger and thirst depending on hunger_rate and thirst_rate
    #     agent.hunger -= agent.hunger_rate
    #     agent.thirst -= agent.thirst_rate

    #     # Penalize for getting too close to preys
    #     if shape:
    #         for p in preys:
    #             rew -= 0.1 * np.sqrt(np.sum(np.square(p.state.p_pos - agent.state.p_pos)))

    #     # Reward for killing preys
    #     if agent.collide:
    #         for p in preys:
    #             if self.is_collision(p, agent):
    #                 rew += 10
    #                 agent.hunger = min(100, agent.hunger + 20)

    #     # Restore thirst
    #     for water in world.water_sources:
    #         if self.is_collision(agent, water):
    #             rew += 0.5
    #             agent.thirst = min(100, agent.thirst + 0.5)
        
    #     # Make it so the agents don't get too far from the boundary
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= 1 * self.bound(x)

    #     # Penalize for starving and thirsting
    #     rew -= 0.05 * min(100 - agent.hunger, 0)
    #     rew -= 0.05 * min(100 - agent.thirst, 0)

    #     # Reward for just being alive
    #     rew += 0.1

    #     rew = np.clip(rew, -10, 10)
    #     return rew

    def pareto_reward(self, agent, world):
        """Calcula la recompensa multiobjetivo para la presa."""
        reward_resources = 0
        reward_survival = 0

        # Penalización por hambre y sed
        agent.hunger -= agent.hunger_rate
        agent.thirst -= agent.thirst_rate

        # Recompensa por encontrar comida
        for food in world.food_sources:
            if self.is_collision(agent, food):
                reward_resources += 10
                agent.hunger = min(100, agent.hunger + 20)

        # Recompensa por encontrar agua
        for water in world.water_sources:
            if self.is_collision(agent, water):
                reward_resources += 10
                agent.thirst = min(100, agent.thirst + 20)

        # Penalización por hambre y sed extrema
        reward_survival -= 0.05 * (max(0, 100 - agent.hunger) + max(0, 100 - agent.thirst))

        # Penalización por salirse de los límites del entorno
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward_survival -= 0.5 * self.bound(x)

        # Recompensa por mantenerse vivo
        reward_survival += 0.1

        # Devolver un vector de recompensas
        return np.array([reward_resources, reward_survival])


    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if other.is_prey:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )

    def observation(self, agent, world):
        # Get the positions of the obstacles
        obs_pos = []
        for obs in world.obstacles:
            if not obs.boundary:
                obs_pos.append(obs.state.p_pos - agent.state.p_pos)

        # Get the positions of the food sources
        food_pos = []
        for food in world.food_sources:
            if not food.boundary:
                food_pos.append(food.state.p_pos - agent.state.p_pos)

        # Get the positions of the water sources
        water_pos = []
        for water in world.water_sources:
            if not water.boundary:
                water_pos.append(water.state.p_pos - agent.state.p_pos)
        
        # Get the positions of the forests
        forest_pos = []
        for forest in world.forests:
            if not forest.boundary:
                forest_pos.append(forest.state.p_pos - agent.state.p_pos)
        
        # Get the positions of the other preys
        preys = self.preys(world)
        prey_pos = []
        prey_vel = []
        for prey in preys:
            #if agent is not prey:
                # Change the prey's position to be relative to the agent
                prey_pos.append(prey.state.p_pos - agent.state.p_pos)
                prey_vel.append(prey.state.p_vel)

        # Get the positions of the predators
        predators = self.predators(world)
        predator_pos = []
        predator_vel = []
        for predator in predators:
            #if agent is not predator:
                predator_pos.append(predator.state.p_pos - agent.state.p_pos)
                predator_vel.append(predator.state.p_vel)

        # Get if the agent is in a forest
        in_forest = 0
        for forest in world.forests:
            if self.is_collision(agent, forest):
                in_forest = 1

        # Get if the agent is in a water source or food source
        consuming = 0
        for water in world.water_sources:
            if self.is_collision(agent, water):
                consuming = 1
        
        if agent.is_prey:
            for food in world.food_sources:
                if self.is_collision(agent, food):
                    consuming = 1

        # Change the observed positions of the preys that are inside a forest to be in the center of the forest
        for i, prey in enumerate(preys):
            for forest in world.forests:
                if self.is_collision(prey, forest):
                    prey_pos[i] = np.array([forest.state.p_pos[0], forest.state.p_pos[1]])

        # Normalizar el hambre y la sed al rango [0, 1]
        hunger = agent.hunger / 100.0
        thirst = agent.thirst / 100.0

        # Returns the state of the agent, and its environment
        return np.concatenate(
            # Agent information
            [agent.state.p_vel]
            + [agent.state.p_pos]
        
            + [np.array([hunger, thirst])]
            + [np.array([in_forest, consuming])]

            # Entity information
            + obs_pos
            + food_pos
            + water_pos
            + forest_pos

            # Prey and predator information
            + prey_vel
            + prey_pos
            + predator_vel
            + predator_pos
        )