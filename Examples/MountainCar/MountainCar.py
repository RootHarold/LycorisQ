"""An example of MountainCar-v0.

Author: RootHarold
Project: https://github.com/RootHarold/LycorisQ
Related Link: https://gym.openai.com/
"""

from LycorisQ import Agent
import gym
import random
import numpy as np

config = {"capacity": 64, "state_dim": 4, "action_dim": 3, "nodes": 120, "connections": 2000, "depths": 6,
          "batch_size": 72, "memory": 10000, "evolution": 64}
agent = Agent(config)

env = gym.make('MountainCar-v0')
e = 0.9


def pre_processing(data):
    data_ = data.copy()
    data_[0] = (data_[0] + 1.2) / 1.8
    data_[1] = (data_[1] + 0.07) / 0.14

    return data_


for i_episode in range(500):
    observation = list(env.reset())
    last_state = observation
    furthest = -1.2

    for t in range(200):
        # env.render()
        current_state = pre_processing(observation)

        if random.random() < max(e, 0.1):
            action = env.action_space.sample()
        else:
            ret = agent.evaluate(last_state + current_state)[0]
            action = np.argmax(ret)

        observation, _, done, _ = env.step(action)
        observation = list(observation)
        if observation[0] > furthest:
            furthest = observation[0]
        next_state = pre_processing(observation)

        weight = 1.0 if next_state[0] > current_state[0] else 0.6
        reward = abs(next_state[0] - current_state[0]) * weight
        reward += furthest

        if observation[0] >= 0.5:
            reward += 0.5
            done_ = True
        else:
            done_ = False

        agent.train([action, reward, last_state + current_state, current_state + next_state, done_])
        last_state = current_state
        e = e * 0.9995

        if done:
            print("Episode", i_episode + 1, "finished. The furthest position is", furthest)
            break

env.close()
