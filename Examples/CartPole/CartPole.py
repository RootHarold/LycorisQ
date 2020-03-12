"""An example of CartPole-v0.

Author: RootHarold
Project: https://github.com/RootHarold/LycorisQ
Related Link: https://gym.openai.com/
"""

from LycorisQ import Agent
import gym
import random
import numpy as np

config = {"capacity": 64, "state_dim": 4, "action_dim": 2, "nodes": 120, "connections": 3000, "depths": 6,
          "batch_size": 48, "memory": 10000, "evolution": 64}
agent = Agent(config)

env = gym.make('CartPole-v0')
e = 0.9


def pre_processing(data):
    data_ = data.copy()
    data_[0] = (data_[0] + 4.9) / 9.8
    data_[1] = 1 / (1 + np.exp(-data_[1]))
    data_[2] = (data_[2] + 0.419) / 0.838
    data_[3] = 1 / (1 + np.exp(-data_[3]))

    return data_


for i_episode in range(300):
    observation = env.reset()

    for t in range(200):
        # env.render()
        current_state = pre_processing(observation)

        if random.random() < max(e, 0.02):
            action = env.action_space.sample()
        else:
            ret = agent.evaluate(current_state)[0]
            action = np.argmax(ret)

        observation, _, done, _ = env.step(action)
        next_state = pre_processing(observation)
        reward = -1.0 if done else 0.1
        agent.train([action, reward, current_state, next_state, done])
        e = e * 0.9995

        if done:
            print("Episode", i_episode + 1, "finished after", t + 1, "timesteps")
            break

env.close()
