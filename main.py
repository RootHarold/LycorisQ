from LycorisQ import Agent
import gym
import random
import numpy as np

config = {"capacity": 64, "state_dim": 4, "action_dim": 2, "nodes": 36, "connections": 120, "depths": 3,
          "batch_size": 48, "memory": 200, "evolution": 20}
agent = Agent(config)

env = gym.make('CartPole-v0')
e = 0.5

for i_episode in range(5000):
    observation = env.reset()
    for t in range(100):
        # env.render()
        current_state = observation

        if random.random() < max(e, 0.01):
            action = env.action_space.sample()
        else:
            ret = agent.evaluate(current_state)[0]
            # print(ret)
            action = np.argmax(ret)

        observation, reward, done, _ = env.step(action)
        next_state = observation
        reward_ = -1.0 if done else 0.1
        agent.train([action, reward_, current_state, next_state, done])
        e = e * 0.9995
        if done:
            print(i_episode + 1, "Episode finished after {} timesteps".format(t + 1))
            break

env.close()
