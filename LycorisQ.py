"""Copyright information.
Copyright (c) 2020, RootHarold
All rights reserved.
Use of this source code is governed by a LGPL-3.0 license that can be found
in the LICENSE file.
"""

from LycorisNet import Lycoris
from LycorisNet import loadModel
import random
import numpy as np
from collections import deque
import logging
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class Agent:
    def __init__(self, config):
        if config is not None:
            self.__check_config(config)
            self.__config = config
            self.__lie = Lycoris(capacity=config["capacity"], inputDim=config["state_dim"],
                                 outputDim=config["action_dim"], mode="predict")
            self.__lie.setMutateOdds(0)
            self.__lie.preheat(config["nodes"], config["connections"], config["depths"])
            self.__memory = deque(maxlen=config["memory"])
            self.__count = 0

    def train(self, data):
        if np.array(data).ndim == 1:
            data = [data]

        for item in data:
            self.__memory.append(item)

        if self.__config["batch_size"] <= len(self.__memory):
            sample = random.sample(self.__memory, self.__config["batch_size"])
        else:
            sample = random.choices(self.__memory, k=self.__config["batch_size"])

        temp1 = [None] * self.__config["batch_size"]
        temp2 = [None] * self.__config["batch_size"]
        for i in range(self.__config["batch_size"]):
            temp1[i], temp2[i] = self.__process(sample[i])

        if self.__count == self.__config["evolution"]:
            self.__lie.enrich()

        if self.__count < self.__config["evolution"]:
            self.__lie.fitAll(temp1, temp2)
        else:
            self.__lie.fit(temp1, temp2)

        self.__count = self.__count + 1

        if self.__config["verbose"]:
            logging.info("Epoch " + str(self.__count) + " : " + str(self.__lie.getLoss()))

    def evaluate(self, data):
        if np.array(data).ndim == 1:
            data = [data]

        return self.__lie.computeBatch(data)

    def save(self, path1, path2):
        self.__lie.saveModel(path=path1)
        json_info = json.dumps(self.__config, indent=4)
        f = open(path2, 'w')
        f.write(json_info)
        f.close()

        if self.__config["verbose"]:
            logging.info("Model saved successfully.")

    @staticmethod
    def load(path1, path2):
        l_q = Agent(None)
        l_q.__count = 0

        l_q.__lie = loadModel(path1, capacity=1)

        f = open(path2, 'r')
        json_info = f.read()
        f.close()

        config = json.loads(json_info)
        config["capacity"] = 1
        config["evolution"] = 0
        l_q.__check_config(config)
        l_q.__config = config
        l_q.__memory = deque(maxlen=config["memory"])

        if l_q.__config["verbose"]:
            logging.info("Model imported successfully.")

        return l_q

    def set_config(self, config):
        self.__check_config(config)
        self.__config = config

    def set_lr(self, learning_rate):
        self.__lie.setLR(learning_rate)

    def set_workers(self, workers):
        self.__lie.setCpuCores(num=workers)

    @staticmethod
    def version():
        lycoris_version = Lycoris.version()
        return "LycorisQ 0.9.18-Beta By RootHarold." + "\nPowered By " + lycoris_version[:-15] + "."

    @staticmethod
    def __check_config(config):
        keys = ["capacity", "state_dim", "action_dim", "nodes", "connections", "depths", "batch_size", "memory"]
        for item in keys:
            if item not in config:
                raise Exception("Invalid configuration.")

        if "evolution" not in config:
            config["evolution"] = 0

        if "verbose" not in config:
            config["verbose"] = False

        if "gamma" not in config:
            config["gamma"] = 0.9

    def __process(self, data):
        action = data[0]
        reward = data[1]
        current_state = data[2]
        next_state = data[3]
        done = data[4]

        Q = self.__lie.compute(current_state)
        if done:
            target = reward
        else:
            target = reward + self.__config["gamma"] * np.max(self.__lie.compute(next_state))
        Q[action] = target

        return current_state, Q
