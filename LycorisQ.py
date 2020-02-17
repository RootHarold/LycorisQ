"""Copyright information.
Copyright (c) 2020, RootHarold
All rights reserved.
Use of this source code is governed by a LGPL-3.0 license that can be found
in the LICENSE file.
"""

from LycorisNet import Lycoris
from LycorisNet import loadModel
import math
import random
import numpy as np
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
            self.__flag = True

    def train(self, data):
        if np.array(data).ndim == 1:
            data = [data]

        flag = True
        if self.__flag:
            self.__flag = False
        else:
            flag = False

        data_copy = data.copy()
        batch = math.ceil(len(data) / float(self.__config["batch_size"]))
        remainder = len(data) % self.__config["batch_size"]

        for _ in range(remainder):
            data_copy.append(random.choice(data))

        for i in range(self.__config["epoch"]):
            random.shuffle(data_copy)
            temp1 = [None] * self.__config["batch_size"]
            temp2 = [None] * self.__config["batch_size"]
            pos = 0

            for j in range(batch):
                for k in range(self.__config["batch_size"]):
                    temp1[k], temp2[k] = self.__process(data_copy[pos])
                    pos = pos + 1

                if flag:
                    if i * batch + j == self.__config["evolution"]:
                        self.__lie.enrich()

                    if i * batch + j < self.__config["evolution"]:
                        self.__lie.fitAll(temp1, temp2)
                    else:
                        self.__lie.fit(temp1, temp2)
                else:
                    self.__lie.fit(temp1, temp2)

            if self.__config["verbose"]:
                logging.info("Epoch " + str(i + 1) + " : " + str(self.__lie.getLoss()))

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
        l_q.__flag = False

        l_q.__lie = loadModel(path1, capacity=1)

        f = open(path2, 'r')
        json_info = f.read()
        f.close()

        config = json.loads(json_info)
        config["capacity"] = 1
        config["evolution"] = 0
        l_q.__check_config(config)
        l_q.__config = config
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
        return "LycorisQ 1.0.0 By RootHarold." + "\nPowered By " + lycoris_version[:-15] + "."

    @staticmethod
    def __check_config(config):
        keys = ["capacity", "state_dim", "action_dim", "nodes", "connections", "depths", "batch_size", "epoch"]
        for item in keys:
            if item not in config:
                raise Exception("Invalid configuration.")

        if "evolution" not in config:
            config["evolution"] = 0

        if "verbose" not in config:
            config["verbose"] = False

        if "alpha" not in config:
            config["alpha"] = 0.01

        if "gamma" not in config:
            config["gamma"] = 0.5

    def __process(self, data):
        action = data[0]
        reward = data[1]
        current_state = data[2]
        next_state = data[3]

        Q = self.__lie.compute(current_state)
        delta = (reward + self.__config["gamma"] * max(self.__lie.compute(next_state)) - Q[action]) * self.__config[
            "alpha"]
        Q[action] = Q[action] + delta

        return current_state, Q
