"""Copyright information.
Copyright (c) 2020, RootHarold
All rights reserved.
Use of this source code is governed by a LGPL-3.0 license that can be found
in the LICENSE file.
"""

from LycorisNet import Lycoris
from LycorisNet import loadModel
import json
import logging

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
            self.__mapping = {}
            self.__flag = True

    def train(self, data):
        pass

    def evaluate(self, data):
        pass

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
        pass

    def set_config(self, config):
        pass

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
        pass
