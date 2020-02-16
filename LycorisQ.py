"""Copyright information.
Copyright (c) 2020, RootHarold
All rights reserved.
Use of this source code is governed by a LGPL-3.0 license that can be found
in the LICENSE file.
"""

from LycorisNet import Lycoris
from LycorisNet import loadModel
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class Agent:
    def __init__(self, config):
        pass

    def save(self, path1, path2):
        pass

    @staticmethod
    def load(path1, path2):
        pass

    def set_config(self, config):
        pass

    def set_lr(self, learning_rate):
        pass

    def set_workers(self, workers):
        pass

    @staticmethod
    def version():
        pass

    @staticmethod
    def __check_config(config):
        pass
