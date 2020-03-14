![logo](https://github.com/RootHarold/LycorisQ/blob/master/logo/logo.svg)

**LycorisQ** is a neat reinforcement learning framework based on [**LycorisNet**](https://github.com/RootHarold/Lycoris).

# Features
* Universal and practical.
* Q network automation design.
* End to end.

# Installation
The project is based on LycorisNet, and the installation of LycorisNet can be found [**here**](https://github.com/RootHarold/Lycoris#Installation).

```
pip install LycorisQ
```

# Documents
The APIs provided by **Agent** (`from LycorisQ import Agent`):

Function | Description |  Inputs | Returns
-|-|-|-
**Agent**(config) | Constructor. | **config**: The configuration information, including 11 configuration fields. | An object of the class Agent.
**train**(data) | Sampling data from the sample pool to train the neural network. | **data**: Each input data has 5 dimensions, which are action, reward, current_state, next_state, and done. | 
**evaluate**(data) | Input observation to evaluate the corresponding strategy. | **data**: Observation. Environmental information. | The scores corresponding to the actions, in the form of a list.
**save**(path1, path2) | Save the model and related configurations. | **path1**: The path to store the model.<br/> **path2**: The path to store the configurations. |
`@staticmethod`<br/>**load**(path1, path2) | Import pre-trained models and related configurations. | **path1**: The path to import the model.<br/> **path2**: The path to import the configurations. |
**set_config**(config) | Set the configuration information of Agent. | **config**: The configuration information, including 11 configuration fields. |
**set_lr**(learning_rate) | Set the learning rate of the neural network. | **learning_rate**: The learning rate of the neural network. | 
**set_workers**(workers) | Set the number of worker threads to train the model. | **workers**: The number of worker threads. | 
`@staticmethod`<br/>**version**() |  |  | Returns the version information of Agent.

Configurable fields:

Field | Description |Default
-|-|-
capacity | Capacity of LycorisNet. |
state_dim | Dimension of the state. |
action_dim | Dimension of the action. | 
nodes | The number of hidden nodes added for each neural network. |
connections| The number of connections added for each neural network. |
depths| Total layers of each neural network. |
batch_size| Batch size. |
memory| The capacity of the sample pool. |
evolution| Number of LycorisNet evolutions. | 0
verbose| Whether to output intermediate information. | False
gamma | A parameter in Q-learning. | 0.9

# Examples
* [**CartPole**](https://github.com/RootHarold/LycorisQ/tree/master/Examples/CartPole)
* [**MountainCar**](https://github.com/RootHarold/LycorisQ/tree/master/Examples/MountainCar)
* *More examples will be released in the future.*

# License
LycorisQ is released under the [LGPL-3.0](https://github.com/RootHarold/LycorisQ/blob/master/LICENSE) license. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.