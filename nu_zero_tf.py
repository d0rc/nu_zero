import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RLConfig:
    def __init__(self,
                 observation_size: int,
                 internal_representation_size: int,
                 actions_size: int,
                 reward_size: int,
                 dynamics_hidden_size: int = 128):
        self.observation_size = observation_size
        self.internal_representation_size = internal_representation_size
        self.actions_size = actions_size
        self.reward_size = reward_size
        self.dynamics_hidden_size = dynamics_hidden_size


def make_representation_network(config: RLConfig):
    inputs = keras.Input(shape=(1, config.observation_size))
    x = layers.Dense(units=config.internal_representation_size, activation="relu")(inputs)
    outputs = x
    return keras.Model(inputs, outputs)


def make_dynamics_network(config: RLConfig, internal_representation, action):
    inputs = keras.Input(shape=(1, config.internal_representation_size + config.actions_size))
    x = layers.Dense(units=config.dynamics_hidden_size, activation="relu")(inputs)
    x = layers.Dense(units=config.dynamics_hidden_size, activation="relu")(x)
    dynamics_common_output = layers.Softmax()(x)

    state_head = layers.Dense(units=config.internal_representation_size,
                              activation="tanh")(dynamics_common_output)

    reward_head = layers.Dense(units=config.reward_size,
                               activation="tanh")(dynamics_common_output)

    return keras.Model(inputs,
                       [state_head, reward_head],
                       name="rl-dynamics-network")
