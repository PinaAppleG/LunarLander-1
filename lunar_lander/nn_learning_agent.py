import operator
import random
from collections import deque

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder


class LearningAgent:
    def __init__(self, replay_size=100000, batch_size=128, gamma=0.995,
                 epsilon=1, min_epsilon=0.05, epsilon_decay_rate=0.995,
                 training_rate=1, training_decay_rate=1, min_training_rate=0,
                 n_explorations=25):
        """
        Learning agent intended to solve the lunar lander environment from
        OpenAI. The agent uses on policy, epsilon greedy Q-Learning backed by a
        neural network with experience replay.

        :param replay_size: replay memory size (default: 100000)
        :param batch_size:  batch training size (default: 128)
        :param gamma: discount factor (default: 0.995)
        :param epsilon: epsilon exploration (default: 1)
        :param min_epsilon: minimum epsilon (default: 0.05)
        :param epsilon_decay_rate: rate at which epsilon decays (default: 0.995)
        :param training_rate: training rate (default: 1)
        :param training_decay_rate: rate at which training decays (default: 1)
        :param min_training_rate: minimum training rate (default: 0)
        :param n_explorations: number of episodes to explore (default: 25)
        """
        # Neural Network
        self.nn = MLPRegressor(warm_start=True)
        self.trained = False

        # Mini batch settings
        self.replay_size = replay_size
        self.replay_memory = deque(maxlen=self.replay_size)
        self.batch_size = batch_size

        # Hyper parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        # Training frequency settings
        self.training_rate = training_rate
        self.training_decay_rate = training_decay_rate
        self.min_training_rate = min_training_rate

        # Number of episodes to purely explore.
        self.n_explorations = n_explorations

        # One hot encode actions
        self.unencoded_actions = [0, 1, 2, 3]
        enc = OneHotEncoder()
        self.actions = enc.fit_transform(
            [[x] for x in self.unencoded_actions]).toarray()

    def act(self, observation):
        """
        Preforms epsilon greedy exploration to determine what action the agent
        should take. If the neural network has yet to be trained, a action
        is randomly sampled.

        :param observation: observation describing the current state of the
        OpenAI Gym environment.
        :return: valid lunar lander action
        """
        # If the neural network is untrained(exploring) sample a random action.
        if not self.trained:
            return random.choice(self.unencoded_actions)

        # Perform epsilon greedy exploration
        elif self.epsilon < random.random():
            return random.choice(self.unencoded_actions)
        else:
            inputs = [np.concatenate((observation, action)) for action in
                      self.actions]
            outputs = self.nn.predict(inputs)
            action, _ = max(enumerate(outputs), key=operator.itemgetter(1))
            return action

    def store_in_replay(self, observation, action, reward, new_observation,
                        done):
        """
        Stores needed information for learning.

        :param observation: observation(state) before the action was taken.
        :param action: action that was taken.
        :param reward: reward received after taking the action.
        :param new_observation: new observation(state) after the action was
        taken.
        :param done: boolean to indicate if the episode has completed.
        :return: None
        """
        # If the episode is done, y should only be the immediate reward, else it
        # should be set to reward plus the discounted sum of expected future
        # rewards.
        if done or not self.trained:
            y = reward
            self.epsilon = max(self.min_epsilon,
                               self.epsilon * self.epsilon_decay_rate)
        else:
            inputs = [np.concatenate((new_observation, a)) for a in
                      self.actions]
            outputs = self.nn.predict(inputs)
            _, value = max(enumerate(outputs), key=operator.itemgetter(1))
            y = reward + self.gamma * value

        # Append observation action pair and y value to memory.
        self.replay_memory.append(
            (np.concatenate((observation, self.actions[action])), y))

        # If episode over, decrement exploring count.
        if done:
            self.n_explorations = max(0, self.n_explorations - 1)

        # If not exploring, train if training_rate is met.
        if self.n_explorations == 0 and random.random() < self.training_rate:
            self.__train()
            self.training_rate = max(self.min_training_rate,
                                     self.training_rate *
                                     self.training_decay_rate)

    def __train(self):
        """
        Sample a mini-batch from the experience replay and train the neural
        network.

        :return: None
        """
        samples = random.sample(self.replay_memory, self.batch_size)
        x = [sample[0] for sample in samples]
        y = [sample[1] for sample in samples]

        self.nn.fit(x, y)
        self.trained = True

    def set_attributes(self, attr_settings):
        """
        Takes a dictionary of attribute settings.
        :param attr_settings: dictionary of attribute settings.
        example: {'training_rate': 0.5, training_decay_rate: 0}
        :return: None
        """
        for key, value in attr_settings.items():
            setattr(self, key, value)

    def stop_training(self):
        """
        Stop training by setting training rate to 0, also set epsilon to 0 to
        prevent any exploration.

        :return: None
        """
        self.epsilon = 0
        self.min_epsilon = 0
        self.min_training_rate = 0
        self.training_rate = 0
