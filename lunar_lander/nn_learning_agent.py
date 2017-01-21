import operator
import random
from collections import deque

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class LearningAgent:
    """
    Learning agent intended to solve the lunar lander environment from OpenAI.
    The agent uses on policy, epsilon greedy Q-Learning backed by a neural
    network with experience replay.
    """

    def __init__(self):
        # Neural Network
        self.nn = MLPRegressor(warm_start=True)
        self.trained = False

        # Mini batch settings
        self.replay_size = 20000
        self.replay_data = deque(maxlen=self.replay_size)
        self.batch_size = 50

        # Hyper parameters
        self.gamma = 0.995
        self.eps = 1
        self.min_eps = 0.05
        self.eps_decay = .995

        # Queue of rewards received
        self.rewards = deque(maxlen=100)

        # One hot encode actions
        enc = OneHotEncoder()
        self.unencoded_actions = [0, 1, 2, 3]
        self.actions = enc.fit_transform(
            [[x] for x in self.unencoded_actions]).toarray()

        # Data normalizer
        self.scalar = StandardScaler()
        self.scalar_fitted = False

    def get_action(self, observation):
        """
        Preforms epsilon greedy exploration to determine what action the agent
        should take. If the neural network has yet to be trained, a action
        is randomly sampled.
        :param observation: observation describing the current state of the
        OpenAI Gym environment.
        :return: valid lunar lander action
        """
        # If the neural network is untrained sample a random action.
        if not self.trained:
            return random.choice(self.unencoded_actions)

        # Perform epsilon greedy exploration
        elif random.random() > self.eps:
            inputs = [np.concatenate((observation, action)) for action in
                      self.actions]
            inputs = self.scalar.transform(inputs)
            outputs = self.nn.predict(inputs)
            action, _ = max(enumerate(outputs), key=operator.itemgetter(1))
            return action
        else:
            return random.choice(self.unencoded_actions)

    def store_in_replay(self, observation, action, reward, new_observation,
                        done):
        """
        Stores needed information for learning.
        :param observation: observation(state) before the action was taken.
        :param action: action that was taken.
        :param reward: reward received after taking the action.
        :param new_observation: observation(state) after the action was taken.
        :param done: boolean to indicate if the episode has completed.
        :return: None
        """
        # If the episode is done, y should only be the immediate reward, else it
        # should be set to reward plus the discounted sum of expected future
        # rewards.
        if done or not self.trained:
            y = reward
            self.eps = max(self.min_eps, self.eps * self.eps_decay)
        else:
            inputs = [np.concatenate((new_observation, a)) for a in
                      self.actions]
            inputs = self.scalar.transform(inputs)
            outputs = self.nn.predict(inputs)
            _, value = max(enumerate(outputs), key=operator.itemgetter(1))
            y = reward + self.gamma * value

        self.replay_data.append(
            (np.concatenate((observation, self.actions[action])), y))

    def train(self):
        """
        Sample a mini-batch from the experience replay and training the neural
        network.
        :return: None
        """
        samples = random.sample(self.replay_data, self.batch_size)
        x = [sample[0] for sample in samples]
        y = [sample[1] for sample in samples]

        # Fit the data normalizer if has yet to be fitted.
        if not self.scalar_fitted:
            data = [d[0] for d in samples]
            self.scalar.fit(data)
            self.scalar_fitted = True

        x = self.scalar.transform(x)

        self.nn.fit(x, y)
        self.trained = True

    def no_eps(self):
        """
        Set epsilon to zero, the agent will now always attempt to maximize its
        reward, rather than performing epsilon greedy exploration.
        :return:
        """
        self.eps = 0
        self.min_eps = 0
