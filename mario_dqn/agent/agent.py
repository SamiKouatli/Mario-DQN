#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import os
import random

import numpy as np
import tensorflow as tf
from keras.backend import manual_variable_initialization

from mario_dqn.agent.memory import Memory
from mario_dqn.agent.nn_model import NNModel


class Agent:
    """
    Agent that learns to play a game.
    """

    def __init__(self,
                 action_dim,
                 max_memory_size,
                 learning_rate,
                 exploration_max,
                 exploration_min,
                 exploration_decay,
                 gamma,
                 batch_size,
                 image_size,
                 buffer_size,
                 is_verbose,
                 ):
        """
        Initialize agent.

        Parameters
        ----------
        action_dim: int
            Number of possible actions.
        max_memory_size: int
            Maximum memory size for experience replay.
        learning_rate: float
            Learning rate (alpha) of agent.
        exploration_max: float
            Exploration (epsilon) max of agent.
        exploration_min: float
            Exploration (epsilon) min of agent.
        exploration_decay: float
            Exploration (epsilon) decay rate of agent.
        gamma: float
            Discount factor (gamma) of agent.
        batch_size: int
            Batch size.
        image_size: int
            Image size (image is a square).
        buffer_size: int
            Buffer size.
        is_verbose: bool
            Is verbose enabled.
        """
        # Set parameters
        self.epsilon = exploration_max
        self.gamma = gamma
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay
        self.action_dim = action_dim

        # Force deterministic behaviour
        self._force_deterministic_behaviour()

        # Create model and memory
        self.model = NNModel(output_dim=self.action_dim)
        self.memory = Memory(max_size=max_memory_size)

        # Build and display summary of model
        self.model.build((batch_size, image_size, image_size, buffer_size))

        # Create and set optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=tf.Variable(learning_rate))
        self.model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

        if is_verbose:
            self.model.summary()

    def act(self, state):
        """
        Predict an action based on state.

        Parameters
        ----------
        state: State
            Environment state.

        Returns
        -------
        action: Action
            Action predicted.
        """
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        act_values = self.model.predict(self._reshape_state_one(state), verbose=0)

        return np.argmax(act_values[0])  # Returns action using policy

    def remember(self, state, action, reward, next_state, done):
        """
        Remember an experience.

        Parameters
        ----------
        state: State
            Environment state.
        action: Action
            Agent action chosen.
        reward: float
            Agent reward for action.
        next_state: State
            Environment next state.
        done: bool
            Has the Agent reached the end of Environment.
        """
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def replay(self, batch_size):
        """
        Trains the model using randomly selected experiences in the replay memory

        Parameters
        ----------
        batch_size: int
            Size of batch to learn on.
        """
        # Get samples
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        batch_q_targets = []

        # Get Q values for next_state
        q_next_states = self.model.predict(next_states, verbose=0)

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, batch_size):
            done = dones[i]

            # If we are in a done state, only equals reward
            if done:
                batch_q_targets.append(rewards[i])
            else:
                target = rewards[i] + self.gamma * np.max(q_next_states[i])
                batch_q_targets.append(target)

        # 1. Use the current model to output the Q-value predictions
        q_target = self.model.predict(states, verbose=0)

        # 2. Rewrite the chosen action value with the computed target
        for i in range(batch_size):
            q_target[i][actions[i]] = batch_q_targets[i]

        # 3. Use vectors in the objective computation
        self.model.fit(x=states,
                       y=q_target,
                       epochs=1,
                       verbose=0)

        # Reduce exploration rate if possible
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads a saved model.

        Parameters
        ----------
        name: str
            Model path.
        """
        # FIXME(SKO): Model loading is defective with both CPU and GPU. Model looses all its performance...
        #  Problem comes from Tensorflow/Keras.
        print("ERROR : Loading model is not working at the moment because of Tensoflow/Keras issue.")
        self.model.load_weights(filepath=name)

    def save(self, name):
        """
        Save a saved model.

        Parameters
        ----------
        name: str
            Model path.
        """
        # FIXME(SKO): model.save() fails with model subclass.
        #  Save_weights fails to load in tf saved format with model subclass.
        #  Problem comes from Tensorflow/Keras.
        self.model.save_weights(filepath=name, save_format="h5")

    @staticmethod
    def _reshape_state_one(state):
        """
        Reshape the state to predict action for only one step.

        Parameters
        ----------
        state: State
            Current state.

        Returns
        ----------
        state: State
            Reshaped state.
        """
        return state.reshape((1, *state.shape))

    @staticmethod
    def _force_deterministic_behaviour(force_cpu = False):
        """
        Force deterministic behaviour
        """
        # FIXME(SKO): This function was suppose to fix load() error
        #  but theses fixes do not seem to work at the moment with GPU nor CPU
        #  Problem comes from Tensorflow/Keras. Try again when problem is fixed !
        # Fix seeding of np, tf and keras
        #os.environ['PYTHONHASHSEED'] = str(1)
        #random.seed(1)
        #np.random.seed(1)
        #tf.random.set_seed(1)

        # Set Tensorflow manual variable initialization
        #manual_variable_initialization(True)

        # Force TF and TF_CUDNN as deterministic
        #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        #os.environ['TF_DETERMINISTIC_OPS'] = '1'
        #os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

        # Force CPU
        #if force_cpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
