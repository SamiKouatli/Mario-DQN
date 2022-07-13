#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import keras
import tensorflow as tf


class NNModel(keras.Model):
    """
    Convolutional neural network made with Keras.
    """

    def __init__(self, output_dim):
        """
        Initialize convolutional model.

        Parameters
        ----------
        output_dim: int
            Output dimension
        """
        super(NNModel, self).__init__()
        self._output_dim = output_dim

        # First layer CONV2D with ELU
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(8, 8),
                                            strides=(4, 4),
                                            padding="VALID",
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            name="conv1")
        self.conv1_act = tf.keras.layers.Activation(activation=tf.keras.activations.elu, name="conv1_act")

        # Second layer CONV2D with ELU
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding="VALID",
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            name="conv2")
        self.conv2_act = tf.keras.layers.Activation(activation=tf.keras.activations.elu, name="conv2_act")

        # Third layer CONV2D with ELU
        self.conv3 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="VALID",
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            name="conv3")
        self.conv3_act = tf.keras.layers.Activation(activation=tf.keras.activations.elu, name="conv3_act")

        # Fully connected layer
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=512,
                                         kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                         name="fc1")
        self.fc1_act = tf.keras.layers.Activation(activation=tf.keras.activations.elu, name="fc1_act")
        self.fc2 = tf.keras.layers.Dense(units=self._output_dim,
                                         kernel_initializer=tf.keras.initializers.GlorotNormal())

    def call(self, inputs, **kwargs):
        """
        Function called on new inputs and returns the outputs as tensors.

        Parameters
        ----------
        inputs: input tensor|dict|list
            Model data input.
        kwargs : keyword arguments
            Arbitrary keyword arguments.

        Returns
        -------
        output :
            Model prediction.
        """
        # First layer
        x = self.conv1(inputs)
        x = self.conv1_act(x)

        # Second layer
        x = self.conv2(x)
        x = self.conv2_act(x)

        # Third layer
        x = self.conv3(x)
        x = self.conv3_act(x)

        # Fully connected layer
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        output = self.fc2(x)

        return output
