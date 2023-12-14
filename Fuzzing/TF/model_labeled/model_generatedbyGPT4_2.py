import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        # Initialize necessary layers and configurations
        self.add_layer = tf.keras.layers.Add()
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
        self.layer = tf.keras.layers.Layer()
        self.string_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.initializers.truncated_normal())
        self.used_apis = [
            'tf.keras.layers.experimental.preprocessing.Rescaling',
            'tf.keras.layers.Layer',
            'tf.keras.layers.Dense',
            'tf.clip_by_norm',
            'tf.pad',
            'tf.experimental.numpy.dstack',
            'tf.experimental.numpy.sort',
            'tf.experimental.numpy.exp2',
            'tf.signal.ifftshift',
            'tf.gather_nd'
        ]

    def call(self, inputs):
        # Record input-output of each API

        x = self.rescaling(inputs)

        # Flatten the input
        x = self.flatten(x)

        # Using tf.keras.layers.Dense
        x = self.dense(x)

        # Additional API usages from the list
        x = tf.clip_by_norm(x, 1.0)

        x = tf.pad(x, paddings=[[0, 0], [0, 1]], mode="CONSTANT")

        x = tf.experimental.numpy.dstack([x, x])

        x = tf.experimental.numpy.sort(x)

        x = tf.experimental.numpy.exp2(x)

        x = tf.signal.ifftshift(x)
        # Final output calculation
        param = tf.ragged.constant([[[0], [0]]], ragged_rank=1)

        output = tf.gather_nd(x, param, batch_dims=1)

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted