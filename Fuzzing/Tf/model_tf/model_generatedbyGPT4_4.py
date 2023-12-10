import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        # Initialize necessary layers and configurations
        self.conv3d = tf.keras.layers.Convolution3D(32, (3, 3, 3), activation=tf.keras.activations.selu)
        self.dense = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.keras.initializers.Identity())
        self.used_apis = [
            'tf.reshape',
            'tf.keras.layers.Convolution3D',
            'tf.keras.layers.Dense',
            'tf.acos',
            'tf.math.exp',
            'tf.experimental.numpy.count_nonzero',
            'tf.experimental.numpy.fabs',
            'tf.math.reduce_any'
        ]

    def call(self, inputs):

        # Using tf.nn.conv3d
        x = tf.reshape(inputs, [4, 3, 4, 4, 1])  # Reshape for conv3d
        x = self.conv3d(x)

        x = self.dense(x)

        x = tf.acos(x)

        x = tf.math.exp(x)

        x = tf.experimental.numpy.count_nonzero(x)

        x = tf.experimental.numpy.fabs(x)

        x = tf.math.reduce_any(tf.math.is_non_decreasing(x))

        # Final output calculation
        output = x  # Final operation

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
