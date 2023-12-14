import tensorflow as tf
import numpy as np

class PreprocessAndCalculateModel(tf.keras.Model):
    def __init__(self):
        super(PreprocessAndCalculateModel, self).__init__()
        # Initialize necessary layers and configurations
        self.conv3d = tf.keras.layers.Conv3D(32, (3, 3, 3), activation=tf.nn.relu)
        self.resizing = tf.keras.layers.Resizing(3, 3)
        self.simple_rnn_cell = tf.keras.layers.SimpleRNNCell(32)
        self.separable_conv2d = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu')
        self.dense = tf.keras.layers.Dense(10, activation='linear', kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, inputs):
        # Record input-output of each API
        api_outputs = {}
        self.used_apis = []

        # Using tf.keras.layers.Conv3D
        x = tf.reshape(inputs, [4, 3, 4, 4, 1])  # Reshape for Conv3D
        x = self.conv3d(x)
        self.used_apis.append('tf.keras.layers.Conv3D')

        # Using tf.keras.layers.Resizing
        x = tf.reshape(x, [8, 4, 4, 4])  # Reshape for Resizing
        x = self.resizing(x)
        self.used_apis.append('tf.keras.layers.Resizing')

        # Using tf.keras.layers.SeparableConv2D
        x = self.separable_conv2d(x)
        self.used_apis.append('tf.keras.layers.SeparableConv2D')

        # Using tf.keras.layers.Dense
        x = self.dense(x)
        self.used_apis.append('tf.keras.layers.Dense')

        # Additional API usages from the list
        x = tf.math.cumprod(x, axis=1)
        self.used_apis.append('tf.math.cumprod')

        x = tf.math.asin(x)
        self.used_apis.append('tf.math.asin')

        x = tf.linalg.diag_part(x)
        self.used_apis.append('tf.linalg.diag_part')

        x = tf.math.special.bessel_i0e(x)
        self.used_apis.append('tf.math.special.bessel_i0e')

        x = tf.math.special.bessel_i1e(x)
        self.used_apis.append('tf.math.special.bessel_i1e')

        x = tf.experimental.numpy.cbrt(x)
        self.used_apis.append('tf.experimental.numpy.cbrt')

        x = tf.experimental.numpy.outer(x, x)
        self.used_apis.append('tf.experimental.numpy.outer')

        # Add more APIs from the list here

        # Final output calculation
        output = x  # Final operation

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
