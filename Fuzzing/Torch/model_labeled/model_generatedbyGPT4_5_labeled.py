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
        self.used_apis = [
            'tf.reshape',
            'tf.keras.layers.Convolution3D',
            'tf.keras.layers.Dense',
            'tf.keras.layers.Resizing',
            'tf.keras.layers.SeparableConv2D',
            'tf.math.cumprod',
            'tf.math.asin',
            'tf.linalg.diag_part',
            'tf.math.special.bessel_i0e',
            'tf.math.special.bessel_i1e',
            'tf.experimental.numpy.cbrt',
            'tf.experimental.numpy.outer'
        ]

    def call(self, inputs):
        api_outputs = {}

        api_outputs['pre_tf.reshape(inputs, [4, 3, 4, 4, 1])'] = inputs
        x = tf.reshape(inputs, [4, 3, 4, 4, 1])  # Reshape for Conv3D
        api_outputs['tf.reshape(inputs, [4, 3, 4, 4, 1])'] = x

        api_outputs['pre_self.conv3d(x)'] = x
        x = self.conv3d(x)
        api_outputs['self.conv3d(x)'] = x

        api_outputs['pre_tf.reshape(x, [8, 4, 4, 4])'] = x
        x = tf.reshape(x, [8, 4, 4, 4])  # Reshape for Resizing
        api_outputs['tf.reshape(x, [8, 4, 4, 4])'] = x

        api_outputs['pre_self.resizing(x)'] = x
        x = self.resizing(x)
        api_outputs['self.resizing(x)'] = x

        api_outputs['pre_self.separable_conv2d(x)'] = x
        x = self.separable_conv2d(x)
        api_outputs['self.separable_conv2d(x)'] = x

        api_outputs['pre_self.dense(x)'] = x
        x = self.dense(x)
        api_outputs['self.dense(x)'] = x

        api_outputs['pre_tf.math.cumprod(x, axis=1)'] = x
        x = tf.math.cumprod(x, axis=1)
        api_outputs['tf.math.cumprod(x, axis=1)'] = x

        api_outputs['pre_tf.math.asin(x)'] = x
        x = tf.math.asin(x)
        api_outputs['tf.math.asin(x)'] = x

        api_outputs['pre_tf.linalg.diag_part(x)'] = x
        x = tf.linalg.diag_part(x)
        api_outputs['tf.linalg.diag_part(x)'] = x

        api_outputs['pre_tf.math.special.bessel_i0e(x)'] = x
        x = tf.math.special.bessel_i0e(x)
        api_outputs['tf.math.special.bessel_i0e(x)'] = x

        api_outputs['pre_tf.math.special.bessel_i1e(x)'] = x
        x = tf.math.special.bessel_i1e(x)
        api_outputs['tf.math.special.bessel_i1e(x)'] = x

        api_outputs['pre_tf.experimental.numpy.cbrt(x)'] = x
        x = tf.experimental.numpy.cbrt(x)
        api_outputs['tf.experimental.numpy.cbrt(x)'] = x

        api_outputs['pre_tf.experimental.numpy.outer(x, x)'] = x
        x = tf.experimental.numpy.outer(x, x)
        api_outputs['tf.experimental.numpy.outer(x, x)'] = x

        # Final output calculation
        output = x  # Final operation

        # List of used APIs
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
